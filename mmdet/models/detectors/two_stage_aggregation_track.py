import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmdet.models.plugins.conv_lstm import ConvLSTM
from mmdet.core import bbox2result, bbox_overlaps, bbox2result_with_id, bbox2roi, build_assigner, build_sampler
from .. import builder
from ..builder import DETECTORS
from .base import BaseDetector
from mmdet.models.roi_heads.test_mixins import BBoxTestMixin, MaskTestMixin, RPNTestMixin
import lap
import cv2
import mmcv
from mmcv.image import tensor2imgs
import pycocotools.mask as mask_util


@DETECTORS.register_module()
class TwoStageDetectorAggregationTrack(BaseDetector, RPNTestMixin, BBoxTestMixin,
                                       MaskTestMixin):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(TwoStageDetectorAggregationTrack, self).__init__()
        self.backbone = builder.build_backbone(backbone)

        if neck is not None:
            self.neck = builder.build_neck(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = builder.build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            self.roi_head = builder.build_head(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        # memory queue for testing
        self.prev_bboxes = None
        self.prev_sem = None
        self.prev_roi_feats = None
        # add frame id
        self.frame_id = 0
        self.frame_record = []
        # add first and second features
        self.first_feature = None
        self.second_feature = None

        self.init_weights(pretrained=pretrained)

        self.fpn_lstms = nn.ModuleList([ConvLSTM(256, 256, 3) for _ in range(5)])

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def init_weights(self, pretrained=None):
        super(TwoStageDetectorAggregationTrack, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_roi_head:
            self.roi_head.init_weights(pretrained)

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck
        """
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmedetection/tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs,)
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs

    def forward_train(self,
                      img,
                      img_meta,
                      ref_img,
                      agg_before_img,
                      gt_bboxes,
                      gt_bboxes_ignore,
                      ref_bboxes,
                      gt_labels,
                      gt_pids,
                      gt_masks=None,
                      proposals=None):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            ref_img: the reference images of img

            img_meta (list[dict]): list of image info dict where each dict has:
                'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            agg_before_img: imgs before the current img, as the sequence
                input of convLSTM

            gt_bboxes (list[Tensor]): each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            ref_bboxes : boxes of reference images

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_pids: the match between the boxes of img and the boxes of ref_img.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # concat image in the batch dimension
        batch = img.shape[0]
        num_img_before = len(agg_before_img)
        agg_before_img = torch.cat(agg_before_img, dim=0)
        all_img = torch.cat([agg_before_img, img, ref_img])
        all_feat_x = self.extract_feat(all_img)

        # fpn feat
        x_feat = []
        ref_x_feat = []
        agg_before_x_feat = []
        for i in range(len(all_feat_x)):
            img_ind = num_img_before * batch
            ref_img = num_img_before * batch + batch
            agg_before_x_feat.append(all_feat_x[i][:img_ind])
            x_feat.append(all_feat_x[i][img_ind:ref_img])
            ref_x_feat.append(all_feat_x[i][ref_img:])

        # lstm for each fpn layer
        lstm_fusion_x = []
        for i in range(len(all_feat_x)):
            x_i = x_feat[i].unsqueeze(1)
            channel = all_feat_x[i].shape[1]
            height = all_feat_x[i].shape[2]
            weight = all_feat_x[i].shape[3]

            agg_before_x_i = agg_before_x_feat[i].unsqueeze(1)
            agg_before_x_i = agg_before_x_i.reshape(batch, num_img_before, channel, height, weight)
            x_all_i = torch.cat([agg_before_x_i, x_i], dim=1)
            x_all_i = self.fpn_lstms[i](x_all_i).contiguous()
            lstm_fusion_x.append(x_all_i)

        x = lstm_fusion_x
        ref_x = ref_x_feat
        losses = {}

        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_meta,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_meta, x_feat, ref_x,
                                                 proposal_list,
                                                 gt_bboxes, gt_bboxes_ignore,
                                                 ref_bboxes, gt_labels,
                                                 gt_pids, gt_masks)
        losses.update(roi_losses)

        return losses

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.async_test_rpn(x, img_meta,
                                                      self.test_cfg.rpn)
        else:
            proposal_list = proposals

        det_bboxes, det_labels = await self.async_test_bboxes(
            x, img_meta, proposal_list, self.test_cfg.rcnn, rescale=rescale)

        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = await self.async_test_mask(
                x,
                img_meta,
                det_bboxes,
                det_labels,
                rescale=rescale,
                mask_test_cfg=self.test_cfg.get('mask'))
            return bbox_results, segm_results

    def simple_test_bboxes(self,
                           x,
                           x_feat,
                           img_meta,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        """ Test only det bboxes without augmentation"""
        rois = bbox2roi(proposals)
        roi_feats = self.roi_head.bbox_roi_extractor(
            x[:len(self.roi_head.bbox_roi_extractor.featmap_strides)], rois)
        cls_score, bbox_pred = self.roi_head.bbox_head(roi_feats)
        img_shape = img_meta[0]['img_shape']
        scale_factor = img_meta[0]['scale_factor']
        is_first = img_meta[0]['is_first']
        if is_first:
            self.frame_id = 0
        else:
            self.frame_id += 1
        det_bboxes, det_labels = self.roi_head.bbox_head.get_bboxes(
            rois,
            cls_score,
            bbox_pred,
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg)
        if det_bboxes.nelement() == 0:
            det_obj_ids = np.array([], dtype=np.int64)
            if is_first:
                self.prev_bboxes = None
                self.prev_sem = None
                self.prev_roi_feats = None
                self.prev_det_labels = None
                self.frame_record = []
            return det_bboxes, det_labels, det_obj_ids

        res_det_bboxes = det_bboxes.clone()
        if rescale:
            res_det_bboxes[:, :4] *= scale_factor

        det_rois = bbox2roi([res_det_bboxes])
        track_roi_feats = self.roi_head.track_roi_extractor(
            x_feat[:self.roi_head.track_roi_extractor.num_inputs], det_rois)
        sem_feats = self.roi_head.mask_head(track_roi_feats, True)
        # add bbox feature
        if not is_first and self.prev_bboxes is not None:
            det_rois_input = det_rois.clone()[:, 1: 5]
            res_prev_bboxes = self.prev_bboxes.clone()
            if rescale:
                res_prev_bboxes[:, :4] *= scale_factor
            prev_rois_input = res_prev_bboxes[:, 0: 4]

        # recompute bbox match feature

        if is_first or (not is_first and self.prev_bboxes is None):
            det_obj_ids = np.arange(det_bboxes.size(0))
            # save bbox and features for later matching
            self.prev_bboxes = det_bboxes
            self.prev_roi_feats = track_roi_feats
            self.prev_det_labels = det_labels
            self.prev_sem = sem_feats
            # record frame id
            self.frame_record = [self.frame_id] * det_bboxes.size(0)
        else:
            assert self.prev_roi_feats is not None
            # only support one image at a time
            bbox_img_n = [det_bboxes.size(0)]
            prev_bbox_img_n = [self.prev_roi_feats.size(0)]
            match_score = self.roi_head.track_head(track_roi_feats, self.prev_roi_feats,
                                          bbox_img_n, prev_bbox_img_n, det_rois_input, prev_rois_input, sem_feats,
                                          self.prev_sem)[0]
            # remove miss object
            cost_matrix = match_score.cpu().numpy()
            miss_id = np.where(self.frame_id - np.array(self.frame_record) > 15)
            cost_matrix[:, miss_id] = np.inf
            # match with linear assignment, Hungarian algorithm
            cost, x, y = lap.lapjv(cost_matrix, True, 0.8)
            det_obj_ids = np.ones((x.shape[0]), dtype=np.int32) * (-1)
            # first match with match_score
            for idx, match_id in enumerate(x):
                if match_id >= 0:
                    det_obj_ids[idx] = match_id
                    # update feature smooth
                    alpha = 0.5
                    self.prev_roi_feats[match_id] = track_roi_feats[idx] * (1 - alpha) + self.prev_roi_feats[
                        match_id] * alpha
                    self.prev_sem[match_id] = sem_feats[idx] * (1 - alpha) + self.prev_sem[match_id] * alpha
                    self.prev_bboxes[match_id] = det_bboxes[idx]
                    # update frame id
                    self.frame_record[match_id] = self.frame_id
            # second match with iou
            unmatched_det = np.where(x < 0)[0]
            unmatched_prev = np.where(y < 0)[0]
            bbox_ious = bbox_overlaps(det_bboxes[unmatched_det, :4], self.prev_bboxes[unmatched_prev, :4])
            cost_matrix = 1 - bbox_ious.cpu().numpy()
            if cost_matrix.size > 0:
                cost, x, y = lap.lapjv(cost_matrix, True, 0.9)
            else:
                x = [-1] * unmatched_det.shape[0]
            for idx, match_id in enumerate(x):
                det_id = unmatched_det[idx]
                if match_id == -1:
                    # add new object
                    det_obj_ids[det_id] = self.prev_roi_feats.size(0)
                    self.prev_roi_feats = torch.cat((self.prev_roi_feats, track_roi_feats[det_id][None]), dim=0)
                    self.prev_sem = torch.cat((self.prev_sem, sem_feats[det_id][None]), dim=0)
                    self.prev_bboxes = torch.cat((self.prev_bboxes, det_bboxes[det_id][None]), dim=0)
                    # add new frame id
                    self.frame_record.append(self.frame_id)
                else:
                    r_match_id = unmatched_prev[match_id]
                    det_obj_ids[unmatched_det[idx]] = r_match_id
                    # update feature smooth
                    alpha = 0.5
                    self.prev_roi_feats[r_match_id] = track_roi_feats[det_id] * (1 - alpha) + self.prev_roi_feats[
                        r_match_id] * alpha
                    self.prev_sem[r_match_id] = sem_feats[det_id] * (1 - alpha) + self.prev_sem[r_match_id] * alpha
                    self.prev_bboxes[r_match_id] = det_bboxes[det_id]
                    # update frame id
                    self.frame_record[r_match_id] = self.frame_id
        return det_bboxes, det_labels, det_obj_ids

    def simple_test(self, img, img_meta, agg_before_img, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."
        assert self.with_track, "Track head must be implemented"

        x_feat = self.extract_feat(img)
        is_first = img_meta[0]['is_first']
        if is_first:
            self.first_feature = x_feat
            self.second_feature = x_feat
        # lstm for each fpn layer
        lstm_fusion_x = []
        for i in range(len(x_feat)):
            x_i = x_feat[i].unsqueeze(1)
            x_all_i = torch.cat([self.first_feature[i].unsqueeze(1), self.second_feature[i].unsqueeze(1), x_i], dim=1)
            x_all_i = self.fpn_lstms[i](x_all_i).contiguous()
            lstm_fusion_x.append(x_all_i)
        # update first and second
        self.first_feature = self.second_feature
        self.second_feature = lstm_fusion_x

        x = lstm_fusion_x

        if proposals is None:
            proposal_list = self.simple_test_rpn(x, img_meta,
                                                 self.test_cfg.rpn)
        else:
            proposal_list = proposals

        det_bboxes, det_labels, det_obj_ids = self.simple_test_bboxes(
            x, x_feat, img_meta, proposal_list, self.test_cfg.rcnn, rescale=rescale)
        print(det_bboxes.shape)
        print(det_obj_ids)

        bbox_results = bbox2result_with_id(det_bboxes, det_labels, det_obj_ids,
                                           self.roi_head.bbox_head.num_classes)
        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_meta, [det_bboxes], [det_labels],
                det_obj_ids=[det_obj_ids], rescale=rescale)
            return bbox_results, segm_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        proposal_list = self.aug_test_rpn(
            self.extract_feats(imgs), img_metas, self.test_cfg.rpn)
        det_bboxes, det_labels = self.aug_test_bboxes(
            self.extract_feats(imgs), img_metas, proposal_list,
            self.test_cfg.rcnn)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= img_metas[0][0]['scale_factor']
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(
                self.extract_feats(imgs), img_metas, det_bboxes, det_labels)
            return bbox_results, segm_results
        else:
            return bbox_results

    # 重构函数，加入video、四边形
    def show_result(self,
                    data,
                    result,
                    img_norm_cfg,
                    score_thr=0.3,
                    bbox_color=(72, 101, 241),
                    text_color=(72, 101, 241),
                    mask_color=None,
                    thickness=2,
                    font_size=13,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_dir=None,
                    is_video=False):
        """
        info:
            Please refer to the class:BaseDetector.show_result for parameter introduction.
        add:
            1、video show
            2、change rectangular display to quadrilateral display
        """
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None

        img_tensor = data['img'][0]
        img_metas = data['img_meta'][0].data[0]
        imgs = tensor2imgs(img_tensor, **img_norm_cfg)

        if isinstance(bbox_result, dict) and len(bbox_result.keys()) == 0:
            return

        segm_result = segm_result[0]
        # if segm_result is list, it means that the dataset is not video
        if isinstance(segm_result, list):
            is_video = False

        for img, img_meta in zip(imgs, img_metas):
            h, w, _ = img_meta['img_shape']
            img_show = img[:h, :w, :]
            img_show = mmcv.imresize(img_show, (img_meta['ori_shape'][1], img_meta['ori_shape'][0]))
            bboxes = np.empty((0, 9), dtype=np.float64)
            boxes_score = []
            obj_ids = []
            if not is_video:
                for segm_res_i in segm_result:
                    if len(segm_res_i) != 0:
                        for box_rle in segm_res_i:
                            box = list(map(float, box_rle['counts'].split(' ')))
                            box = np.array(box)
                            boxes_score.append(box[-1])
                            bboxes = np.concatenate(bboxes, box[np.newaxis, :])
            else:
                for box_id, box_rle in segm_result.items():
                    obj_ids.append(box_id)
                    box = list(map(float, box_rle['counts'].split(' ')))
                    box = np.array(box)
                    boxes_score.append(box[-1])
                    bboxes = np.concatenate((bboxes, box[np.newaxis, :]))

            # if out_file specified, do not show image in window
            if out_dir is not None:
                show = False
                save_path = '{}/{}/{}.png'.format(out_dir, img_meta['video_id'], img_meta['frame_id'])
            else:
                show = True
                save_path = None

            self.imshow_seg_bboxes(img_show, bboxes, obj_ids, show=show, text_color='white', out_file=save_path)

    def imshow_seg_bboxes(self,
                          img,
                          bboxes,
                          boxes_id,
                          score_thr=0,
                          bbox_color='green',
                          text_color='green',
                          thickness=1,
                          font_scale=0.5,
                          show=True,
                          win_name='',
                          wait_time=0,
                          out_file=None):
        """Draw bboxes and class labels (with scores) on an image.

        Args:
            img (str or ndarray): The image to be displayed.
            bboxes (ndarray): Bounding boxes (with scores), shaped (n, 9)
            score_thr (float): Minimum score of bboxes to be shown.
            bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
            text_color (str or tuple or :obj:`Color`): Color of texts.
            thickness (int): Thickness of lines.
            font_scale (float): Font scales of texts.
            show (bool): Whether to show the image.
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
            out_file (str or None): The filename to write the image.
        """
        assert bboxes.ndim == 2
        assert bboxes.shape[1] == 9
        img = mmcv.image.imread(img)

        if score_thr > 0:
            scores = bboxes[:, -1]
            inds = scores > score_thr
            bboxes = bboxes[inds, :]

        bbox_color = mmcv.color_val(bbox_color)
        text_color = mmcv.color_val(text_color)

        for bbox, id in zip(bboxes, boxes_id):
            bbox_int = bbox[:-1].reshape(-1, 2).astype(np.int32)
            cv2.polylines(img, [bbox_int], True, bbox_color, thickness=thickness)
            label_text = 'id:{}'.format(id)
            cv2.putText(img, label_text, (bbox_int[3, 0], bbox_int[3, 1] - 2),
                        cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)

        if show:
            mmcv.imshow(img, win_name, wait_time)
        if out_file is not None:
            mmcv.image.imwrite(img, out_file)

        if not (show or out_file):
            return img
