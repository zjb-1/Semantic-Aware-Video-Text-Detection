import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmdet.core import bbox2result, bbox_overlaps, bbox2result_with_id, bbox2roi, build_assigner, build_sampler, bbox2roi_global
from .. import builder
from ..builder import DETECTORS
from .base import BaseDetector
from mmdet.models.roi_heads.test_mixins import BBoxTestMixin, MaskTestMixin, RPNTestMixin
import lap
from mmdet.models.roi_heads import KalmanFilter
import mmcv


@DETECTORS.register_module()
class TwoStageDetectorTrack(BaseDetector, RPNTestMixin, BBoxTestMixin,
                       MaskTestMixin):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 shared_head=None,
                 rpn_head=None,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 track_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 track_roi_extractor=None):
        super(TwoStageDetectorTrack, self).__init__()
        self.backbone = builder.build_backbone(backbone)

        if neck is not None:
            self.neck = builder.build_neck(neck)

        if shared_head is not None:
            self.shared_head = builder.build_shared_head(shared_head)

        if rpn_head is not None:
            self.rpn_head = builder.build_head(rpn_head)

        if bbox_head is not None:
            self.bbox_roi_extractor = builder.build_roi_extractor(
                bbox_roi_extractor)
            self.bbox_head = builder.build_head(bbox_head)
        if track_head is not None:
            self.track_roi_extractor = builder.build_roi_extractor(
                track_roi_extractor)
            self.track_head = builder.build_head(track_head)

        if mask_head is not None:
            if mask_roi_extractor is not None:
                self.mask_roi_extractor = builder.build_roi_extractor(
                    mask_roi_extractor)
                self.share_roi_extractor = False
            else:
                self.share_roi_extractor = True
                self.mask_roi_extractor = self.bbox_roi_extractor
            self.mask_head = builder.build_head(mask_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        # memory queue for testing
        self.prev_bboxes = None
        self.prev_sem = None
        self.prev_top = None
        self.prev_roi_feats = None
        # add kalman
        self.prev_mean = []
        self.prev_covariance = []
        self.kalman_filter = KalmanFilter() 
        # add frame id
        self.frame_id = 0
        self.frame_record = []
        self.init_weights(pretrained=pretrained)
        # add res file
        self.res_file = mmcv.load('/lustre/home/wfeng/detection_predict.json')

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    def init_weights(self, pretrained=None):
        super(TwoStageDetectorTrack, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()
        if self.with_mask:
            self.mask_head.init_weights()
            if not self.share_roi_extractor:
                self.mask_roi_extractor.init_weights()
        if self.with_track:
            self.track_roi_extractor.init_weights()
            self.track_head.init_weights()

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
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).cuda()
        # bbox head
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            cls_score, bbox_pred = self.bbox_head(bbox_feats)
            outs = outs + (cls_score, bbox_pred)
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], mask_rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
            mask_pred = self.mask_head(mask_feats)
            outs = outs + (mask_pred, )
        return outs

    def get_topology_feats(self, rois, gt_rois, gt_bbox_feats, batch_num): 
        #### extract topology features for text tracking
        #### rois shape (n, 5), [batch_ind, x1, y1, x2, y2]
        #### gt_roi ground truth (m, 5)
        #### gt_bbox_feats [m, channel, width, height]
        #### coeff_matrix [batch_n, batch_m]
        #### return topology_feats [n, channels * width * height]
        rois_center = torch.stack((rois[:, 0], (rois[:, 1] + rois[:, 3]) / 2, (rois[:, 2] + rois[:, 4]) / 2), 1)
        gt_center = torch.stack((gt_rois[:, 0], (gt_rois[:, 1] + gt_rois[:, 3]) / 2, (gt_rois[:, 2] + gt_rois[:, 4]) / 2), 1)
        gt_bbox_feats = gt_bbox_feats.view(gt_bbox_feats.size(0), -1)
        topology_feats = torch.zeros((rois.shape[0], gt_bbox_feats.shape[1])).to(rois.device)
        #### calulate dis in each batch_ind, and generate feats 
        for i in range(batch_num):
            rois_i = rois_center[rois[:, 0] == i, 1::]
            gt_rois_i = gt_center[gt_rois[:, 0] == i, 1::]
            coeff_matrix = torch.zeros((rois_i.shape[0], gt_rois_i.shape[0])).to(rois_i.device)
            for gt_id in range(gt_rois_i.shape[0]):
               dis = F.pairwise_distance(rois_i, gt_rois_i[gt_id,:], p=2)
               coeff_matrix[:, gt_id] = dis
            coeff_matrix = 1 / (1 + coeff_matrix)
            ### add filter
            coeff_matrix[coeff_matrix < 0.005] = 0
            topology_feats[rois[:, 0] == i] = torch.mm(coeff_matrix, gt_bbox_feats[gt_rois[:, 0] == i])

        return topology_feats

    def forward_train(self,
                      img,
                      img_meta,
                      ref_img,
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

            img_meta (list[dict]): list of image info dict where each dict has:
                'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(img)
        #print(img.shape)
        ref_x = self.extract_feat(ref_img)

        # conv LSTM

        #if self.train_cfg.ConvLSTM:
        losses = {}

        # RPN forward and loss
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta,
                                          self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            proposal_inputs = rpn_outs + (img_meta, proposal_cfg)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals

        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
            bbox_sampler = build_sampler(
                self.train_cfg.rcnn.sampler, context=self)
            num_imgs = img.size(0)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = bbox_assigner.assign(proposal_list[i],
                                                     gt_bboxes[i],
                                                     gt_bboxes_ignore[i],
                                                     gt_labels[i],
                                                     gt_pids[i])
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    gt_pids[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        # bbox head forward and loss
        if self.with_bbox:
            #import pdb;pdb.set_trace()
            rois = bbox2roi([res.bboxes for res in sampling_results])
            # TODO: a more flexible way to decide which feature maps to use
            bbox_img_n = [res.bboxes.size(0) for res in sampling_results]
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            cls_score, bbox_pred = self.bbox_head(bbox_feats)

            bbox_targets, (ids, id_weights) = self.bbox_head.get_target(sampling_results,
                                                     gt_bboxes, gt_labels,
                                                     self.train_cfg.rcnn)
            loss_bbox = self.bbox_head.loss(cls_score, bbox_pred,
                                            *bbox_targets)
            losses.update(loss_bbox)
            # tracking head
            #ref_roi = bbox2roi(ref_bboxes)
            #ref_bbox_feats = self.bbox_roi_extractor(
            #    ref_x[:self.bbox_roi_extractor.num_inputs], ref_roi)
            #ref_bbox_img_n = [x.size(0) for x in ref_bboxes]
            #match_score = self.track_head(bbox_feats, ref_bbox_feats,
            #                              bbox_img_n, ref_bbox_img_n)

            # tracking head with appearance-semantic-geometry feature
            # appearance
            ref_roi = bbox2roi(ref_bboxes)
            ref_bbox_feats = self.track_roi_extractor(
                ref_x[:self.track_roi_extractor.num_inputs], ref_roi)
            bbox_track_feats = self.track_roi_extractor(
                x[:self.track_roi_extractor.num_inputs], rois)
            ref_bbox_img_n = [x.size(0) for x in ref_bboxes]
            # geometry
            rois_input = rois.clone()[:, 1 : 5]
            ref_roi_input = ref_roi.clone()[:, 1 : 5]         
            # semantic
            ref_sem_feats_input = self.mask_head(ref_bbox_feats, True)
            bbox_sem_feats_input = self.mask_head(bbox_track_feats, True)          
            # topology
            gt_roi = bbox2roi(gt_bboxes)
            gt_bbox_feats = self.track_roi_extractor(
                x[:self.track_roi_extractor.num_inputs], gt_roi)
            batch_num = len(ref_bbox_img_n)
            bbox_top_feats_input = self.get_topology_feats(rois, gt_roi, gt_bbox_feats, batch_num) 
            ref_top_feats_input = self.get_topology_feats(ref_roi, ref_roi, ref_bbox_feats, batch_num)
            
            match_score = self.track_head(bbox_track_feats, ref_bbox_feats,
                                          bbox_img_n, ref_bbox_img_n, rois_input, ref_roi_input,
                                          bbox_sem_feats_input, ref_sem_feats_input, bbox_top_feats_input, ref_top_feats_input)
            loss_match = self.track_head.loss(match_score, ids, id_weights)
            losses.update(loss_match)

        # mask head forward and loss
        if self.with_mask:
            if not self.share_roi_extractor:
                pos_rois = bbox2roi(
                    [res.pos_bboxes for res in sampling_results])
                mask_feats = self.mask_roi_extractor(
                    x[:self.mask_roi_extractor.num_inputs], pos_rois)
                if self.with_shared_head:
                    mask_feats = self.shared_head(mask_feats)
            else:
                pos_inds = []
                device = bbox_feats.device
                for res in sampling_results:
                    pos_inds.append(
                        torch.ones(
                            res.pos_bboxes.shape[0],
                            device=device,
                            dtype=torch.uint8))
                    pos_inds.append(
                        torch.zeros(
                            res.neg_bboxes.shape[0],
                            device=device,
                            dtype=torch.uint8))
                pos_inds = torch.cat(pos_inds)
                mask_feats = bbox_feats[pos_inds]
            if mask_feats.shape[0] > 0:
                mask_pred = self.mask_head(mask_feats)
                mask_targets = self.mask_head.get_target(
                    sampling_results, gt_masks, self.train_cfg.rcnn)
                pos_labels = torch.cat(
                    [res.pos_gt_labels for res in sampling_results])
                loss_mask = self.mask_head.loss(mask_pred, mask_targets,
                                                pos_labels)
                losses.update(loss_mask)

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

    def fuse_motion(self, kf, cost_matrix, means, covariances, detections, only_position=False, lambda_=0.98):
        chi2inv95 = {
            1: 3.8415,
            2: 5.9915,
            3: 7.8147,
            4: 9.4877,
            5: 11.070,
            6: 12.592,
            7: 14.067,
            8: 15.507,
            9: 16.919}
        if cost_matrix.size == 0:
            return cost_matrix
        gating_dim = 8
        gating_threshold = chi2inv95[gating_dim] * 1.5
        det = detections.cpu().numpy().copy()
        ### tlbr_to_tlwh
        det[:, 2:] -= det[:, :2]
        ### tlwh_to_xyah
        det[:, :2] += det[:, 2:] / 2
        det[:, 2] /= det[:, 3]
        for i in range(len(means)):
            gating_distance = kf.gating_distance(
                means[i], covariances[i], det, only_position, metric='maha')
            cost_matrix[gating_distance > gating_threshold, i] = np.inf
            #cost_matrix[:, i] = lambda_ * cost_matrix[:, i] + (1 - lambda_) * gating_distance
        return cost_matrix

    def multi_predict(self, means, covariances):
        if len(means) > 0:
            multi_mean = np.asarray([mean for mean in means])
            multi_covariance = np.asarray([covariance for covariance in covariances])
            multi_mean, multi_covariance = self.kalman_filter.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                means[i] = mean
                covariances[i] = cov
        return means, covariances

    def simple_test_bboxes(self,
                           x,
                           img_meta,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        """ Test only det bboxes without augmentation"""
        rois = bbox2roi(proposals)
        roi_feats = self.bbox_roi_extractor(
            x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
        cls_score, bbox_pred = self.bbox_head(roi_feats)
        img_shape = img_meta[0]['img_shape']
        scale_factor = img_meta[0]['scale_factor']
        is_first = img_meta[0]['is_first']
        if is_first:
           self.frame_id = 0
        else:
           self.frame_id += 1
        det_bboxes, det_labels = self.bbox_head.get_det_bboxes(
            rois,
            cls_score,
            bbox_pred,
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg)
        ### add use detection results
        use_det_res = False
        if (use_det_res):
            img_path = img_meta[0]['file_name']
            det_res = self.res_file[img_path]['content_ann']['bboxes'] 
            det_res = np.array(det_res)
            det_labels = torch.zeros(det_res.shape[0], dtype=int).to(det_labels.device) 
            res_bboxes = np.ones((det_res.shape[0], 5))
            res_bboxes[:, 0] = np.min(det_res[:, ::2], axis=1)
            res_bboxes[:, 1] = np.min(det_res[:, 1::2], axis=1)
            res_bboxes[:, 2] = np.max(det_res[:, ::2], axis=1)
            res_bboxes[:, 3] = np.max(det_res[:, 1::2], axis=1)
            det_bboxes = torch.Tensor(res_bboxes).to(det_bboxes.device)
            
        if det_bboxes.nelement()==0:
            det_obj_ids=np.array([], dtype=np.int64)
            if is_first:
                self.prev_bboxes = None
                self.prev_sem = None
                self.prev_top = None
                self.prev_roi_feats = None
                self.prev_mean = []
                self.prev_covariance = []
                self.frame_record = []                              
            return det_bboxes, det_labels, det_obj_ids

        res_det_bboxes = det_bboxes.clone()
        if rescale:
            res_det_bboxes[:, :4] *= scale_factor
        det_rois = bbox2roi([res_det_bboxes])
        track_roi_feats = self.track_roi_extractor(
            x[:self.track_roi_extractor.num_inputs], det_rois)
        sem_feats = self.mask_head(track_roi_feats, True)
        top_feats = self.get_topology_feats(det_rois, det_rois, track_roi_feats, 1) 
        # add bbox feature
        if not is_first and self.prev_bboxes is not None:
            det_rois_input = det_rois.clone()[:, 1 : 5]         
            res_prev_bboxes = self.prev_bboxes.clone()
            if rescale:
                res_prev_bboxes[:, :4] *= scale_factor
            prev_rois_input = res_prev_bboxes[:, 0 : 4]

        # recompute bbox match feature

        if is_first or (not is_first and self.prev_bboxes is None):
            det_obj_ids = np.arange(det_bboxes.size(0))
            # save bbox and features for later matching
            self.prev_bboxes = det_bboxes
            self.prev_roi_feats = track_roi_feats
            self.prev_sem = sem_feats
            self.prev_top = top_feats
            # save motion 
            for i in range(res_det_bboxes.shape[0]):
               mean, covariance = self.kalman_filter.initiate(res_det_bboxes[i, :])
               self.prev_mean.append(mean)
               self.prev_covariance.append(covariance)
            # record frame id
            self.frame_record = [self.frame_id] * det_bboxes.size(0)
        else:
            assert self.prev_roi_feats is not None
            # only support one image at a time
            bbox_img_n = [det_bboxes.size(0)]
            prev_bbox_img_n = [self.prev_roi_feats.size(0)]
            match_score = self.track_head(track_roi_feats, self.prev_roi_feats,
                                      bbox_img_n, prev_bbox_img_n, det_rois_input, prev_rois_input, sem_feats, self.prev_sem, top_feats, self.prev_top)[0]
            ### add kalman filter
            cost_matrix = match_score.cpu().numpy()
            self.prev_mean, self.prev_covariance = self.multi_predict(self.prev_mean, self.prev_covariance)
            #cost_matrix = self.fuse_motion(self.kalman_filter, cost_matrix, self.prev_mean, self.prev_covariance, det_rois_input)            
            ### remove miss object
            miss_id = np.where(self.frame_id - np.array(self.frame_record) > 15)
            cost_matrix[:, miss_id] = np.inf
            #import pdb;pdb.set_trace()
            ### match with linear assignment
            cost, x, y = lap.lapjv(cost_matrix, True, 0.8)
            det_obj_ids = np.ones((x.shape[0]), dtype=np.int32) * (-1)
            ### first match
            for idx, match_id in enumerate(x):
               if match_id >= 0:
                  det_obj_ids[idx] = match_id
                  # udpate feature
                  #self.prev_roi_feats[match_id] = track_roi_feats[idx]
                  # update feature smooth
                  alpha = 0.5
                  self.prev_roi_feats[match_id] = track_roi_feats[idx] * (1 - alpha) + self.prev_roi_feats[match_id] * alpha
                  self.prev_sem[match_id] = sem_feats[idx] * (1 - alpha) + self.prev_sem[match_id] * alpha
                  self.prev_top[match_id] = top_feats[idx] * (1 - alpha) + self.prev_top[match_id] * alpha
                  self.prev_bboxes[match_id] = det_bboxes[idx]
                  # update motion
                  self.prev_mean[match_id], self.prev_covariance[match_id] = self.kalman_filter.update(self.prev_mean[match_id], self.prev_covariance[match_id], det_rois_input[idx])
                  # update frame id
                  self.frame_record[match_id] = self.frame_id
            ### second match with iou
            unmatched_det = np.where(x < 0)[0]
            unmatched_prev = np.where(y < 0)[0]
            bbox_ious = bbox_overlaps(det_bboxes[unmatched_det, :4], self.prev_bboxes[unmatched_prev, :4])
            cost_matrix = 1 - bbox_ious.cpu().numpy()
            #import pdb;pdb.set_trace()
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
                  self.prev_top = torch.cat((self.prev_top, top_feats[det_id][None]), dim=0)
                  self.prev_bboxes = torch.cat((self.prev_bboxes, det_bboxes[det_id][None]), dim=0)
                  # add motion
                  mean, covariance = self.kalman_filter.initiate(det_rois_input[det_id])
                  self.prev_mean.append(mean)
                  self.prev_covariance.append(covariance)
                  # add new frame id
                  self.frame_record.append(self.frame_id)
               else:
                  r_match_id = unmatched_prev[match_id]
                  det_obj_ids[unmatched_det[idx]] = r_match_id
                  # udpate feature
                  #self.prev_roi_feats[match_id] = track_roi_feats[idx]
                  # update feature smooth
                  alpha = 0.5
                  self.prev_roi_feats[r_match_id] = track_roi_feats[det_id] * (1 - alpha) + self.prev_roi_feats[r_match_id] * alpha
                  self.prev_sem[r_match_id] = sem_feats[det_id] * (1 - alpha) + self.prev_sem[r_match_id] * alpha
                  self.prev_top[r_match_id] = top_feats[det_id] * (1 - alpha) + self.prev_top[r_match_id] * alpha     
                  self.prev_bboxes[r_match_id] = det_bboxes[det_id]
                  # update motion
                  self.prev_mean[r_match_id], self.prev_covariance[r_match_id] = self.kalman_filter.update(self.prev_mean[r_match_id], self.prev_covariance[r_match_id], det_rois_input[det_id]) 
                  # update frame id
                  self.frame_record[match_id] = self.frame_id
        return det_bboxes, det_labels, det_obj_ids

    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."
        assert self.with_track, "Track head must be implemented"

        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = self.simple_test_rpn(x, img_meta,
                                                 self.test_cfg.rpn)
        else:
            proposal_list = proposals

        det_bboxes, det_labels, det_obj_ids = self.simple_test_bboxes(
            x, img_meta, proposal_list, self.test_cfg.rcnn, rescale=rescale)
        print(det_bboxes.shape)
        print(det_obj_ids)
        bbox_results = bbox2result_with_id(det_bboxes, det_labels, det_obj_ids,
                                   self.bbox_head.num_classes)

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_meta, det_bboxes, det_labels,
                rescale=rescale, det_obj_ids=det_obj_ids)
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
