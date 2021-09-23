import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmdet.models.losses import  cross_entropy, smooth_l1_loss, accuracy, ContrastiveLoss, ContrastiveLoss2
from mmdet.models.builder import HEADS


@HEADS.register_module()
class TrackHead(nn.Module):
    """Tracking head, predict tracking features and match with reference objects
       Use dynamic option to deal with different number of objects in different
       images. A non-match entry is added to the reference objects with all-zero
       features. Object matched with the non-match entry is considered as a new
       object.
    """

    def __init__(self,
                 with_avg_pool=False,
                 num_fcs = 2,
                 in_channels=256,
                 roi_feat_size=7,
                 fc_out_channels=1024,
                 match_coeff=None,
                 bbox_dummy_iou=0,
                 loss_weight=None,
                 neg_ratio=5
                 ):
        super(TrackHead, self).__init__()
        self.in_channels = in_channels
        self.with_avg_pool = with_avg_pool
        self.roi_feat_size = roi_feat_size
        self.match_coeff = match_coeff
        self.bbox_dummy_iou = bbox_dummy_iou
        self.num_fcs = num_fcs
        self.loss_weight = loss_weight
        self.neg_ratio = neg_ratio
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(roi_feat_size)
        else:
            in_channels *= (self.roi_feat_size * self.roi_feat_size)
        self.fcs = nn.ModuleList()
        for i in range(num_fcs):
            in_channels = (in_channels
                          if i == 0 else fc_out_channels)
            fc = nn.Linear(in_channels, fc_out_channels)
            self.fcs.append(fc)
        # init geometry fc
        self.fcs_b = nn.ModuleList()
        in_channels_b = 4
        fc_out_channels_b = 64
        for i in range(num_fcs):
            in_channels_b = (in_channels_b
                          if i == 0 else fc_out_channels_b)
            fc = nn.Linear(in_channels_b, fc_out_channels_b)
            self.fcs_b.append(fc)
        # init semantic fc
        self.fcs_s = nn.ModuleList()
        in_channels_s = 256 * 7 * 7
        fc_out_channels_s = 64
        for i in range(num_fcs):
            in_channels_s = (in_channels_s
                          if i == 0 else fc_out_channels_s)
            fc = nn.Linear(in_channels_s, fc_out_channels_s)
            self.fcs_s.append(fc)
        # init done
        self.relu = nn.ReLU(inplace=True)
        self.debug_imgs = None

    def init_weights(self):
        for fc in self.fcs:
            nn.init.normal_(fc.weight, 0, 0.01)
            nn.init.constant_(fc.bias, 0)
        # init geometry fc weight
        for fc in self.fcs_b:
            nn.init.normal_(fc.weight, 0, 0.01)
            nn.init.constant_(fc.bias, 0)
        # init semantic fc weight
        for fc in self.fcs_s:
            nn.init.normal_(fc.weight, 0, 0.01)
            nn.init.constant_(fc.bias, 0)
        ### init done

    def compute_comp_scores(self, match_ll, bbox_scores, bbox_ious, label_delta, add_bbox_dummy=False):
        # compute comprehensive matching score based on matchig likelihood,
        # bbox confidence, and ious
        if add_bbox_dummy:
            bbox_iou_dummy =  torch.ones(bbox_ious.size(0), 1,
                device=torch.cuda.current_device()) * self.bbox_dummy_iou
            bbox_ious = torch.cat((bbox_iou_dummy, bbox_ious), dim=1)
            label_dummy = torch.ones(bbox_ious.size(0), 1,
                device=torch.cuda.current_device())
            label_delta = torch.cat((label_dummy, label_delta),dim=1)
        if self.match_coeff is None:
            return match_ll
        else:
            # match coeff needs to be length of 3
            assert(len(self.match_coeff) == 3)
            return match_ll + self.match_coeff[0] * \
                torch.log(bbox_scores) + self.match_coeff[1] * bbox_ious \
                + self.match_coeff[2] * label_delta

    def forward(self, x, ref_x, x_n, ref_x_n, x_b, ref_b, x_s, ref_s):
        # x and ref_x are the grouped bbox features of current and reference frame
        # x_n are the numbers of proposals in the current images in the mini-batch,
        # ref_x_n are the numbers of ground truth bboxes in the reference images.
        # here we compute a correlation matrix of x and ref_x
        # we also add a all 0 column denote no matching
        # x_b and ref_b are the bounding boxes of rois of current and reference frame
        # x_s and ref_s are the semantic features of current and reference frame
        assert len(x_n) == len(ref_x_n)
        if self.with_avg_pool:
            x = self.avg_pool(x)
            ref_x = self.avg_pool(ref_x)
        x = x.view(x.size(0), -1)
        ref_x = ref_x.view(ref_x.size(0), -1)
        for idx, fc in enumerate(self.fcs):
            x = fc(x)
            ref_x = fc(ref_x)
            if idx < len(self.fcs) - 1:
                x = self.relu(x)
                ref_x = self.relu(ref_x)
        # concat geometry feature
        x_b = x_b.view(x_b.size(0), -1)
        ref_b = ref_b.view(ref_b.size(0), -1)
        for idx, fc in enumerate(self.fcs_b):
            x_b = fc(x_b)
            ref_b = fc(ref_b)
            if idx < len(self.fcs_b) - 1:
                x_b = self.relu(x_b)
                ref_b = self.relu(ref_b)
        x = torch.cat((x, x_b), 1)
        ref_x = torch.cat((ref_x, ref_b), 1)
        # concat semantic feature
        x_s = x_s.view(x_s.size(0), -1)
        ref_s = ref_s.view(ref_s.size(0), -1)
        for idx, fc in enumerate(self.fcs_s):
            x_s = fc(x_s)
            ref_s = fc(ref_s)
            if idx < len(self.fcs_s) - 1:
                x_s = self.relu(x_s)
                ref_s = self.relu(ref_s)
        x = torch.cat((x, x_s), 1)
        ref_x = torch.cat((ref_x, ref_s), 1)
        # concat done
        n = len(x_n)
        x_split = torch.split(x, x_n, dim=0)
        ref_x_split = torch.split(ref_x, ref_x_n, dim=0)
        prods = []  # match score
        for i in range(n):
            # change prod to euc distance
            prod = torch.zeros((x_split[i].shape[0], ref_x_split[i].shape[0])).to(x_split[i].device)
            for ref_id in range(ref_x_split[i].shape[0]):
               dis = F.pairwise_distance(x_split[i], ref_x_split[i][ref_id, :], p=2)
               prod[:, ref_id] = dis
            prods.append(prod)

        return prods

    def loss(self,
             match_score,
             ids,
             id_weights):
        losses = dict()
        losses['loss_match'] = torch.tensor(0.).to(ids.device)
        losses['match_acc'] = torch.tensor(0.).to(ids.device)
        track_loss = ContrastiveLoss2(0.3, 1.0)

        n = len(match_score)
        x_n = [s.size(0) for s in match_score]
        ids = torch.split(ids, x_n, dim=0)
        loss_match = 0.
        match_acc = 0.
        n_total = 0
        for score, cur_ids, cur_weights in zip(match_score, ids, id_weights):
            valid_idx = torch.nonzero(cur_weights).squeeze()
            if len(valid_idx.size()) == 0: continue

            n_valid = valid_idx.size(0)
            n_total += n_valid
            score_valid = torch.index_select(score, 0, valid_idx)
            ids_valid = torch.index_select(cur_ids, 0, valid_idx)
            # convert to one-hot label
            label = torch.zeros(score_valid.shape).to(score_valid.device)
            for i in range(len(ids_valid)):
                if ids_valid[i] > 0:
                    label[i, ids_valid[i] - 1] = 1

            label = label.view(-1)
            score_valid = score_valid.view(-1)

            mining = True
            mining_ratio = 1
            if mining:
               pos_n = int(label.sum())
               neg_n = min(int(label.shape[0]) - pos_n, pos_n * self.neg_ratio)
               if neg_n == pos_n * self.neg_ratio and (pos_n + neg_n) > 0:
                  # begin mining
                  score_neg = score_valid[label == 0]
                  score_sort, sort_id = score_neg.sort()
                  score_mask = torch.ones(score_valid.shape).to(score_valid.device)
                  neg_mask = torch.ones(score_neg.shape).to(score_valid.device)
                  neg_mask[sort_id[neg_n: sort_id.shape[0]]] = 1000000
                  score_mask[label == 0] = neg_mask
                  score_valid = score_mask * score_valid
                  mining_ratio = label.shape[0] / (pos_n + neg_n)
            loss_match += self.loss_weight * mining_ratio * track_loss(score_valid, label)
            match_acc += accuracy(torch.index_select(score, 0, valid_idx),
                                  torch.index_select(cur_ids, 0, valid_idx)) * n_valid
        if loss_match > 0:
            losses['loss_match'] = loss_match / n
        if n_total > 0:
            losses['match_acc'] = match_acc / n_total

        return losses


