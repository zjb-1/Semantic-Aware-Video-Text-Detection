from ..builder import DETECTORS
from .two_stage_aggregation_track import TwoStageDetectorAggregationTrack


@DETECTORS.register_module()
class MaskTrackAggregationRCNN(TwoStageDetectorAggregationTrack):

    def __init__(self,
                 backbone,
                 rpn_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 roi_head=None,
                 pretrained=None):
        super(MaskTrackAggregationRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
