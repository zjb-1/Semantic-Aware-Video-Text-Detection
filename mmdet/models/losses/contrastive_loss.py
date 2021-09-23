import torch
import torch.nn as nn

from ..builder import LOSSES

@LOSSES.register_module()
class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, out_put, label):
        # euclidean_distance = F.pairwise_distance(output1, output2)
        out_put = torch.squeeze(out_put)
        label = torch.squeeze(label)
        loss_contrastive = torch.mean((label) * torch.pow(out_put, 2) + (1 - label) * torch.pow(torch.clamp(self.margin - out_put, min=0.0), 2))

        return loss_contrastive
