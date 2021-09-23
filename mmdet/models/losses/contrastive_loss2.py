import torch
import torch.nn as nn

from ..builder import LOSSES

@LOSSES.register_module()
class ContrastiveLoss2(nn.Module):
    """
    加了另一个margin
    """

    def __init__(self, margin1=0.2, margin2=1):
        super().__init__()
        self.margin1 = margin1
        self.margin2 = margin2

    def forward(self, out_put, label):
        out_put = torch.squeeze(out_put)
        label = torch.squeeze(label)
        sigloss2 = torch.mean((1 - label) * torch.pow(torch.clamp(self.margin2 - out_put, min=0.0), 2) +
                              (label) * torch.pow(torch.clamp(out_put - self.margin1, min=0.0), 2))

        return sigloss2
