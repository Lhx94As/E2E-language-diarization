import torch
import torch.nn as nn
from itertools import permutations


# predict & label size: (B, T, C)
# B: Batch size
# T: Number of frames
# C: number of classes, in our case: silence, language 1 and language 2.

class DeepClusteringLoss(nn.Module):
    def __init__(self):
        super(DeepClusteringLoss, self).__init__()

    def forward(self, output, target):
        target = nn.functional.one_hot(target).float()
        num_frames = output.size()[0]
        vt_v = torch.norm(torch.matmul(torch.transpose(output, 0, 1), output), p=2) ** 2
        vt_y = torch.norm(torch.matmul(torch.transpose(output, 0, 1), target), p=2) ** 2
        yt_y = torch.norm(torch.matmul(torch.transpose(target, 0, 1), target), p=2) ** 2
        DC_loss = vt_v - 2 * vt_y + yt_y
        return DC_loss/(num_frames**2)

