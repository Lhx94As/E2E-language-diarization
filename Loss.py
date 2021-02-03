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
        #
        # num_frames = output.size()[1]*output.size()[0]
        # vt_v = torch.norm(torch.matmul(torch.transpose(output, 1, 2), output), p=2) ** 2
        # vt_y = torch.norm(torch.matmul(torch.transpose(output, 1, 2), target), p=2) ** 2
        # yt_y = torch.norm(torch.matmul(torch.transpose(target, 1, 2), target), p=2) ** 2
        DC_loss = vt_v - 2 * vt_y + yt_y
        return DC_loss/(num_frames**2)





class PermutationInvariantLoss(nn.Module):
    def __init__(self,device):
        super(PermutationInvariantLoss, self).__init__()
        self.device = device

    def forward(self, predicts, targets, seq_len):
        min_loss = 0
        loss_func = nn.CrossEntropyLoss().to(self.device)
        predicts = torch.split(predicts, seq_len)
        targets = torch.split(targets, seq_len)
        for i in range(len(seq_len)):
            predicts_ = predicts[i]
            targets_ = targets[i]
            n_frames = seq_len[i]
            label_perms = [targets_[..., list(p)]
                           for p in permutations(range(targets_.size()[-1]))]
            min_utt_loss = 0
            for ii in range(len(label_perms)):
                label = label_perms[ii]
                label = torch.argmax(label, dim=-1)
                loss_ = loss_func(predicts_, label)/n_frames
                print(loss_)
                if min_utt_loss > loss_:
                    min_utt_loss = loss_
            min_loss += min_utt_loss
        return min_loss/self.batch_size
