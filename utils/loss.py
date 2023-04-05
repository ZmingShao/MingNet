import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dice_score import dice_loss


def _neg_loss(pred, gt):
    """ Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
      Arguments:
        pred (batch x c x h x w)
        gt (batch x c x h x w)
    """
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target):
        out = torch.clamp(out.sigmoid_(), min=1e-4, max=1 - 1e-4)
        if target.ndim == 3:
            target = target.unsqueeze(1)

        return self.neg_loss(out[:, 1:, ...], target)


class MultiTaskLoss(nn.Module):
    def __init__(self, n_classes, task='SEG'):
        super(MultiTaskLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.focal_loss = FocalLoss()
        if task == 'DET':
            # self.loss = lambda pred, mask: self.ce_loss(pred, mask)
            self.loss = lambda pred, mask: self.focal_loss(pred, mask)
        elif task == 'SEG':
            # self.loss = lambda pred, mask: self.focal_loss(pred, mask)
            self.loss = lambda pred, mask: 0.5 * self.ce_loss(pred, mask) + 0.5 * dice_loss(
                F.softmax(pred, dim=1).float(),
                F.one_hot(mask, n_classes).permute(0, 3, 1, 2).float(),
                multiclass=True
            )
        else:
            self.loss = lambda pred, mask: self.ce_loss(pred, mask)
            print(f'Unrecognized task name: {task}, expected `DET` or `SEG`')

    def forward(self, pred, mask):
        assert isinstance(mask, torch.Tensor), 'The GT mask must be `Tensor`'
        if isinstance(pred, torch.Tensor):
            loss = self.loss(pred, mask)
        elif isinstance(pred, (tuple, list)):
            loss = sum(self.loss(p, mask) for p in pred) / len(pred)
        else:
            loss = 0
            print('The output of model should be either `Tensor` or `Tuple/List`')

        return loss
