import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dice_score import dice_loss


class MultiTaskLoss(nn.Module):
    def __init__(self, n_classes, task='SEG'):
        super(MultiTaskLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        if task == 'DET':
            self.loss = lambda pred, mask: self.ce_loss(pred, mask)
        elif task == 'SEG':
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
