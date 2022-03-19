import torch
import torch.nn as nn

from ..builder import LOSSES
from .utils import weighted_loss

def multilabel_categorical_crossentropy_loss(y_pred, y_true):
    y_pred = (1 - 2*y_true)*y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[...,:1])
    y_pred_neg = torch.cat((y_pred_neg,zeros),dim=-1)
    y_pred_pos = torch.cat((y_pred_pos,zeros),dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg,dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos,dim=-1)
    return torch.mean(neg_loss + pos_loss)

@LOSSES.register_module()
class MultilabelCatCrossLoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(MultilabelCatCrossLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * multilabel_categorical_crossentropy_loss(pred, target)
        return loss