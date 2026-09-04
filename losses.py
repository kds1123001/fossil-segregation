import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceCELoss(nn.Module):
    def __init__(self, ce_weight=None, dice_weight=1.0, ce_scale=1.0, smooth=1e-5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=ce_weight)
        self.dice_weight = dice_weight
        self.ce_scale = ce_scale
        self.smooth = smooth

    def forward(self, logits, target):
        ce_loss = self.ce(logits, target)

        probs = F.softmax(logits, dim=1)
        n_classes = probs.shape[1]
        target_oh = F.one_hot(target, n_classes).permute(0, 4, 1, 2, 3).float()

        dims = (0, 2, 3, 4)
        intersection = torch.sum(probs * target_oh, dims)
        cardinality = torch.sum(probs + target_oh, dims)
        dice_per_class = (2 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1 - dice_per_class[1:].mean()

        return self.ce_scale * ce_loss + self.dice_weight * dice_loss


def dice_score(pred_mask, target_mask, smooth=1e-5):
    inter = (pred_mask & target_mask).sum().item()
    denom = pred_mask.sum().item() + target_mask.sum().item()
    return (2 * inter + smooth) / (denom + smooth)
