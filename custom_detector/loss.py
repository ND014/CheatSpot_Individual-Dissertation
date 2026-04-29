import torch
import torch.nn as nn
from utils import iou


class DetectionLoss(nn.Module):
    def __init__(self, S=13, lambda_coord=5.0, lambda_noobj=0.5):
        super().__init__()
        self.S           = S
        self.lambda_coord = lambda_coord   # weight for box loss
        self.lambda_noobj = lambda_noobj   # weight for no-object loss
        self.mse = nn.MSELoss(reduction='sum')
        self.bce = nn.BCEWithLogitsLoss(reduction='sum')

    def forward(self, predictions, targets):
        """
        predictions: (B, 6, S, S)
        targets:     (B, 6, S, S)
        """
        # obj_mask  = cells that HAVE an object
        # noobj_mask = cells that DON'T have an object
        obj_mask   = targets[:, 4, :, :] == 1   # (B, S, S)
        noobj_mask = targets[:, 4, :, :] == 0   # (B, S, S)

        # ── Box Loss (only where objects exist) ───────
        pred_xy = predictions[:, 0:2, :, :]   # tx, ty
        pred_wh = predictions[:, 2:4, :, :]   # tw, th
        true_xy = targets[:, 0:2, :, :]
        true_wh = targets[:, 2:4, :, :]

        mask_4d = obj_mask.unsqueeze(1).expand_as(pred_xy)

        box_loss = self.lambda_coord * (
            self.mse(
                pred_xy[mask_4d],
                true_xy[mask_4d]
            ) +
            self.mse(
                pred_wh[mask_4d],
                true_wh[mask_4d]
            )
        )

        # ── Objectness Loss ───────────────────────────
        pred_obj = predictions[:, 4, :, :]
        true_obj = targets[:, 4, :, :]

        obj_loss   = self.bce(pred_obj[obj_mask],   true_obj[obj_mask])
        noobj_loss = self.lambda_noobj * self.bce(
                         pred_obj[noobj_mask],
                         true_obj[noobj_mask]
                     )

        # ── Class Loss (only where objects exist) ─────
        pred_cls = predictions[:, 5, :, :]
        true_cls = targets[:, 5, :, :]

        cls_loss = self.bce(pred_cls[obj_mask], true_cls[obj_mask])

        total_loss = box_loss + obj_loss + noobj_loss + cls_loss
        return total_loss, box_loss, obj_loss, noobj_loss, cls_loss
