import numpy as np
import torch
import torch.nn as nn

from src.evaluation.iou import compute_iou
from src.evaluation.insertion import insertion_auc
from src.evaluation.deletion import deletion_auc


# Dummy model (VERY IMPORTANT)
class DummyModel(nn.Module):
    def forward(self, x):
        return x.mean(dim=(1, 2, 3), keepdim=True)


model = DummyModel()


# Fake data
image = torch.rand(3, 128, 128)
exp_map = np.random.rand(128, 128)
gt_mask = np.random.randint(0, 2, (128, 128))


# Run metrics
iou = compute_iou(exp_map, gt_mask)
ins_auc, _ = insertion_auc(model, image, exp_map)
del_auc, _ = deletion_auc(model, image, exp_map)


print("IoU:", iou)
print("Insertion AUC:", ins_auc)
print("Deletion AUC:", del_auc)