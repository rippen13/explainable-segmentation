import numpy as np
from src.evaluation.iou import compute_iou

# Fake data
exp_map = np.random.rand(256, 256)
gt_mask = np.random.randint(0, 2, (256, 256))

iou = compute_iou(exp_map, gt_mask)

print("IoU Score:", iou)