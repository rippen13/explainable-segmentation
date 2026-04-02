import numpy as np

def compute_iou(exp_map, gt_mask, threshold=0.5):
    exp_binary = exp_map >= threshold
    gt_binary = gt_mask.astype(bool)

    intersection = np.logical_and(exp_binary, gt_binary).sum()
    union = np.logical_or(exp_binary, gt_binary).sum()

    if union == 0:
        return 0.0

    return intersection / union