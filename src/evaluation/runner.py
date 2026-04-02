from .iou import compute_iou
from .insertion import insertion_auc
from .deletion import deletion_auc
from .utils import normalize_map

def evaluate_sample(model, image, mask, exp_map):
    exp_map = normalize_map(exp_map)

    iou = compute_iou(exp_map, mask)
    ins_auc, ins_curve = insertion_auc(model, image, exp_map)
    del_auc, del_curve = deletion_auc(model, image, exp_map)

    return {
        "iou": iou,
        "insertion_auc": ins_auc,
        "deletion_auc": del_auc,
        "ins_curve": ins_curve,
        "del_curve": del_curve
    }