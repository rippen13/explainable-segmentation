import numpy as np

def normalize_map(exp_map):
    exp_map = exp_map - exp_map.min()
    exp_map = exp_map / (exp_map.max() + 1e-8)
    return exp_map


def resize_check(image, mask, exp_map):
    assert image.shape[1:] == mask.shape == exp_map.shape, \
        "Shapes mismatch!"