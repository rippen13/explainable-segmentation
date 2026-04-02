import numpy as np
import torch

def insertion_auc(model, image, exp_map, steps=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    image = image.to(device)

    flat = exp_map.flatten()
    indices = np.argsort(-flat)

    baseline = torch.zeros_like(image).to(device)
    scores = []

    total_pixels = flat.shape[0]
    step_size = total_pixels // steps

    for i in range(steps):
        idx = indices[: (i+1)*step_size]

        mask = np.zeros_like(flat)
        mask[idx] = 1
        mask = mask.reshape(exp_map.shape)

        mask = torch.tensor(mask).float().to(device)
        mask = mask.unsqueeze(0).repeat(image.shape[0], 1, 1)

        modified = baseline + image * mask

        with torch.no_grad():
            output = model(modified.unsqueeze(0))
            score = output.mean().item()

        scores.append(score)

    auc = np.trapezoid(scores) / steps
    return auc, scores