import os
import numpy as np
import torch
from src.evaluation.runner import evaluate_sample

def run_evaluation(model, dataset):
    results = []

    for i, (image, mask, exp_map) in enumerate(dataset):
        metrics = evaluate_sample(model, image, mask, exp_map)
        results.append(metrics)

        print(f"Processed {i+1}")

    return results


def summarize(results):
    iou = np.mean([r["iou"] for r in results])
    ins = np.mean([r["insertion_auc"] for r in results])
    dele = np.mean([r["deletion_auc"] for r in results])

    print("\n=== FINAL RESULTS ===")
    print("IoU:", iou)
    print("Insertion AUC:", ins)
    print("Deletion AUC:", dele)


if __name__ == "__main__":
    model = torch.load("model.pth")
    dataset = []  # Replace with your loader

    results = run_evaluation(model, dataset)
    summarize(results)