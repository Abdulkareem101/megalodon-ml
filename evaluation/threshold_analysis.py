"""
Sweep the detector's decision threshold and report accuracy / precision /
recall / F1 / missed-malware count at each one.

Raising the threshold trades recall for precision: fewer false alarms, but
more missed malware. For a security tool, missed malware (false negatives)
is usually the more expensive mistake, so this sweep is what we used to
pick an operating point instead of defaulting to 0.5.

Usage:
    python threshold_analysis.py --data-dir /path/to/dataset --model malware_detector_refined_best.pt
"""

import argparse

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b1

DEFAULT_THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]


def load_model(model_path, device):
    model = efficientnet_b1(weights=None)
    original_conv = model.features[0][0]
    model.features[0][0] = nn.Conv2d(
        1, original_conv.out_channels, kernel_size=original_conv.kernel_size,
        stride=original_conv.stride, padding=original_conv.padding, bias=False,
    )
    model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.classifier[1].in_features, 2))
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()


def collect_probabilities(model, loader, device):
    """Run inference once and reuse the probabilities across every threshold."""
    all_labels, all_probs = [], []
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))
            probs = torch.softmax(outputs, dim=1)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    return np.array(all_labels), np.array(all_probs)


def sweep(labels, probs, thresholds):
    malware_probs = probs[:, 1]
    rows = []
    for t in thresholds:
        preds = (malware_probs >= t).astype(int)
        cm = confusion_matrix(labels, preds)
        tn, fp, fn, tp = cm.ravel()
        rows.append({
            "threshold": t,
            "accuracy": accuracy_score(labels, preds),
            "precision": precision_score(labels, preds, zero_division=0),
            "recall": recall_score(labels, preds, zero_division=0),
            "f1": f1_score(labels, preds, zero_division=0),
            "missed_malware": int(fn),
        })
    return rows


def print_table(rows):
    header = f"{'Threshold':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Missed (FN)'}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['threshold']:<10.2f} {r['accuracy']:<10.4f} {r['precision']:<10.4f} "
              f"{r['recall']:<10.4f} {r['f1']:<10.4f} {r['missed_malware']:,}")


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    dataset = datasets.ImageFolder(f"{args.data_dir}/{args.split}", transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    model = load_model(args.model, device)
    labels, probs = collect_probabilities(model, loader, device)

    thresholds = args.thresholds if args.thresholds else DEFAULT_THRESHOLDS
    rows = sweep(labels, probs, thresholds)
    print_table(rows)

    best_f1_row = max(rows, key=lambda r: r["f1"])
    print(f"\nBest F1 at threshold={best_f1_row['threshold']:.2f} (f1={best_f1_row['f1']:.4f})")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model", default="malware_detector_refined_best.pt")
    parser.add_argument("--split", default="val")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--thresholds", type=float, nargs="+", help="Override default threshold list")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
