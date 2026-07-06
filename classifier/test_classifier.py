"""
Evaluate the 34-family classifier on Malimg, the Google dataset, and the
combined set. Writes a classification report, per-class CSV, and confusion
matrix plots for each.

Usage:
    python test_classifier.py --model malware_cnn_traced_34cls.pt \
        --malimg-dir /path/to/malimg --google-dir /path/to/google \
        --classmap artifacts/class_to_idx_20251025.json
"""

import argparse
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import ConcatDataset, DataLoader, random_split
from torchvision import datasets, transforms


def load_class_map(path):
    import json
    with open(path) as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return class_to_idx, idx_to_class


def evaluate_model(model, dataloader, device):
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images.to(device))
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)


def generate_report(preds, labels, dataset_name, num_classes, idx_to_class, out_dir):
    subdir = os.path.join(out_dir, dataset_name.lower())
    os.makedirs(subdir, exist_ok=True)

    class_names = [idx_to_class.get(i, f"Class_{i}") for i in range(num_classes)]
    accuracy = (preds == labels).mean()
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, labels=list(range(num_classes)), average=None, zero_division=0
    )
    _, _, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, labels=list(range(num_classes)), average="macro", zero_division=0
    )

    report = classification_report(
        labels, preds, labels=list(range(num_classes)), target_names=class_names, zero_division=0
    )
    with open(os.path.join(subdir, "classification_report.txt"), "w") as f:
        f.write(f"{dataset_name.upper()} - CLASSIFICATION REPORT\n")
        f.write(f"Test samples: {len(labels)}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Overall accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)\n\n")
        f.write(report)

    pd.DataFrame({
        "class": class_names, "precision": precision, "recall": recall,
        "f1": f1, "support": support,
    }).sort_values("f1", ascending=False).to_csv(
        os.path.join(subdir, "per_class_metrics.csv"), index=False
    )

    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))
    plt.figure(figsize=(20, 18))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix - {dataset_name}")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.xticks(rotation=90); plt.tight_layout()
    plt.savefig(os.path.join(subdir, "confusion_matrix.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print(f"{dataset_name}: accuracy={accuracy:.4f} macro_f1={f1_macro:.4f}")
    return {"dataset": dataset_name, "samples": len(labels), "accuracy": float(accuracy), "macro_f1": float(f1_macro)}


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)
    class_to_idx, idx_to_class = load_class_map(args.classmap)
    num_classes = len(class_to_idx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(args.model, map_location=device).eval()

    eval_tf = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    # Reproduce the same 70/15/15 split used at training time so the test
    # portion here matches what the model never saw during training.
    malimg_ds = datasets.ImageFolder(root=args.malimg_dir, transform=eval_tf)
    total = len(malimg_ds)
    tr, va = int(0.7 * total), int(0.15 * total)
    te = total - tr - va
    _, _, malimg_test = random_split(
        malimg_ds, [tr, va, te], generator=torch.Generator().manual_seed(42)
    )

    google_ds = datasets.ImageFolder(root=args.google_dir, transform=eval_tf)
    local_to_global = [class_to_idx[name] for name in google_ds.classes]
    google_ds.target_transform = lambda i: local_to_global[i]
    google_total = len(google_ds)
    g_train = int(google_total * 0.8)
    _, google_test = random_split(
        google_ds, [g_train, google_total - g_train], generator=torch.Generator().manual_seed(42)
    )

    loaders = {
        "Malimg": DataLoader(malimg_test, batch_size=args.batch_size, shuffle=False),
        "Google": DataLoader(google_test, batch_size=args.batch_size, shuffle=False),
        "Combined": DataLoader(ConcatDataset([malimg_test, google_test]), batch_size=args.batch_size, shuffle=False),
    }

    summaries = []
    for name, loader in loaders.items():
        preds, labels = evaluate_model(model, loader, device)
        summaries.append(generate_report(preds, labels, name, num_classes, idx_to_class, args.output_dir))

    pd.DataFrame(summaries).to_csv(os.path.join(args.output_dir, "dataset_comparison.csv"), index=False)
    print(f"All results saved to {args.output_dir}/")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to traced classifier .pt file")
    parser.add_argument("--malimg-dir", required=True)
    parser.add_argument("--google-dir", required=True)
    parser.add_argument("--classmap", required=True, help="Path to class_to_idx JSON produced during training")
    parser.add_argument("--output-dir", default="evaluation_results")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
