"""
Evaluate the binary malware/benign detector at the default 0.5 threshold.

Writes a text report plus a confusion matrix / ROC / per-class metrics
figure to --results-dir.

Usage:
    python test_detector.py --data-dir /path/to/dataset --model malware_detector_refined_best.pt
"""

import argparse
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score, auc, classification_report, confusion_matrix,
    precision_recall_fscore_support, roc_curve,
)
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b1


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
    model = model.to(device).eval()
    print(f"Loaded model from epoch {checkpoint['epoch']} "
          f"(val_acc={checkpoint['val_acc']:.2f}%, val_f1={checkpoint['val_f1']:.4f})")
    return model


def run_inference(model, loader, device):
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))
            probs = torch.softmax(outputs, dim=1)
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def run(args):
    os.makedirs(args.results_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    test_dataset = datasets.ImageFolder(os.path.join(args.data_dir, args.split), transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    print(f"Loaded {len(test_dataset):,} images, classes: {test_dataset.classes}")

    model = load_model(args.model, device)
    true_labels, predictions, probabilities = run_inference(model, test_loader, device)

    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, support = precision_recall_fscore_support(true_labels, predictions, average=None)
    avg_precision, avg_recall, avg_f1, _ = precision_recall_fscore_support(true_labels, predictions, average="binary")
    cm = confusion_matrix(true_labels, predictions)
    tn, fp, fn, tp = cm.ravel()
    fpr, tpr, _ = roc_curve(true_labels, probabilities[:, 1])
    roc_auc = auc(fpr, tpr)

    print(f"Accuracy: {accuracy * 100:.2f}%  F1: {avg_f1:.4f}  ROC AUC: {roc_auc:.4f}")
    print(f"TN={tn:,} FP={fp:,} FN={fn:,} TP={tp:,}")

    report_text = classification_report(true_labels, predictions, target_names=test_dataset.classes, digits=4)
    print(report_text)

    fig = plt.figure(figsize=(16, 12))
    ax1 = plt.subplot(2, 2, 1)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=test_dataset.classes, yticklabels=test_dataset.classes)
    ax1.set_title("Confusion Matrix"); ax1.set_xlabel("Predicted"); ax1.set_ylabel("Actual")

    ax2 = plt.subplot(2, 2, 2)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Greens", xticklabels=test_dataset.classes, yticklabels=test_dataset.classes)
    ax2.set_title("Normalized Confusion Matrix"); ax2.set_xlabel("Predicted"); ax2.set_ylabel("Actual")

    ax3 = plt.subplot(2, 2, 3)
    x = np.arange(len(test_dataset.classes)); width = 0.25
    ax3.bar(x - width, precision, width, label="Precision")
    ax3.bar(x, recall, width, label="Recall")
    ax3.bar(x + width, f1, width, label="F1")
    ax3.set_xticks(x); ax3.set_xticklabels(test_dataset.classes); ax3.set_ylim([0, 1.05])
    ax3.set_title("Per-Class Metrics"); ax3.legend()

    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(fpr, tpr, label=f"ROC curve (AUC={roc_auc:.4f})")
    ax4.plot([0, 1], [0, 1], linestyle="--", label="Random")
    ax4.set_xlabel("False Positive Rate"); ax4.set_ylabel("True Positive Rate")
    ax4.set_title("ROC Curve"); ax4.legend(loc="lower right")

    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(args.results_dir, f"test_results_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {plot_path}")

    report_path = os.path.join(args.results_dir, f"test_report_{timestamp}.txt")
    with open(report_path, "w") as f:
        f.write("MALWARE DETECTOR - TEST REPORT\n\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Test date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Test images: {len(test_dataset):,}\n\n")
        f.write(f"Overall accuracy: {accuracy * 100:.2f}%\n\n")
        for i, class_name in enumerate(test_dataset.classes):
            f.write(f"{class_name}: precision={precision[i]:.4f} recall={recall[i]:.4f} "
                    f"f1={f1[i]:.4f} support={support[i]:,}\n")
        f.write(f"\nConfusion matrix: TN={tn:,} FP={fp:,} FN={fn:,} TP={tp:,}\n")
        f.write(f"ROC AUC: {roc_auc:.4f}\n\n")
        f.write(report_text)
    print(f"Saved report: {report_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model", default="malware_detector_refined_best.pt")
    parser.add_argument("--split", default="val", help="Subfolder of data-dir to evaluate on")
    parser.add_argument("--results-dir", default="test_results")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
