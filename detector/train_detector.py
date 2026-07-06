"""
Train the binary malware/benign detector.

Fine-tunes an ImageNet-pretrained EfficientNet-B1 on grayscale byteplot
images. Handles class imbalance with weighted sampling + focal loss, and
uses a OneCycle learning-rate schedule for faster convergence.

Expects a data directory laid out as:
    data_dir/train/<benign|malware>/*.png
    data_dir/val/<benign|malware>/*.png

Usage:
    python train_detector.py --data-dir /path/to/dataset
"""

import argparse
import json
import os
import random
from collections import Counter
from datetime import datetime

import torch
from sklearn.metrics import precision_recall_fscore_support
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.models import EfficientNet_B1_Weights, efficientnet_b1
from tqdm.auto import tqdm


class FocalLoss(nn.Module):
    """Down-weights easy examples so training focuses on the hard cases."""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        return (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()


def get_balanced_subset(dataset, max_per_class, desc="dataset"):
    """Cap each class at max_per_class samples, keeping the split balanced."""
    class_indices = {0: [], 1: []}
    for idx in tqdm(range(len(dataset)), desc=f"Indexing {desc}", unit="img"):
        _, label = dataset.imgs[idx]
        if len(class_indices[label]) < max_per_class:
            class_indices[label].append(idx)
        if all(len(v) >= max_per_class for v in class_indices.values()):
            break

    selected = class_indices[0] + class_indices[1]
    random.shuffle(selected)
    counts = Counter(dataset.imgs[i][1] for i in selected)
    for class_idx, class_name in enumerate(dataset.classes):
        print(f"  {class_name}: {counts[class_idx]:,} images")
    return Subset(dataset, selected), selected


def build_model(device):
    model = efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V1)

    # Swap the first conv layer to accept single-channel (grayscale) input,
    # initializing it by averaging the pretrained RGB weights.
    original_conv = model.features[0][0]
    model.features[0][0] = nn.Conv2d(
        1, original_conv.out_channels, kernel_size=original_conv.kernel_size,
        stride=original_conv.stride, padding=original_conv.padding, bias=False,
    )
    with torch.no_grad():
        model.features[0][0].weight = nn.Parameter(original_conv.weight.mean(dim=1, keepdim=True))

    model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.classifier[1].in_features, 2))
    return model.to(device)


def train(args):
    os.makedirs(args.results_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.3),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15), shear=5),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    val_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    train_ds_full = datasets.ImageFolder(os.path.join(args.data_dir, "train"), transform=train_transform)
    print(f"Training pool: {len(train_ds_full):,} images available")
    train_ds, train_indices = get_balanced_subset(train_ds_full, args.max_per_class, "train")

    val_ds_full = datasets.ImageFolder(os.path.join(args.data_dir, "val"), transform=val_transform)
    val_ds, _ = get_balanced_subset(val_ds_full, args.max_per_class // 10, "val")

    train_labels = [train_ds_full.imgs[i][1] for i in train_indices]
    class_counts = Counter(train_labels)
    class_weights = {c: 1.0 / n for c, n in class_counts.items()}
    sampler = WeightedRandomSampler(
        weights=[class_weights[label] for label in train_labels],
        num_samples=len(train_labels), replacement=True,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = build_model(device)
    print(f"EfficientNet-B1 loaded, {sum(p.numel() for p in model.parameters()):,} parameters")

    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = OneCycleLR(
        optimizer, max_lr=args.lr * 5, epochs=args.epochs, steps_per_epoch=len(train_loader),
        pct_start=0.3, anneal_strategy="cos", div_factor=25.0, final_div_factor=10000.0,
    )
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    best_f1, best_val_acc = 0.0, 0.0
    history = {"train_loss": [], "train_acc": [], "val_acc": [], "val_f1": [], "val_precision": [], "val_recall": []}

    for epoch in range(args.epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [train]", unit="batch"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / len(train_loader)
        train_acc = 100 * correct / total

        model.eval()
        val_correct, val_total, all_preds, all_labels = 0, 0, [], []
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [val]", unit="batch", leave=False):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = 100 * val_correct / val_total
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="binary", pos_label=1, zero_division=0
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(f1)
        history["val_precision"].append(precision)
        history["val_recall"].append(recall)

        print(f"Epoch {epoch + 1}/{args.epochs} | train_acc={train_acc:.2f}% val_acc={val_acc:.2f}% "
              f"precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}")

        if f1 > best_f1 or (f1 == best_f1 and val_acc > best_val_acc):
            best_f1, best_val_acc = f1, val_acc
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), "val_acc": val_acc,
                "val_f1": f1, "precision": precision, "recall": recall,
                "architecture": "EfficientNet-B1",
            }, args.output_path)
            print(f"New best (f1={f1:.4f}) -> saved {args.output_path}")

    print(f"Training complete. best_val_acc={best_val_acc:.2f}% best_f1={best_f1:.4f}")

    history_path = os.path.join(args.results_dir, f"training_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History saved: {history_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Root containing train/ and val/ ImageFolder splits")
    parser.add_argument("--output-path", default="malware_detector_refined_best.pt")
    parser.add_argument("--results-dir", default="detector_results")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-per-class", type=int, default=150000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
