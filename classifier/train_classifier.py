"""
Train the 34-family malware classifier.

Combines two image datasets into a single 34-class problem:
  - Malimg (25 classes, byteplot images already grouped by family)
  - A Google-sourced dataset covering 9 additional families

Both datasets are converted to grayscale byteplot-style images and fed
into a small CNN. The Google folder labels get remapped onto the tail
end of the combined label space (25..33) so both sources share one
classifier head.

Usage:
    python train_classifier.py --malimg-dir /path/to/malimg --google-dir /path/to/google
"""

import argparse
import json
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, random_split
from torchvision import datasets, transforms
from tqdm.auto import tqdm

GOOGLE_CLASS_NAMES = [
    "Ramnit", "Lollipop", "Kelihos_ver3", "Vundo",
    "Simda", "Tracur", "Kelihos_ver1", "Obfuscator.ACY", "Gatak",
]


class MalwareCNN(nn.Module):
    """Small CNN over 256x256 grayscale byteplot images."""

    def __init__(self, num_classes=34):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 64 * 64, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 256 -> 128
        x = self.pool(F.relu(self.conv2(x)))  # 128 -> 64
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def build_combined_dataset(malimg_dir, google_dir, transform):
    """Load both datasets and remap Google's local labels onto a shared 34-class space."""
    malimg_ds = datasets.ImageFolder(root=malimg_dir, transform=transform)
    malimg_class_to_idx = dict(malimg_ds.class_to_idx)
    print(f"Malimg: {len(malimg_ds)} samples across {len(malimg_ds.classes)} classes")

    google_global_map = {25 + i: name for i, name in enumerate(GOOGLE_CLASS_NAMES)}
    combined_class_to_idx = {
        **malimg_class_to_idx,
        **{name: idx for idx, name in google_global_map.items()},
    }

    google_ds = datasets.ImageFolder(root=google_dir, transform=transform)
    missing = [name for name in google_ds.classes if name not in combined_class_to_idx]
    if missing:
        raise ValueError(
            f"Google folder names not found in combined class map: {missing}. "
            f"Expected: {GOOGLE_CLASS_NAMES}"
        )

    local_to_global = [combined_class_to_idx[name] for name in google_ds.classes]
    google_ds.target_transform = lambda local_idx: local_to_global[local_idx]
    print(f"Google: {len(google_ds)} samples across {len(google_ds.classes)} classes")

    return ConcatDataset([malimg_ds, google_ds]), combined_class_to_idx


def train(args):
    os.makedirs(args.artifacts_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    combined, class_to_idx = build_combined_dataset(args.malimg_dir, args.google_dir, transform)

    classmap_path = os.path.join(
        args.artifacts_dir, f"class_to_idx_{datetime.now().strftime('%Y%m%d')}.json"
    )
    with open(classmap_path, "w") as f:
        json.dump(class_to_idx, f, indent=2)
    print(f"Saved class map: {classmap_path}")

    n = len(combined)
    train_size = int(0.7 * n)
    val_size = int(0.15 * n)
    test_size = n - train_size - val_size
    train_ds, val_ds, test_ds = random_split(
        combined, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"Split -> train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    # num_workers=0 avoids a DataLoader deadlock we hit on Windows with worker processes.
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = MalwareCNN(num_classes=34).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [train]", unit="batch")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix(loss=f"{running_loss / max(1, pbar.n):.4f}", acc=f"{correct / max(1, total):.4f}")

        train_acc = correct / max(1, total)

        model.eval()
        val_correct, val_total, val_loss = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [val]", unit="batch", leave=False):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / max(1, val_total)
        print(f"Epoch {epoch + 1}/{args.epochs} | train_acc={train_acc:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            example_input = torch.randn(1, 1, 256, 256).to(device)
            traced = torch.jit.trace(model, example_input)
            traced.save(args.output_path)
            print(f"New best (val_acc={val_acc:.4f}) -> saved {args.output_path}")

    print(f"Training complete. Best val_acc={best_val_acc:.4f}. Model at {args.output_path}")

    # Quick sanity check on the held-out split (full evaluation lives in test_classifier.py)
    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing", unit="batch"):
            images, labels = images.to(device), labels.to(device)
            test_correct += (model(images).argmax(1) == labels).sum().item()
            test_total += labels.size(0)
    print(f"Test accuracy: {test_correct / max(1, test_total):.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--malimg-dir", required=True, help="Path to Malimg dataset root (ImageFolder layout)")
    parser.add_argument("--google-dir", required=True, help="Path to Google dataset root (ImageFolder layout)")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Where to save the class map JSON")
    parser.add_argument("--output-path", default="malware_cnn_traced_34cls.pt", help="Output TorchScript model path")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
