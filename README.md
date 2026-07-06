# Megalodon — Malware Classification & Detection Pipeline

This repo contains the machine learning component of **Megalodon**, a
capstone project built by a 4-person team as part of a BSc Cybersecurity
Engineering program. Megalodon is a static-analysis pipeline that flags
uploaded files as benign or malicious and, if malicious, identifies the
malware family.

**This repo covers my individual contribution to the project: the ML
model development, training, and evaluation.** The full system also
includes an API/processing pipeline, a web dashboard, and business-model
work built by my teammates — those pieces aren't part of this repo.

| Component | Contributor |
|---|---|
| ML models (this repo) + security risk assessment | Me |
| API / Redis processing pipeline | Omar |
| Web dashboard | Abdulrahman |
| Design / business model | Sultan |

## What's in here

Two models work together in the pipeline:

1. **Detector** — binary classifier (benign vs. malicious), EfficientNet-B1
   fine-tuned on grayscale byteplot images generated from ~300K samples.
2. **Classifier** — 34-family classifier, a small CNN trained on ~20K
   samples combining the [Malimg dataset](https://www.kaggle.com/datasets/manmandes/malimg)
   (25 families) with an additional 9-family dataset.

Both take a suspicious file, convert it to a byteplot image, and run
inference. The detector runs first; if it flags malware, the classifier
identifies the likely family.

```
classifier/
├── train_classifier.py    # trains the 34-class CNN
└── test_classifier.py     # evaluates on Malimg / Google / combined test sets
detector/
├── train_detector.py      # fine-tunes EfficientNet-B1 for binary detection
└── test_detector.py       # evaluates at the default 0.5 threshold
evaluation/
└── threshold_analysis.py  # sweeps decision thresholds, reports the tradeoffs
results/                   # saved reports from our runs
```

## Results

### Classifier (34 families)

| Metric | Score |
|---|---|
| Accuracy | 96.64% |
| Recall | 96.64% |
| F1-score | 96.63% |

Full per-class report: [`results/classifier_classification_report.txt`](results/classifier_classification_report.txt)

### Detector (binary)

The detector's decision threshold trades precision against recall — raising
it cuts false alarms but misses more malware. We swept a range of
thresholds to pick an operating point rather than defaulting to 0.5:

| Threshold | Accuracy | Precision | Recall | F1 | Missed malware |
|---|---|---|---|---|---|
| 0.30 | 89.74% | 90.55% | 97.82% | 94.05% | 3,762 |
| 0.40 | 90.12% | 94.00% | 94.06% | 94.03% | 10,265 |
| 0.50 | 86.90% | 96.78% | 87.08% | 91.67% | 22,346 |
| 0.60 | 80.76% | 98.27% | 78.14% | 87.06% | 37,803 |
| 0.70 | 73.86% | 99.12% | 69.04% | 81.39% | 53,535 |

Full sweep: [`results/threshold_analysis.txt`](results/threshold_analysis.txt)
Default-threshold report: [`results/detector_test_report.txt`](results/detector_test_report.txt)

Since a missed malware sample is more costly than a false alarm in a
security context, we favored the lower end of this range over raw
accuracy.

## Running it

```bash
pip install -r requirements.txt

# Train the classifier
python classifier/train_classifier.py --malimg-dir /path/to/malimg --google-dir /path/to/google

# Evaluate it
python classifier/test_classifier.py --model malware_cnn_traced_34cls.pt \
    --malimg-dir /path/to/malimg --google-dir /path/to/google \
    --classmap artifacts/class_to_idx_YYYYMMDD.json

# Train the detector
python detector/train_detector.py --data-dir /path/to/dataset

# Evaluate it, and sweep thresholds
python detector/test_detector.py --data-dir /path/to/dataset --model malware_detector_refined_best.pt
python evaluation/threshold_analysis.py --data-dir /path/to/dataset --model malware_detector_refined_best.pt
```

Datasets and trained weights aren't included in this repo (large binary
files) — the Malimg dataset is public on Kaggle; the detector's dataset
and trained model weights are available on request.

## Notes on the architecture

- **Classifier**: 2 conv layers (32, 64 channels) + max pooling, feeding
  into a 256-unit FC layer and a 34-way softmax output. Input images are
  256x256 grayscale.
- **Detector**: EfficientNet-B1 with the first conv layer modified to
  accept single-channel input (weights initialized by averaging the
  pretrained RGB channels), and a dropout + linear head for binary
  classification. Trained with focal loss and weighted sampling to
  handle class imbalance, and a OneCycle LR schedule. Input images are
  224x224 grayscale.
