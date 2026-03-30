import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import Dataset

from data_loader import get_all_subjects, load_sleep_edf_subject
from features import extract_features_all

TARGET_NAMES = ["Wake", "N1", "N2", "N3", "REM"]
NUM_CLASSES = len(TARGET_NAMES)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_subjects(data_dir, max_subjects=None):
    subjects = get_all_subjects(data_dir)
    if not subjects:
        raise FileNotFoundError(
            f"No Sleep-EDF subject pairs were found under {data_dir!r}."
        )
    if max_subjects is not None:
        subjects = subjects[:max_subjects]
    if len(subjects) < 2:
        raise ValueError("At least 2 subjects are required to create train/test splits.")
    return subjects


def split_subjects(subjects, test_subjects=1, val_subjects=0):
    if len(subjects) <= test_subjects:
        raise ValueError("Not enough subjects left after reserving the test split.")

    test_split = subjects[-test_subjects:]
    remaining = subjects[:-test_subjects]

    if val_subjects > 0:
        if len(remaining) <= val_subjects:
            raise ValueError("Not enough subjects left after reserving the validation split.")
        val_split = remaining[-val_subjects:]
        train_split = remaining[:-val_subjects]
    else:
        val_split = []
        train_split = remaining

    return train_split, val_split, test_split


def subject_ids(subjects):
    return [psg.name[:6] for psg, _ in subjects]


def load_feature_dataset(subjects):
    feature_frames = []
    labels = []
    total_subjects = len(subjects)

    for idx, (psg, hyp) in enumerate(subjects, start=1):
        print(f"  [{idx}/{total_subjects}] Processing {psg.name}...")
        X, y, sfreq = load_sleep_edf_subject(psg, hyp)
        feature_frames.append(extract_features_all(X, sfreq))
        labels.append(y)

    features = pd.concat(feature_frames, axis=0, ignore_index=True).to_numpy(dtype=np.float32)
    return features, np.concatenate(labels)


def load_raw_dataset(subjects):
    signals = []
    labels = []
    sfreq = None
    total_subjects = len(subjects)

    for idx, (psg, hyp) in enumerate(subjects, start=1):
        print(f"  [{idx}/{total_subjects}] Processing {psg.name}...")
        X, y, sfreq = load_sleep_edf_subject(psg, hyp)
        signals.append(X.astype(np.float32))
        labels.append(y.astype(np.int64))

    return np.concatenate(signals, axis=0), np.concatenate(labels), sfreq


def load_raw_subject_sequences(subjects):
    sequences = []
    sfreq = None
    total_subjects = len(subjects)

    for idx, (psg, hyp) in enumerate(subjects, start=1):
        print(f"  [{idx}/{total_subjects}] Processing {psg.name}...")
        X, y, sfreq = load_sleep_edf_subject(psg, hyp)
        sequences.append(
            {
                "subject_id": psg.name[:6],
                "X": X.astype(np.float32),
                "y": y.astype(np.int64),
            }
        )

    return sequences, sfreq


def standardize_features(X_train, X_test):
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return ((X_train - mean) / std).astype(np.float32), ((X_test - mean) / std).astype(np.float32)


def normalize_epoch_splits(*arrays):
    train = arrays[0]
    mean = train.mean(axis=(0, 2), keepdims=True)
    std = train.std(axis=(0, 2), keepdims=True)
    std[std < 1e-6] = 1.0
    normalized = [((arr - mean) / std).astype(np.float32) for arr in arrays]
    return normalized, mean, std


def flatten_subject_labels(sequences):
    return np.concatenate([sequence["y"] for sequence in sequences], axis=0)


def normalize_sequence_splits(*sequence_groups):
    train_sequences = sequence_groups[0]
    train_X = np.concatenate([sequence["X"] for sequence in train_sequences], axis=0)
    mean = train_X.mean(axis=(0, 2), keepdims=True)
    std = train_X.std(axis=(0, 2), keepdims=True)
    std[std < 1e-6] = 1.0

    normalized_groups = []
    for group in sequence_groups:
        normalized_group = []
        for sequence in group:
            normalized_group.append(
                {
                    "subject_id": sequence["subject_id"],
                    "X": ((sequence["X"] - mean) / std).astype(np.float32),
                    "y": sequence["y"].copy(),
                }
            )
        normalized_groups.append(normalized_group)

    return normalized_groups, mean, std


def compute_class_weights(y):
    counts = np.bincount(y, minlength=NUM_CLASSES)
    total = counts.sum()
    safe_counts = np.maximum(counts, 1)
    weights = total / (NUM_CLASSES * safe_counts)
    return torch.tensor(weights, dtype=torch.float32)


def estimate_sequence_priors(sequences, smoothing=1.0):
    start_counts = np.full(NUM_CLASSES, smoothing, dtype=np.float64)
    transition_counts = np.full((NUM_CLASSES, NUM_CLASSES), smoothing, dtype=np.float64)

    for sequence in sequences:
        y = sequence["y"]
        if len(y) == 0:
            continue
        start_counts[y[0]] += 1.0
        if len(y) > 1:
            for prev_label, next_label in zip(y[:-1], y[1:]):
                transition_counts[prev_label, next_label] += 1.0

    start_probs = start_counts / start_counts.sum()
    transition_probs = transition_counts / transition_counts.sum(axis=1, keepdims=True)
    return np.log(start_probs), np.log(transition_probs)


def viterbi_decode(log_probs, start_log_probs, transition_log_probs):
    if len(log_probs) == 0:
        return np.array([], dtype=np.int64)

    n_steps, n_states = log_probs.shape
    dp = np.empty((n_steps, n_states), dtype=np.float64)
    backpointers = np.zeros((n_steps, n_states), dtype=np.int64)
    dp[0] = start_log_probs + log_probs[0]

    for step in range(1, n_steps):
        transition_scores = dp[step - 1][:, None] + transition_log_probs
        backpointers[step] = transition_scores.argmax(axis=0)
        dp[step] = transition_scores[backpointers[step], np.arange(n_states)] + log_probs[step]

    states = np.empty(n_steps, dtype=np.int64)
    states[-1] = dp[-1].argmax()
    for step in range(n_steps - 2, -1, -1):
        states[step] = backpointers[step + 1, states[step + 1]]
    return states


def save_results(y_true, y_pred, output_dir, model_name, extra_lines=None):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(
        y_true,
        y_pred,
        labels=np.arange(NUM_CLASSES),
        average="macro",
        zero_division=0,
    )
    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(NUM_CLASSES),
        target_names=TARGET_NAMES,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(NUM_CLASSES))
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=TARGET_NAMES,
        yticklabels=TARGET_NAMES,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix ({model_name})\nAccuracy: {acc:.4f}, Macro F1: {f1:.4f}")
    plt.tight_layout()
    plt.savefig(output_path / "confusion_matrix.png")
    plt.close()

    summary_lines = [
        f"Model: {model_name}",
        f"Accuracy: {acc:.4f}",
        f"Macro F1: {f1:.4f}",
    ]
    if extra_lines:
        summary_lines.extend(extra_lines)
    summary_lines.extend(["", "Classification Report:", report])

    (output_path / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    (output_path / "metrics.json").write_text(
        json.dumps(
            {
                "model": model_name,
                "accuracy": acc,
                "macro_f1": f1,
                "labels": TARGET_NAMES,
                "confusion_matrix": cm.tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved confusion matrix to: {output_path / 'confusion_matrix.png'}")
    print(f"Saved summary to: {output_path / 'summary.txt'}")
    return {"accuracy": acc, "macro_f1": f1, "report": report}


class SleepEpochDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SleepContextDataset(Dataset):
    def __init__(self, sequences, context_epochs=5):
        if context_epochs < 1 or context_epochs % 2 == 0:
            raise ValueError("context_epochs must be a positive odd integer.")

        self.sequences = sequences
        self.context_epochs = context_epochs
        self.context_radius = context_epochs // 2
        self.index = []

        for subject_idx, sequence in enumerate(sequences):
            for epoch_idx in range(len(sequence["y"])):
                self.index.append((subject_idx, epoch_idx))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        subject_idx, epoch_idx = self.index[idx]
        sequence = self.sequences[subject_idx]
        X = sequence["X"]
        y = sequence["y"]

        window_indices = np.arange(
            epoch_idx - self.context_radius,
            epoch_idx + self.context_radius + 1,
        )
        window_indices = np.clip(window_indices, 0, len(y) - 1)
        context_window = X[window_indices]

        return (
            torch.from_numpy(context_window).float(),
            torch.tensor(y[epoch_idx], dtype=torch.long),
            torch.tensor(subject_idx, dtype=torch.long),
            torch.tensor(epoch_idx, dtype=torch.long),
        )
