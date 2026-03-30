from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from training_utils import (
    NUM_CLASSES,
    SleepEpochDataset,
    TARGET_NAMES,
    compute_class_weights,
    load_raw_dataset,
    normalize_epoch_splits,
    save_results,
    seed_everything,
    select_subjects,
    split_subjects,
    subject_ids,
)


def build_deep_arg_parser(default_output_dir):
    parser = ArgumentParser()
    parser.add_argument("--data-dir", default="data/sleep-edf")
    parser.add_argument("--output-dir", default=default_output_dir)
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--test-subjects", type=int, default=1)
    parser.add_argument("--val-subjects", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def resolve_device(device_name):
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_name)


def train_one_epoch(model, loader, criterion, optimizer, device, max_batches=None):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch_idx, (batch_X, batch_y) in enumerate(loader, start=1):
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_X)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        batch_size = batch_y.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        if max_batches is not None and batch_idx >= max_batches:
            break

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate_model(model, loader, criterion, device, max_batches=None):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_targets = []
    all_predictions = []

    for batch_idx, (batch_X, batch_y) in enumerate(loader, start=1):
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        logits = model(batch_X)
        loss = criterion(logits, batch_y)

        preds = logits.argmax(dim=1)
        batch_size = batch_y.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        all_targets.append(batch_y.cpu().numpy())
        all_predictions.append(preds.cpu().numpy())

        if max_batches is not None and batch_idx >= max_batches:
            break

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_predictions)
    avg_loss = total_loss / max(total_samples, 1)
    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=np.arange(NUM_CLASSES),
        average="macro",
        zero_division=0,
    )
    return avg_loss, macro_f1, y_true, y_pred


def run_deep_training(args, model_factory, model_name):
    seed_everything(args.seed)

    subjects = select_subjects(args.data_dir, args.max_subjects)
    train_subjects, val_subjects, test_subjects = split_subjects(
        subjects,
        test_subjects=args.test_subjects,
        val_subjects=args.val_subjects,
    )

    print(f"Total subjects found: {len(subjects)}")
    print(f"Train subjects: {subject_ids(train_subjects)}")
    print(f"Validation subjects: {subject_ids(val_subjects)}")
    print(f"Test subjects: {subject_ids(test_subjects)}")

    print("\n[Stage 1] Loading training data...")
    X_train, y_train, sfreq = load_raw_dataset(train_subjects)

    print("\n[Stage 2] Loading validation data...")
    X_val, y_val, _ = load_raw_dataset(val_subjects)

    print("\n[Stage 3] Loading test data...")
    X_test, y_test, _ = load_raw_dataset(test_subjects)

    (X_train, X_val, X_test), norm_mean, norm_std = normalize_epoch_splits(
        X_train,
        X_val,
        X_test,
    )

    print(
        f"\nTraining set size: {X_train.shape}, Validation set size: {X_val.shape}, Test set size: {X_test.shape}"
    )
    print(f"Class distribution (train): {np.unique(y_train, return_counts=True)}")

    train_loader = DataLoader(
        SleepEpochDataset(X_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        SleepEpochDataset(X_val, y_val),
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        SleepEpochDataset(X_test, y_test),
        batch_size=args.batch_size,
        shuffle=False,
    )

    device = resolve_device(args.device)
    print(f"\n[Stage 4] Training on device: {device}")

    model = model_factory(in_channels=X_train.shape[1], num_classes=NUM_CLASSES).to(device)
    class_weights = compute_class_weights(y_train).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=1,
    )

    best_state = None
    best_val_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            max_batches=args.max_train_batches,
        )
        val_loss, val_f1, _, _ = evaluate_model(
            model,
            val_loader,
            criterion,
            device,
            max_batches=args.max_eval_batches,
        )
        scheduler.step(val_f1)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_macro_f1={val_f1:.4f}"
        )

        if val_f1 > best_val_f1 + 1e-4:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping triggered after epoch {epoch}.")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")

    model.load_state_dict(best_state)

    print("\n[Stage 5] Evaluating best checkpoint on the test set...")
    test_loss, test_f1, y_true, y_pred = evaluate_model(
        model,
        test_loader,
        criterion,
        device,
        max_batches=args.max_eval_batches,
    )
    extra_lines = [
        f"Best validation macro F1: {best_val_f1:.4f}",
        f"Best epoch: {best_epoch}",
        f"Test loss: {test_loss:.4f}",
        f"Test macro F1: {test_f1:.4f}",
        f"Sampling rate (Hz): {sfreq}",
        f"Train subjects: {subject_ids(train_subjects)}",
        f"Validation subjects: {subject_ids(val_subjects)}",
        f"Test subjects: {subject_ids(test_subjects)}",
    ]
    if args.max_train_batches is not None:
        extra_lines.append(f"Smoke train batch limit: {args.max_train_batches}")
    if args.max_eval_batches is not None:
        extra_lines.append(f"Smoke eval batch limit: {args.max_eval_batches}")

    metrics = save_results(
        y_true,
        y_pred,
        args.output_dir,
        model_name,
        extra_lines=extra_lines,
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "normalization_mean": norm_mean,
            "normalization_std": norm_std,
            "sampling_rate_hz": sfreq,
            "target_names": TARGET_NAMES,
            "model_name": model_name,
            "subjects": {
                "train": subject_ids(train_subjects),
                "val": subject_ids(val_subjects),
                "test": subject_ids(test_subjects),
            },
            "args": vars(args),
        },
        Path(args.output_dir) / "model.pt",
    )
    print(f"Saved model checkpoint to: {Path(args.output_dir) / 'model.pt'}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1-score: {metrics['macro_f1']:.4f}")
    return metrics
