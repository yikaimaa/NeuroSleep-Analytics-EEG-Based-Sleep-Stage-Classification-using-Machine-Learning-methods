from argparse import ArgumentParser
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from training_utils import (
    NUM_CLASSES,
    TARGET_NAMES,
    SleepContextDataset,
    compute_class_weights,
    estimate_sequence_priors,
    flatten_subject_labels,
    load_raw_subject_sequences,
    normalize_sequence_splits,
    save_results,
    seed_everything,
    select_subjects,
    split_subjects,
    subject_ids,
    viterbi_decode,
)


def build_deep_arg_parser(default_output_dir):
    parser = ArgumentParser()
    parser.add_argument("--data-dir", default="data/sleep-edf")
    parser.add_argument("--output-dir", default=default_output_dir)
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--test-subjects", type=int, default=1)
    parser.add_argument("--val-subjects", type=int, default=1)
    parser.add_argument("--context-epochs", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--transition-smoothing", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
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


def batch_to_device(batch, device):
    batch_X, batch_y, batch_subject_idx, batch_epoch_idx = batch
    return (
        batch_X.to(device),
        batch_y.to(device),
        batch_subject_idx.cpu().numpy(),
        batch_epoch_idx.cpu().numpy(),
    )


def numpy_log_softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def decode_with_sequence_prior(logits, targets, subject_indices, epoch_indices, sequence_prior):
    if sequence_prior is None:
        predictions = logits.argmax(axis=1)
        return targets, predictions, None

    start_log_probs, transition_log_probs = sequence_prior
    ordered_targets = []
    ordered_predictions = []

    for subject_idx in np.unique(subject_indices):
        mask = subject_indices == subject_idx
        subject_logits = logits[mask]
        subject_targets = targets[mask]
        subject_epochs = epoch_indices[mask]
        order = np.argsort(subject_epochs)

        subject_log_probs = numpy_log_softmax(subject_logits[order])
        subject_predictions = viterbi_decode(
            subject_log_probs,
            start_log_probs,
            transition_log_probs,
        )
        ordered_targets.append(subject_targets[order])
        ordered_predictions.append(subject_predictions)

    y_true = np.concatenate(ordered_targets)
    y_pred = np.concatenate(ordered_predictions)
    sequence_macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=np.arange(NUM_CLASSES),
        average="macro",
        zero_division=0,
    )
    return y_true, y_pred, sequence_macro_f1


def make_progress(iterable, total, desc):
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        leave=False,
        dynamic_ncols=True,
        disable=not sys.stderr.isatty(),
    )


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    max_batches=None,
    progress_label="Train",
):
    model.train()
    total_loss = 0.0
    total_samples = 0
    total_batches = len(loader)
    if max_batches is not None:
        total_batches = min(total_batches, max_batches)

    progress = make_progress(loader, total_batches, progress_label)
    for batch_idx, batch in enumerate(progress, start=1):
        batch_X, batch_y, _, _ = batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_X)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        batch_size = batch_y.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        progress.set_postfix(
            batch=batch_idx,
            avg_loss=f"{(total_loss / max(total_samples, 1)):.4f}",
        )

        if max_batches is not None and batch_idx >= max_batches:
            break

    progress.close()
    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate_model(
    model,
    loader,
    criterion,
    device,
    sequence_prior=None,
    max_batches=None,
    progress_label="Eval",
):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_targets = []
    all_predictions = []
    all_logits = []
    all_subject_indices = []
    all_epoch_indices = []
    total_batches = len(loader)
    if max_batches is not None:
        total_batches = min(total_batches, max_batches)

    progress = make_progress(loader, total_batches, progress_label)
    for batch_idx, batch in enumerate(progress, start=1):
        batch_X, batch_y, batch_subject_idx, batch_epoch_idx = batch_to_device(batch, device)

        logits = model(batch_X)
        loss = criterion(logits, batch_y)

        preds = logits.argmax(dim=1)
        batch_size = batch_y.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        all_targets.append(batch_y.cpu().numpy())
        all_predictions.append(preds.cpu().numpy())
        all_logits.append(logits.cpu().numpy())
        all_subject_indices.append(batch_subject_idx)
        all_epoch_indices.append(batch_epoch_idx)
        progress.set_postfix(
            batch=batch_idx,
            avg_loss=f"{(total_loss / max(total_samples, 1)):.4f}",
        )

        if max_batches is not None and batch_idx >= max_batches:
            break

    progress.close()
    y_true_raw = np.concatenate(all_targets)
    y_pred_raw = np.concatenate(all_predictions)
    logits_raw = np.concatenate(all_logits)
    subject_indices = np.concatenate(all_subject_indices)
    epoch_indices = np.concatenate(all_epoch_indices)

    avg_loss = total_loss / max(total_samples, 1)
    raw_macro_f1 = f1_score(
        y_true_raw,
        y_pred_raw,
        labels=np.arange(NUM_CLASSES),
        average="macro",
        zero_division=0,
    )
    y_true_final, y_pred_final, sequence_macro_f1 = decode_with_sequence_prior(
        logits_raw,
        y_true_raw,
        subject_indices,
        epoch_indices,
        sequence_prior,
    )
    selection_f1 = sequence_macro_f1 if sequence_macro_f1 is not None else raw_macro_f1
    return {
        "loss": avg_loss,
        "selection_f1": selection_f1,
        "raw_macro_f1": raw_macro_f1,
        "sequence_macro_f1": sequence_macro_f1,
        "y_true": y_true_final,
        "y_pred": y_pred_final,
    }


def build_dataloader(dataset, batch_size, shuffle, num_workers, device):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


def run_deep_training(args, model_factory, model_name):
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
    train_sequences, sfreq = load_raw_subject_sequences(train_subjects)

    print("\n[Stage 2] Loading validation data...")
    val_sequences, _ = load_raw_subject_sequences(val_subjects)

    print("\n[Stage 3] Loading test data...")
    test_sequences, _ = load_raw_subject_sequences(test_subjects)

    (train_sequences, val_sequences, test_sequences), norm_mean, norm_std = normalize_sequence_splits(
        train_sequences,
        val_sequences,
        test_sequences,
    )

    train_labels = flatten_subject_labels(train_sequences)
    val_labels = flatten_subject_labels(val_sequences)
    test_labels = flatten_subject_labels(test_sequences)
    print(
        f"\nTraining epochs: {len(train_labels)}, Validation epochs: {len(val_labels)}, Test epochs: {len(test_labels)}"
    )
    print(f"Class distribution (train): {np.unique(train_labels, return_counts=True)}")
    print(f"Context window size (epochs): {args.context_epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Early stopping patience: {args.patience}")
    if args.val_subjects < 2:
        print(
            "Validation currently uses 1 subject. If metrics remain noisy, prefer increasing "
            "--val-subjects before increasing --epochs."
        )

    train_dataset = SleepContextDataset(train_sequences, context_epochs=args.context_epochs)
    val_dataset = SleepContextDataset(val_sequences, context_epochs=args.context_epochs)
    test_dataset = SleepContextDataset(test_sequences, context_epochs=args.context_epochs)

    device = resolve_device(args.device)
    print(f"\n[Stage 4] Training on device: {device}")

    train_loader = build_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        device=device,
    )
    val_loader = build_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        device=device,
    )
    test_loader = build_dataloader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        device=device,
    )

    in_channels = train_sequences[0]["X"].shape[1]
    model = model_factory(in_channels=in_channels, num_classes=NUM_CLASSES).to(device)
    class_weights = compute_class_weights(train_labels).to(device)
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
    sequence_prior = estimate_sequence_priors(
        train_sequences,
        smoothing=args.transition_smoothing,
    )

    best_state = None
    best_val_f1 = -1.0
    best_epoch = 0
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_checkpoint_path = output_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            max_batches=args.max_train_batches,
            progress_label=f"Epoch {epoch:02d} train",
        )
        val_metrics = evaluate_model(
            model,
            val_loader,
            criterion,
            device,
            sequence_prior=sequence_prior,
            max_batches=args.max_eval_batches,
            progress_label=f"Epoch {epoch:02d} val",
        )
        scheduler.step(val_metrics["selection_f1"])

        val_sequence_f1 = val_metrics["sequence_macro_f1"]
        val_metric_message = (
            f"val_seq_macro_f1={val_sequence_f1:.4f}"
            if val_sequence_f1 is not None
            else f"val_macro_f1={val_metrics['raw_macro_f1']:.4f}"
        )
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_metrics['loss']:.4f} | "
            f"val_raw_macro_f1={val_metrics['raw_macro_f1']:.4f} | {val_metric_message} | "
            f"lr={current_lr:.2e}"
        )

        if val_metrics["selection_f1"] > best_val_f1 + 1e-4:
            best_val_f1 = val_metrics["selection_f1"]
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": best_epoch,
                    "best_val_f1": best_val_f1,
                    "best_val_loss": best_val_loss,
                    "model_state_dict": best_state,
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
                    "sequence_prior": {
                        "start_log_probs": sequence_prior[0],
                        "transition_log_probs": sequence_prior[1],
                    },
                    "args": vars(args),
                },
                best_checkpoint_path,
            )
            print(
                f"Best checkpoint updated at epoch {best_epoch}: "
                f"val_f1={best_val_f1:.4f}, val_loss={best_val_loss:.4f}"
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping triggered after epoch {epoch}.")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")

    model.load_state_dict(best_state)
    print(
        f"\nTraining complete. Best epoch: {best_epoch} | "
        f"Best val F1: {best_val_f1:.4f} | Best val loss: {best_val_loss:.4f}"
    )

    print("\n[Stage 5] Evaluating best checkpoint on the test set...")
    test_metrics = evaluate_model(
        model,
        test_loader,
        criterion,
        device,
        sequence_prior=sequence_prior,
        max_batches=args.max_eval_batches,
        progress_label="Test",
    )
    extra_lines = [
        f"Best validation selection F1: {best_val_f1:.4f}",
        f"Best epoch: {best_epoch}",
        f"Test loss: {test_metrics['loss']:.4f}",
        f"Test raw macro F1: {test_metrics['raw_macro_f1']:.4f}",
        f"Sampling rate (Hz): {sfreq}",
        f"Context epochs: {args.context_epochs}",
        f"Train subjects: {subject_ids(train_subjects)}",
        f"Validation subjects: {subject_ids(val_subjects)}",
        f"Test subjects: {subject_ids(test_subjects)}",
    ]
    if test_metrics["sequence_macro_f1"] is not None:
        extra_lines.append(f"Test sequence-prior macro F1: {test_metrics['sequence_macro_f1']:.4f}")
        extra_lines.append(
            "Sequence prior: Viterbi decoding with a transition matrix estimated from training labels."
        )
    if args.max_train_batches is not None:
        extra_lines.append(f"Smoke train batch limit: {args.max_train_batches}")
    if args.max_eval_batches is not None:
        extra_lines.append(f"Smoke eval batch limit: {args.max_eval_batches}")

    metrics = save_results(
        test_metrics["y_true"],
        test_metrics["y_pred"],
        args.output_dir,
        model_name,
        extra_lines=extra_lines,
    )

    final_checkpoint_path = output_dir / "model.pt"
    torch.save(
        {
            "epoch": best_epoch,
            "best_val_f1": best_val_f1,
            "best_val_loss": best_val_loss,
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
            "sequence_prior": {
                "start_log_probs": sequence_prior[0],
                "transition_log_probs": sequence_prior[1],
            },
            "args": vars(args),
        },
        final_checkpoint_path,
    )
    print(f"Saved best checkpoint during training to: {best_checkpoint_path}")
    print(f"Saved final selected model to: {final_checkpoint_path}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1-score: {metrics['macro_f1']:.4f}")
    print(f"Raw macro F1-score: {test_metrics['raw_macro_f1']:.4f}")
    if test_metrics["sequence_macro_f1"] is not None:
        print(f"Sequence-prior macro F1-score: {test_metrics['sequence_macro_f1']:.4f}")
    return metrics
