import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from training_utils import (
    SleepContextDataset,
    TARGET_NAMES,
    build_cv_folds,
    compute_classification_metrics,
    compute_class_weights,
    estimate_hmm_parameters,
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
    parser.add_argument("--test-subjects", type=int, default=2)
    parser.add_argument("--cv-folds", type=int, default=4)
    parser.add_argument(
        "--val-subjects",
        type=int,
        default=None,
        help="Deprecated. Validation subjects are now selected via subject-level cross-validation.",
    )
    parser.add_argument("--context-epochs", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--transition-smoothing", type=float, default=1.0)
    parser.add_argument("--transition-weight", type=float, default=1.0)
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


def build_loader(dataset, batch_size, shuffle, num_workers=0):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def create_training_components(model, y_train, args, device):
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
    return criterion, optimizer, scheduler


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


def metric_snapshot(metrics):
    if metrics is None:
        return None
    return {
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
    }


def decode_with_hmm(
    logits,
    targets,
    subject_indices,
    epoch_indices,
    hmm_parameters,
    transition_weight=1.0,
):
    if hmm_parameters is None:
        predictions = logits.argmax(axis=1)
        return targets, predictions, None

    start_log_probs, transition_log_probs = hmm_parameters
    weighted_transition_log_probs = transition_weight * transition_log_probs

    ordered_targets = []
    ordered_predictions = []

    for subject_idx in np.unique(subject_indices):
        subject_mask = subject_indices == subject_idx
        subject_logits = logits[subject_mask]
        subject_targets = targets[subject_mask]
        subject_epoch_indices = epoch_indices[subject_mask]
        order = np.argsort(subject_epoch_indices)

        subject_log_probs = numpy_log_softmax(subject_logits[order])
        subject_predictions = viterbi_decode(
            subject_log_probs,
            start_log_probs,
            weighted_transition_log_probs,
        )
        ordered_targets.append(subject_targets[order])
        ordered_predictions.append(subject_predictions)

    y_true = np.concatenate(ordered_targets)
    y_pred = np.concatenate(ordered_predictions)
    return y_true, y_pred, compute_classification_metrics(y_true, y_pred)


def train_one_epoch(model, loader, criterion, optimizer, device, max_batches=None):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch_idx, batch in enumerate(loader, start=1):
        batch_X, batch_y, _, _ = batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)
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
def evaluate_model(
    model,
    loader,
    criterion,
    device,
    hmm_parameters=None,
    transition_weight=1.0,
    max_batches=None,
):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_targets = []
    all_predictions = []
    all_logits = []
    all_subject_indices = []
    all_epoch_indices = []

    for batch_idx, batch in enumerate(loader, start=1):
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

        if max_batches is not None and batch_idx >= max_batches:
            break

    y_true_raw = np.concatenate(all_targets)
    y_pred_raw = np.concatenate(all_predictions)
    logits_raw = np.concatenate(all_logits)
    subject_indices = np.concatenate(all_subject_indices)
    epoch_indices = np.concatenate(all_epoch_indices)

    raw_metrics = compute_classification_metrics(y_true_raw, y_pred_raw)
    y_true_final, y_pred_final, hmm_metrics = decode_with_hmm(
        logits_raw,
        y_true_raw,
        subject_indices,
        epoch_indices,
        hmm_parameters,
        transition_weight=transition_weight,
    )
    selection_f1 = hmm_metrics["macro_f1"] if hmm_metrics is not None else raw_metrics["macro_f1"]

    return {
        "loss": total_loss / max(total_samples, 1),
        "selection_f1": float(selection_f1),
        "raw_metrics": raw_metrics,
        "hmm_metrics": hmm_metrics,
        "raw_y_true": y_true_raw,
        "raw_y_pred": y_pred_raw,
        "y_true": y_true_final,
        "y_pred": y_pred_final,
    }


def run_cross_validation_fold(args, fold_config, model_factory, device):
    fold_idx = fold_config["fold"]
    train_subjects = fold_config["train_subjects"]
    val_subjects = fold_config["val_subjects"]
    seed_everything(args.seed + fold_idx)

    print(f"\n[CV Fold {fold_idx}] Loading training data...")
    train_sequences, sfreq = load_raw_subject_sequences(train_subjects)
    print(f"[CV Fold {fold_idx}] Loading validation data...")
    val_sequences, _ = load_raw_subject_sequences(val_subjects)
    (train_sequences, val_sequences), _, _ = normalize_sequence_splits(train_sequences, val_sequences)

    train_labels = flatten_subject_labels(train_sequences)
    train_dataset = SleepContextDataset(train_sequences, context_epochs=args.context_epochs)
    val_dataset = SleepContextDataset(val_sequences, context_epochs=args.context_epochs)
    train_loader = build_loader(
        train_dataset,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = build_loader(
        val_dataset,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = model_factory(
        in_channels=train_sequences[0]["X"].shape[1],
        num_classes=len(TARGET_NAMES),
    ).to(device)
    criterion, optimizer, scheduler = create_training_components(model, train_labels, args, device)
    hmm_parameters = estimate_hmm_parameters(
        train_sequences,
        smoothing=args.transition_smoothing,
    )

    best_state = None
    best_selection_f1 = -1.0
    best_epoch = 0
    best_val_loss = float("inf")
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
        val_metrics = evaluate_model(
            model,
            val_loader,
            criterion,
            device,
            hmm_parameters=hmm_parameters,
            transition_weight=args.transition_weight,
            max_batches=args.max_eval_batches,
        )
        scheduler.step(val_metrics["selection_f1"])

        raw_f1 = val_metrics["raw_metrics"]["macro_f1"]
        hmm_f1 = (
            val_metrics["hmm_metrics"]["macro_f1"]
            if val_metrics["hmm_metrics"] is not None
            else None
        )
        hmm_message = f" | val_hmm_macro_f1={hmm_f1:.4f}" if hmm_f1 is not None else ""
        print(
            f"[CV Fold {fold_idx}] Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_metrics['loss']:.4f} | "
            f"val_raw_macro_f1={raw_f1:.4f}{hmm_message}"
        )

        if val_metrics["selection_f1"] > best_selection_f1 + 1e-4:
            best_selection_f1 = val_metrics["selection_f1"]
            best_epoch = epoch
            best_val_loss = val_metrics["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"[CV Fold {fold_idx}] Early stopping triggered after epoch {epoch}.")
                break

    if best_state is None:
        raise RuntimeError(f"Fold {fold_idx} did not produce a checkpoint.")

    model.load_state_dict(best_state)
    val_metrics = evaluate_model(
        model,
        val_loader,
        criterion,
        device,
        hmm_parameters=hmm_parameters,
        transition_weight=args.transition_weight,
        max_batches=args.max_eval_batches,
    )
    return {
        "fold": fold_idx,
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "selection_macro_f1": float(val_metrics["selection_f1"]),
        "raw_metrics": metric_snapshot(val_metrics["raw_metrics"]),
        "hmm_metrics": metric_snapshot(val_metrics["hmm_metrics"]),
        "train_subjects": subject_ids(train_subjects),
        "val_subjects": subject_ids(val_subjects),
        "sampling_rate_hz": float(sfreq),
    }


def run_deep_training(args, model_factory, model_name):
    seed_everything(args.seed)
    if args.val_subjects is not None:
        print(
            "--val-subjects is deprecated and ignored because validation subjects are now "
            "determined by subject-level cross-validation."
        )

    subjects = select_subjects(args.data_dir, args.max_subjects)
    development_subjects, test_subjects = split_subjects(
        subjects,
        test_subjects=args.test_subjects,
        seed=args.seed,
    )
    cv_folds = build_cv_folds(
        development_subjects,
        cv_folds=args.cv_folds,
        seed=args.seed,
    )

    print(f"Total subjects found: {len(subjects)}")
    print(f"Development subjects: {subject_ids(development_subjects)}")
    print(f"Test subjects: {subject_ids(test_subjects)}")
    print(f"Cross-validation folds: {len(cv_folds)}")
    print(f"Context window size (epochs): {args.context_epochs}")

    device = resolve_device(args.device)
    print(f"\n[Stage 1] Training on device: {device}")

    fold_metrics = []
    for fold in cv_folds:
        fold_metrics.append(
            run_cross_validation_fold(
                args=args,
                fold_config=fold,
                model_factory=model_factory,
                device=device,
            )
        )

    selection_scores = [fold["selection_macro_f1"] for fold in fold_metrics]
    raw_scores = [fold["raw_metrics"]["macro_f1"] for fold in fold_metrics]
    hmm_scores = [
        fold["hmm_metrics"]["macro_f1"]
        for fold in fold_metrics
        if fold["hmm_metrics"] is not None
    ]
    hmm_mean_message = f"{np.mean(hmm_scores):.4f}" if hmm_scores else "n/a"
    selected_epochs = max(
        1,
        int(round(float(np.median([fold["best_epoch"] for fold in fold_metrics])))),
    )
    print(
        f"\n[Stage 2] Cross-validation complete | "
        f"selection_macro_f1={np.mean(selection_scores):.4f} +/- {np.std(selection_scores):.4f} | "
        f"raw_macro_f1={np.mean(raw_scores):.4f} | "
        f"hmm_macro_f1={hmm_mean_message} | "
        f"selected_epochs={selected_epochs}"
    )

    print("\n[Stage 3] Loading development data...")
    development_sequences, sfreq = load_raw_subject_sequences(development_subjects)
    print("\n[Stage 4] Loading test data...")
    test_sequences, _ = load_raw_subject_sequences(test_subjects)
    (development_sequences, test_sequences), norm_mean, norm_std = normalize_sequence_splits(
        development_sequences,
        test_sequences,
    )

    development_labels = flatten_subject_labels(development_sequences)
    print(
        f"\nDevelopment epochs: {len(development_labels)}, "
        f"Test epochs: {len(flatten_subject_labels(test_sequences))}"
    )
    print(f"Class distribution (development): {np.unique(development_labels, return_counts=True)}")

    development_dataset = SleepContextDataset(
        development_sequences,
        context_epochs=args.context_epochs,
    )
    test_dataset = SleepContextDataset(
        test_sequences,
        context_epochs=args.context_epochs,
    )
    train_loader = build_loader(
        development_dataset,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    test_loader = build_loader(
        test_dataset,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = model_factory(
        in_channels=development_sequences[0]["X"].shape[1],
        num_classes=len(TARGET_NAMES),
    ).to(device)
    criterion, optimizer, _ = create_training_components(model, development_labels, args, device)
    hmm_parameters = estimate_hmm_parameters(
        development_sequences,
        smoothing=args.transition_smoothing,
    )

    print(f"\n[Stage 5] Training final model for {selected_epochs} epochs on all development subjects...")
    for epoch in range(1, selected_epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            max_batches=args.max_train_batches,
        )
        print(f"Final training epoch {epoch:02d}/{selected_epochs} | train_loss={train_loss:.4f}")

    print("\n[Stage 6] Evaluating final model on the held-out test set...")
    test_metrics = evaluate_model(
        model,
        test_loader,
        criterion,
        device,
        hmm_parameters=hmm_parameters,
        transition_weight=args.transition_weight,
        max_batches=args.max_eval_batches,
    )

    cv_summary = {
        "cv_folds": len(cv_folds),
        "mean_selection_macro_f1": float(np.mean(selection_scores)),
        "std_selection_macro_f1": float(np.std(selection_scores)),
        "mean_raw_macro_f1": float(np.mean(raw_scores)),
        "mean_hmm_macro_f1": float(np.mean(hmm_scores)) if hmm_scores else None,
        "selected_epochs": selected_epochs,
        "context_epochs": args.context_epochs,
        "transition_smoothing": float(args.transition_smoothing),
        "transition_weight": float(args.transition_weight),
        "development_subjects": subject_ids(development_subjects),
        "test_subjects": subject_ids(test_subjects),
        "folds": fold_metrics,
    }

    raw_test_f1 = test_metrics["raw_metrics"]["macro_f1"]
    hmm_test_f1 = (
        test_metrics["hmm_metrics"]["macro_f1"]
        if test_metrics["hmm_metrics"] is not None
        else None
    )

    common_lines = [
        f"Cross-validation selection macro F1: {cv_summary['mean_selection_macro_f1']:.4f}",
        f"Cross-validation raw macro F1: {cv_summary['mean_raw_macro_f1']:.4f}",
        (
            f"Cross-validation HMM macro F1: {cv_summary['mean_hmm_macro_f1']:.4f}"
            if cv_summary["mean_hmm_macro_f1"] is not None
            else "Cross-validation HMM macro F1: n/a"
        ),
        f"Selected final epochs from CV: {selected_epochs}",
        f"Context epochs: {args.context_epochs}",
        f"Transition smoothing: {args.transition_smoothing:.4f}",
        f"Transition weight: {args.transition_weight:.4f}",
        f"Test loss: {test_metrics['loss']:.4f}",
        f"Sampling rate (Hz): {sfreq}",
        f"Development subjects: {subject_ids(development_subjects)}",
        f"Test subjects: {subject_ids(test_subjects)}",
    ]
    if args.max_train_batches is not None:
        common_lines.append(f"Smoke train batch limit: {args.max_train_batches}")
    if args.max_eval_batches is not None:
        common_lines.append(f"Smoke eval batch limit: {args.max_eval_batches}")

    raw_extra_lines = common_lines + [
        f"Test raw macro F1: {raw_test_f1:.4f}",
        "Prediction mode: raw per-epoch argmax (no HMM decoding).",
    ]

    hmm_extra_lines = common_lines + [
        f"Test HMM macro F1: {hmm_test_f1:.4f}" if hmm_test_f1 is not None else "Test HMM macro F1: n/a",
        "Prediction mode: HMM/Viterbi decoding.",
        "Final predictions use HMM/Viterbi decoding estimated from development subjects.",
    ]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_output_dir = output_dir / "raw"
    hmm_output_dir = output_dir / "hmm"

    raw_metrics_saved = save_results(
        test_metrics["raw_y_true"],
        test_metrics["raw_y_pred"],
        raw_output_dir,
        f"{model_name} (Raw)",
        extra_lines=raw_extra_lines,
        extra_metrics={
            "prediction_mode": "raw",
            "transition_weight": float(args.transition_weight),
            "transition_smoothing": float(args.transition_smoothing),
            "cross_validation": cv_summary,
            "test_loss": float(test_metrics["loss"]),
            "raw_test_metrics": metric_snapshot(test_metrics["raw_metrics"]),
            "hmm_test_metrics": metric_snapshot(test_metrics["hmm_metrics"]),
        },
    )

    hmm_metrics_saved = save_results(
        test_metrics["y_true"],
        test_metrics["y_pred"],
        hmm_output_dir,
        f"{model_name} (HMM)",
        extra_lines=hmm_extra_lines,
        extra_metrics={
            "prediction_mode": "hmm",
            "transition_weight": float(args.transition_weight),
            "transition_smoothing": float(args.transition_smoothing),
            "cross_validation": cv_summary,
            "test_loss": float(test_metrics["loss"]),
            "raw_test_metrics": metric_snapshot(test_metrics["raw_metrics"]),
            "hmm_test_metrics": metric_snapshot(test_metrics["hmm_metrics"]),
        },
    )

    comparison = {
        "model": model_name,
        "transition_smoothing": float(args.transition_smoothing),
        "transition_weight": float(args.transition_weight),
        "test_loss": float(test_metrics["loss"]),
        "raw_test_metrics": metric_snapshot(test_metrics["raw_metrics"]),
        "hmm_test_metrics": metric_snapshot(test_metrics["hmm_metrics"]),
        "cv_summary": cv_summary,
    }

    (output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2),
        encoding="utf-8",
    )
    (output_dir / "cv_summary.json").write_text(json.dumps(cv_summary, indent=2), encoding="utf-8")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "normalization_mean": norm_mean,
            "normalization_std": norm_std,
            "sampling_rate_hz": sfreq,
            "target_names": TARGET_NAMES,
            "model_name": model_name,
            "context_epochs": args.context_epochs,
            "hmm_decoder": {
                "start_log_probs": hmm_parameters[0],
                "transition_log_probs": hmm_parameters[1],
                "transition_smoothing": float(args.transition_smoothing),
                "transition_weight": float(args.transition_weight),
            },
            "subjects": {
                "development": subject_ids(development_subjects),
                "test": subject_ids(test_subjects),
            },
            "cross_validation": cv_summary,
            "args": vars(args),
        },
        output_dir / "model.pt",
    )

    print(f"Saved CV summary to: {output_dir / 'cv_summary.json'}")
    print(f"Saved comparison summary to: {output_dir / 'comparison.json'}")
    print(f"Saved model checkpoint to: {output_dir / 'model.pt'}")
    print(f"Saved raw results to: {raw_output_dir}")
    print(f"Saved HMM results to: {hmm_output_dir}")
    print(f"Raw Macro F1-score: {raw_metrics_saved['macro_f1']:.4f}")
    print(f"HMM Macro F1-score: {hmm_metrics_saved['macro_f1']:.4f}")

    return {
        "raw": raw_metrics_saved,
        "hmm": hmm_metrics_saved,
    }