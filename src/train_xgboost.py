import inspect
import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from xgboost import XGBClassifier
from training_utils import (
    build_cv_folds,
    compute_classification_metrics,
    compute_sample_weights,
    load_feature_dataset,
    save_results,
    seed_everything,
    select_subjects,
    split_subjects,
    standardize_feature_splits,
    subject_ids,
)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data-dir", default="data/sleep-edf")
    parser.add_argument("--output-dir", default="results/xgboost")
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--test-subjects", type=int, default=2)
    parser.add_argument("--cv-folds", type=int, default=4)
    parser.add_argument("--n-estimators", type=int, default=800)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_xgb_classifier(args, n_estimators):
    return XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=5,
        random_state=args.seed,
        eval_metric="mlogloss",
    )


def fit_xgb_model(model, X_train, y_train, sample_weight, X_val=None, y_val=None, early_stopping_rounds=None):
    fit_kwargs = {
        "sample_weight": sample_weight,
    }
    if X_val is not None and y_val is not None:
        fit_kwargs["eval_set"] = [(X_val, y_val)]

    fit_signature = inspect.signature(model.fit)
    if "verbose" in fit_signature.parameters:
        fit_kwargs["verbose"] = False
    if early_stopping_rounds is not None and X_val is not None and y_val is not None:
        if "early_stopping_rounds" in fit_signature.parameters:
            fit_kwargs["early_stopping_rounds"] = early_stopping_rounds
        else:
            model.set_params(early_stopping_rounds=early_stopping_rounds)

    model.fit(X_train, y_train, **fit_kwargs)
    return model


def resolve_best_n_estimators(model):
    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is None:
        booster = model.get_booster()
        best_iteration = getattr(booster, "best_iteration", None)
    if best_iteration is None or best_iteration < 0:
        return int(model.get_params()["n_estimators"])
    return int(best_iteration) + 1


def train_and_evaluate(args):
    seed_everything(args.seed)
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

    fold_metrics = []
    best_n_estimators = []

    for fold in cv_folds:
        fold_idx = fold["fold"]
        train_subjects = fold["train_subjects"]
        val_subjects = fold["val_subjects"]
        print(f"\n[CV Fold {fold_idx}] Preprocessing and feature extraction for training set...")
        X_train, y_train = load_feature_dataset(train_subjects)
        print(f"[CV Fold {fold_idx}] Preprocessing and feature extraction for validation set...")
        X_val, y_val = load_feature_dataset(val_subjects)
        (X_train_scaled, X_val_scaled), _, _ = standardize_feature_splits(X_train, X_val)
        sample_weight = compute_sample_weights(y_train)

        model = make_xgb_classifier(args, n_estimators=args.n_estimators)
        fit_xgb_model(
            model,
            X_train_scaled,
            y_train,
            sample_weight=sample_weight,
            X_val=X_val_scaled,
            y_val=y_val,
            early_stopping_rounds=args.early_stopping_rounds,
        )

        y_val_pred = model.predict(X_val_scaled)
        metrics = compute_classification_metrics(y_val, y_val_pred)
        metrics["confusion_matrix"] = metrics["confusion_matrix"].tolist()
        selected_trees = resolve_best_n_estimators(model)
        metrics.update(
            {
                "fold": fold_idx,
                "selected_n_estimators": selected_trees,
                "train_subjects": subject_ids(train_subjects),
                "val_subjects": subject_ids(val_subjects),
            }
        )
        fold_metrics.append(metrics)
        best_n_estimators.append(selected_trees)
        print(
            f"[CV Fold {fold_idx}] val_macro_f1={metrics['macro_f1']:.4f} | "
            f"selected_n_estimators={selected_trees}"
        )

    cv_macro_f1_scores = [fold["macro_f1"] for fold in fold_metrics]
    cv_mean_f1 = float(np.mean(cv_macro_f1_scores))
    cv_std_f1 = float(np.std(cv_macro_f1_scores))
    selected_n_estimators = max(1, int(round(float(np.median(best_n_estimators)))))
    print(
        f"\n[Stage 1] Cross-validation complete | "
        f"mean_macro_f1={cv_mean_f1:.4f} | std_macro_f1={cv_std_f1:.4f} | "
        f"selected_n_estimators={selected_n_estimators}"
    )

    print("\n[Stage 2] Preprocessing and feature extraction for development set...")
    X_dev, y_dev = load_feature_dataset(development_subjects)
    print("\n[Stage 3] Preprocessing and feature extraction for test set...")
    X_test, y_test = load_feature_dataset(test_subjects)
    (X_dev_scaled, X_test_scaled), feature_mean, feature_std = standardize_feature_splits(X_dev, X_test)
    sample_weight = compute_sample_weights(y_dev)

    print(f"\nDevelopment set size: {X_dev.shape}, Test set size: {X_test.shape}")
    print(f"Class distribution (development): {np.unique(y_dev, return_counts=True)}")

    print("\n[Stage 4] Training final XGBoost model on all development subjects...")
    xgb = make_xgb_classifier(args, n_estimators=selected_n_estimators)
    xgb.fit(
        X_dev_scaled,
        y_dev,
        sample_weight=sample_weight,
    )

    print("\n[Stage 5] Evaluating model...")
    y_pred = xgb.predict(X_test_scaled)
    cv_summary = {
        "cv_folds": len(cv_folds),
        "mean_macro_f1": cv_mean_f1,
        "std_macro_f1": cv_std_f1,
        "selected_n_estimators": selected_n_estimators,
        "development_subjects": subject_ids(development_subjects),
        "test_subjects": subject_ids(test_subjects),
        "folds": fold_metrics,
    }

    metrics = save_results(
        y_test,
        y_pred,
        args.output_dir,
        "XGBoost",
        extra_lines=[
            f"Cross-validation mean macro F1: {cv_mean_f1:.4f}",
            f"Cross-validation std macro F1: {cv_std_f1:.4f}",
            f"Selected n_estimators from CV: {selected_n_estimators}",
            f"Development subjects: {subject_ids(development_subjects)}",
            f"Test subjects: {subject_ids(test_subjects)}",
        ],
        extra_metrics={"cross_validation": cv_summary},
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cv_summary.json").write_text(json.dumps(cv_summary, indent=2), encoding="utf-8")
    xgb.save_model(output_dir / "model.json")
    np.savez(output_dir / "feature_scaling.npz", mean=feature_mean, std=feature_std)
    print(f"Saved CV summary to: {output_dir / 'cv_summary.json'}")
    print(f"Saved model to: {output_dir / 'model.json'}")
    print(f"Saved feature scaling stats to: {output_dir / 'feature_scaling.npz'}")
    print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1-score: {metrics['macro_f1']:.4f}")
    print("\nClassification Report:")
    print(metrics["report"])


if __name__ == "__main__":
    train_and_evaluate(parse_args())
