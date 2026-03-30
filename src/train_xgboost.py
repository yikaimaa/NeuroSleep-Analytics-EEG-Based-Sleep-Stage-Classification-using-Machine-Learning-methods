from argparse import ArgumentParser

import numpy as np
from xgboost import XGBClassifier

from training_utils import (
    load_feature_dataset,
    save_results,
    seed_everything,
    select_subjects,
    split_subjects,
    standardize_features,
    subject_ids,
)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data-dir", default="data/sleep-edf")
    parser.add_argument("--output-dir", default="results/xgboost")
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--test-subjects", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def train_and_evaluate(args):
    seed_everything(args.seed)
    subjects = select_subjects(args.data_dir, args.max_subjects)
    train_subjects, _, test_subjects = split_subjects(
        subjects,
        test_subjects=args.test_subjects,
        val_subjects=0,
    )
    print(f"Total subjects found: {len(subjects)}")
    print(f"Train subjects: {subject_ids(train_subjects)}")
    print(f"Test subjects: {subject_ids(test_subjects)}")

    print("\n[Stage 1] Preprocessing and feature extraction for training set...")
    X_train, y_train = load_feature_dataset(train_subjects)

    print("\n[Stage 2] Preprocessing and feature extraction for test set...")
    X_test, y_test = load_feature_dataset(test_subjects)

    print(f"\nTraining set size: {X_train.shape}, Test set size: {X_test.shape}")
    print(f"Class distribution (train): {np.unique(y_train, return_counts=True)}")

    X_train_scaled, X_test_scaled = standardize_features(X_train, X_test)

    print("\n[Stage 3] Training XGBoost model...")
    xgb = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        objective="multi:softprob",
        num_class=5,
        random_state=args.seed,
        eval_metric="mlogloss",
    )

    xgb.fit(X_train_scaled, y_train)

    print("\n[Stage 4] Evaluating model...")
    y_pred = xgb.predict(X_test_scaled)

    metrics = save_results(
        y_test,
        y_pred,
        args.output_dir,
        "XGBoost",
        extra_lines=[
            f"Train subjects: {subject_ids(train_subjects)}",
            f"Test subjects: {subject_ids(test_subjects)}",
        ],
    )
    print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1-score: {metrics['macro_f1']:.4f}")
    print("\nClassification Report:")
    print(metrics["report"])
    return metrics


if __name__ == "__main__":
    train_and_evaluate(parse_args())
