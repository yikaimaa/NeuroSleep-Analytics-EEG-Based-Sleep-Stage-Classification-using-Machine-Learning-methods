import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from xgboost import XGBClassifier
from data_loader import get_all_subjects, load_sleep_edf_subject
from features import extract_features_all
from pathlib import Path

def train_and_evaluate():
    data_dir = "data/sleep-edf"
    subjects = get_all_subjects(data_dir)
    print(f"Total subjects found: {len(subjects)}")
    
    # Subject-level split: 3 subjects for training, 1 for testing
    # We will use the last subject SC4031 as the test set
    train_subjects = subjects[:-1]
    test_subjects = subjects[-1:]
    
    X_train_list = []
    y_train_list = []
    
    print("\n[Stage 1] Preprocessing and feature extraction for training set...")
    for psg, hyp in train_subjects:
        print(f"  Processing {psg.name}...")
        X, y, sfreq = load_sleep_edf_subject(psg, hyp)
        df_feats = extract_features_all(X, sfreq)
        X_train_list.append(df_feats)
        y_train_list.append(y)
        
    X_train = pd.concat(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list)
    
    print("\n[Stage 2] Preprocessing and feature extraction for test set...")
    X_test_list = []
    y_test_list = []
    for psg, hyp in test_subjects:
        print(f"  Processing {psg.name}...")
        X, y, sfreq = load_sleep_edf_subject(psg, hyp)
        df_feats = extract_features_all(X, sfreq)
        X_test_list.append(df_feats)
        y_test_list.append(y)
        
    X_test = pd.concat(X_test_list, axis=0)
    y_test = np.concatenate(y_test_list)
    
    print(f"\nTraining set size: {X_train.shape}, Test set size: {X_test.shape}")
    print(f"Class distribution (train): {np.unique(y_train, return_counts=True)}")
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train XGBoost
    print("\n[Stage 3] Training XGBoost model...")
    # Use default parameters + some tuning for multi-class classification
    xgb = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        objective='multi:softprob',
        num_class=5,
        random_state=42,
        eval_metric='mlogloss'
    )
    
    xgb.fit(X_train_scaled, y_train)
    
    # Evaluate
    print("\n[Stage 4] Evaluating model...")
    y_pred = xgb.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"\nAccuracy: {acc:.4f}")
    print(f"Macro F1-score: {f1:.4f}")
    
    print("\nClassification Report:")
    target_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix (XGBoost)\nAccuracy: {acc:.4f}, Macro F1: {f1:.4f}')
    
    if not os.path.exists("results"):
        os.makedirs("results")
    plt.savefig("results/confusion_matrix.png")
    print("Saved confusion matrix TO: results/confusion_matrix.png")
    
    # Save results summary
    with open("results/summary.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Macro F1: {f1:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report(y_test, y_pred, target_names=target_names))

if __name__ == "__main__":
    train_and_evaluate()
