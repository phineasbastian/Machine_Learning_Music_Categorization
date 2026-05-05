
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- Path Logic ---
# This ensures the script finds your CSVs regardless of which machine you run it on
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "processed", "csv")

def load_data():
    """Loads the feature CSVs generated in Milestone 2."""
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train_features.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "val_features.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test_features.csv"))
    return train_df, val_df, test_df

def preprocess_data(train, val, test):
    """Encodes labels and scales features."""
    # Separate Features (X) and Target (y)
    X_train = train.drop(columns=['genre'])
    y_train = train['genre']
    X_val = val.drop(columns=['genre'])
    y_val = val['genre']
    X_test = test.drop(columns=['genre'])
    y_test = test['genre']

    # 1. Label Encoding (Converts 'blues' to 0, 'metal' to 1, etc.)
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)

    # 2. Standard Scaling (Crucial for SVM accuracy)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled, y_train_enc, y_val_enc, y_test_enc, le.classes_, X_train.columns

def tune_svm(X_train, y_train):
    """
    Performs a Grid Search to find the best SVM parameters.
    This is the key to breaking the 75% accuracy barrier.
    """
    print("\n[STEP] Starting Hyperparameter Tuning for SVM (this may take a minute)...")
    
    param_grid = {
        'C': [0.1, 1, 10, 100],            # Penalty for errors
        'gamma': [1, 0.1, 0.01, 0.001],   # Kernel coefficient
        'kernel': ['rbf']                 # Radial Basis Function kernel
    }
    
    # Using 5-fold cross-validation
    grid = GridSearchCV(SVC(probability=True), param_grid, refit=True, verbose=0, cv=5)
    grid.fit(X_train, y_train)
    
    print(f"[RESULT] Best SVM Parameters: {grid.best_params_}")
    return grid.best_estimator_

def analyze_results(model, X_test, y_test, class_names, model_name):
    """Prints metrics and plots the confusion matrix."""
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    print(f"\n--- {model_name} Results ---")
    print(f"Accuracy: {acc:.4f}")
    
    # Plotting
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix: {model_name}\nAccuracy: {acc:.4f}')
    plt.ylabel('Actual Genre')
    plt.xlabel('Predicted Genre')
    plt.show()
    
    if model_name == "Optimized SVM":
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, preds, target_names=class_names))

def plot_feature_importance(rf_model, feature_names):
    """Visualizes which audio features are doing the heavy lifting."""
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(12, 6))
    plt.title("Top 10 Feature Importances (Random Forest)")
    plt.bar(range(10), importances[indices[:10]], color='darkblue', align="center")
    plt.xticks(range(10), [feature_names[i] for i in indices[:10]], rotation=45)
    plt.tight_layout()
    plt.show()

def main():
    # 1. Preparation
    train_df, val_df, test_df = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test, classes, feat_cols = preprocess_data(train_df, val_df, test_df)

    # 2. Baseline: Random Forest
    print("Training Random Forest Baseline...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    analyze_results(rf, X_test, y_test, classes, "Random Forest")
    plot_feature_importance(rf, feat_cols)

    # 3. Optimization: Tuned SVM
    best_svm = tune_svm(X_train, y_train)
    analyze_results(best_svm, X_test, y_test, classes, "Optimized SVM")

if __name__ == "__main__":
    main()