"""
Model training, baseline comparison, sensitivity analysis, and hyperparameter tuning module.
All modeling and tuning is strictly performed on the training set using 5-fold StratifiedKFold.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(__file__))

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, precision_score, recall_score,
    accuracy_score, make_scorer, confusion_matrix
)

from data_preprocessing import load_and_clean_data, get_feature_splits, build_preprocessor, RANDOM_STATE

METRICS_DIR = "outputs/metrics"
os.makedirs(METRICS_DIR, exist_ok=True)

def specificity_scorer(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0

def run_sensitivity_analysis():
    """
    Compares 5-fold CV performance using Domain-based zero imputation vs Median imputation for TotalCharges.
    """
    print("\n================ SENSITIVITY ANALYSIS (TotalCharges Imputation) ================")
    results = {}
    
    for imp_method in ["domain_zero", "median"]:
        df = load_and_clean_data(total_charges_imputation=imp_method)
        X_train, _, y_train, _, num_features, cat_features = get_feature_splits(df)
        
        preprocessor = build_preprocessor(num_features, cat_features)
        clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_validate(
            pipeline, X_train, y_train, cv=cv,
            scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision", "f1": "f1"}
        )
        
        results[imp_method] = {
            "roc_auc_mean": float(np.mean(cv_scores["test_roc_auc"])),
            "roc_auc_std": float(np.std(cv_scores["test_roc_auc"])),
            "pr_auc_mean": float(np.mean(cv_scores["test_pr_auc"])),
            "pr_auc_std": float(np.std(cv_scores["test_pr_auc"])),
            "f1_mean": float(np.mean(cv_scores["test_f1"])),
            "f1_std": float(np.std(cv_scores["test_f1"]))
        }
        print(f"Imputation: {imp_method:<12} | ROC-AUC: {results[imp_method]['roc_auc_mean']:.4f} "
              f"| PR-AUC: {results[imp_method]['pr_auc_mean']:.4f} | F1: {results[imp_method]['f1_mean']:.4f}")
        
    with open(os.path.join(METRICS_DIR, "sensitivity_analysis.json"), "w") as f:
        json.dump(results, f, indent=2)
        
    return results

def train_and_compare_models(X_train, y_train, num_features, cat_features):
    """
    Trains and compares multiple candidate models via 5-fold Stratified CV on training set.
    """
    print("\n================ MODEL COMPARISON (5-Fold Stratified CV on Train) ================")
    preprocessor = build_preprocessor(num_features, cat_features)
    
    models = {
        "Dummy (Most Frequent)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression (Default)": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Logistic Regression (Balanced)": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        "Decision Tree (Max Depth 5)": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
        "Decision Tree (Balanced)": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest (Default)": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "Random Forest (Balanced)": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE)
    }
    
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "specificity": make_scorer(specificity_scorer)
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    comparison_results = {}
    
    for name, clf in models.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])
        
        cv_out = cross_validate(
            pipe, X_train, y_train, cv=cv, scoring=scoring,
            return_train_score=True, n_jobs=-1
        )
        
        summary = {
            "val_roc_auc_mean": float(np.mean(cv_out["test_roc_auc"])),
            "val_roc_auc_std": float(np.std(cv_out["test_roc_auc"])),
            "val_pr_auc_mean": float(np.mean(cv_out["test_pr_auc"])),
            "val_pr_auc_std": float(np.std(cv_out["test_pr_auc"])),
            "val_f1_mean": float(np.mean(cv_out["test_f1"])),
            "val_f1_std": float(np.std(cv_out["test_f1"])),
            "val_recall_mean": float(np.mean(cv_out["test_recall"])),
            "val_precision_mean": float(np.mean(cv_out["test_precision"])),
            "val_accuracy_mean": float(np.mean(cv_out["test_accuracy"])),
            "val_specificity_mean": float(np.mean(cv_out["test_specificity"])),
            "train_roc_auc_mean": float(np.mean(cv_out["train_roc_auc"])),
            "train_f1_mean": float(np.mean(cv_out["train_f1"])),
            "train_accuracy_mean": float(np.mean(cv_out["train_accuracy"]))
        }
        comparison_results[name] = summary
        
        print(f"{name:<30} | ROC-AUC: {summary['val_roc_auc_mean']:.4f} | PR-AUC: {summary['val_pr_auc_mean']:.4f} "
              f"| F1: {summary['val_f1_mean']:.4f} | Rec: {summary['val_recall_mean']:.4f} | Prec: {summary['val_precision_mean']:.4f}")
        
    with open(os.path.join(METRICS_DIR, "cv_comparison_results.json"), "w") as f:
        json.dump(comparison_results, f, indent=2)
        
    return comparison_results

def hyperparameter_tuning(X_train, y_train, num_features, cat_features):
    """
    Performs systematic GridSearchCV tuning on promising models using Stratified 5-Fold CV.
    Target objective: PR-AUC (average_precision).
    """
    print("\n================ HYPERPARAMETER TUNING (GridSearchCV) ================")
    preprocessor = build_preprocessor(num_features, cat_features)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    # 1. Tuning Gradient Boosting
    gb_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", GradientBoostingClassifier(random_state=RANDOM_STATE))
    ])
    gb_param_grid = {
        "classifier__n_estimators": [100, 150, 200],
        "classifier__learning_rate": [0.03, 0.05, 0.1],
        "classifier__max_depth": [3, 4],
        "classifier__subsample": [0.8, 1.0]
    }
    print("Tuning Gradient Boosting Classifier...")
    gb_grid = GridSearchCV(
        gb_pipe, gb_param_grid, cv=cv, scoring="average_precision",
        n_jobs=-1, verbose=1, return_train_score=True
    )
    gb_grid.fit(X_train, y_train)
    print(f"GB Best Params: {gb_grid.best_params_}")
    print(f"GB Best CV PR-AUC: {gb_grid.best_score_:.4f}")
    
    # 2. Tuning Logistic Regression
    lr_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
    ])
    lr_param_grid = {
        "classifier__C": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
        "classifier__class_weight": [None, "balanced"],
        "classifier__solver": ["lbfgs", "liblinear"]
    }
    print("Tuning Logistic Regression...")
    lr_grid = GridSearchCV(
        lr_pipe, lr_param_grid, cv=cv, scoring="average_precision",
        n_jobs=-1, verbose=1, return_train_score=True
    )
    lr_grid.fit(X_train, y_train)
    print(f"LR Best Params: {lr_grid.best_params_}")
    print(f"LR Best CV PR-AUC: {lr_grid.best_score_:.4f}")
    
    # 3. Tuning Random Forest
    rf_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=RANDOM_STATE))
    ])
    rf_param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [8, 12, 16],
        "classifier__min_samples_split": [5, 10],
        "classifier__class_weight": [None, "balanced", "balanced_subsample"]
    }
    print("Tuning Random Forest Classifier...")
    rf_grid = GridSearchCV(
        rf_pipe, rf_param_grid, cv=cv, scoring="average_precision",
        n_jobs=-1, verbose=1, return_train_score=True
    )
    rf_grid.fit(X_train, y_train)
    print(f"RF Best Params: {rf_grid.best_params_}")
    print(f"RF Best CV PR-AUC: {rf_grid.best_score_:.4f}")
    
    tuning_summary = {
        "GradientBoosting": {
            "best_params": gb_grid.best_params_,
            "best_cv_pr_auc": float(gb_grid.best_score_)
        },
        "LogisticRegression": {
            "best_params": lr_grid.best_params_,
            "best_cv_pr_auc": float(lr_grid.best_score_)
        },
        "RandomForest": {
            "best_params": rf_grid.best_params_,
            "best_cv_pr_auc": float(rf_grid.best_score_)
        }
    }
    
    with open(os.path.join(METRICS_DIR, "tuning_results.json"), "w") as f:
        json.dump(tuning_summary, f, indent=2)
        
    return gb_grid, lr_grid, rf_grid

if __name__ == "__main__":
    run_sensitivity_analysis()
    
    df = load_and_clean_data(total_charges_imputation="domain_zero")
    X_train, X_test, y_train, y_test, num_features, cat_features = get_feature_splits(df)
    
    comparison_results = train_and_compare_models(X_train, y_train, num_features, cat_features)
    gb_grid, lr_grid, rf_grid = hyperparameter_tuning(X_train, y_train, num_features, cat_features)
    print("\nTraining and tuning completed successfully.")
