"""
Final evaluation, threshold tuning, interpretation, and visualization module.
Threshold optimization is performed strictly on out-of-fold training data.
Final evaluation is performed once on the untouched test set.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(__file__))

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, precision_score, recall_score,
    accuracy_score, confusion_matrix, roc_curve, precision_recall_curve, brier_score_loss
)
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance

from data_preprocessing import load_and_clean_data, get_feature_splits, build_preprocessor, RANDOM_STATE

OUTPUT_FIG_DIR = "outputs/figures"
METRICS_DIR = "outputs/metrics"
MODELS_DIR = "outputs/models"
os.makedirs(OUTPUT_FIG_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "font.sans-serif": "Arial",
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.titlesize": 16,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

def calculate_metrics_at_threshold(y_true, y_prob, threshold=0.50):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return {
        "threshold": float(threshold),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "specificity": float(spec),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "confusion_matrix": cm.tolist()
    }

def find_optimal_threshold_oof(pipeline, X_train, y_train):
    """
    Computes out-of-fold predicted probabilities on X_train using 5-fold Stratified CV.
    Determines threshold that maximizes F1-score on training folds.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_probs = cross_val_predict(pipeline, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    
    thresholds = np.linspace(0.1, 0.9, 81)
    f1_scores = []
    
    for t in thresholds:
        preds = (oof_probs >= t).astype(int)
        f1_scores.append(f1_score(y_train, preds, zero_division=0))
        
    best_idx = np.argmax(f1_scores)
    best_threshold = float(thresholds[best_idx])
    best_f1 = float(f1_scores[best_idx])
    
    print(f"\n[Threshold Tuning on OOF Train] Best F1-optimal threshold: {best_threshold:.2f} (OOF F1: {best_f1:.4f})")
    
    return best_threshold, oof_probs

def plot_model_comparison(comparison_file="outputs/metrics/cv_comparison_results.json"):
    if not os.path.exists(comparison_file):
        print(f"Warning: {comparison_file} not found for model comparison plot.")
        return
        
    with open(comparison_file, "r") as f:
        data = json.load(f)
        
    models = list(data.keys())
    # Exclude dummy for clearer visual scale or include it
    metrics = ["val_roc_auc_mean", "val_pr_auc_mean", "val_f1_mean", "val_accuracy_mean"]
    labels = ["ROC-AUC", "PR-AUC", "F1-Score", "Accuracy"]
    
    df_plot = []
    for m in models:
        # short name
        short_name = m.replace(" (Default)", "").replace(" (Balanced)", " [Bal]")
        for met, lab in zip(metrics, labels):
            df_plot.append({"Model": short_name, "Metric": lab, "Score": data[m][met]})
            
    df_plot = pd.DataFrame(df_plot)
    
    fig, ax = plt.subplots(figsize=(13, 6))
    palette = ["#2b5c8f", "#d9534f", "#e8a838", "#4da375"]
    sns.barplot(data=df_plot, x="Model", y="Score", hue="Metric", palette=palette, ax=ax, edgecolor="black", linewidth=0.8)
    
    ax.set_title("5-Fold Cross-Validation Model Comparison across Key Metrics", pad=15, fontweight="bold")
    ax.set_ylabel("Score (0.0 - 1.0)")
    ax.set_ylim(0, 1.0)
    plt.xticks(rotation=25, ha="right", fontsize=10)
    ax.legend(title="Metric", frameon=True, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FIG_DIR, "13_model_comparison.png"))
    plt.close(fig)
    print("Saved 13_model_comparison.png")

def evaluate_and_interpret():
    # 1. Load data and splits
    df = load_and_clean_data(total_charges_imputation="domain_zero")
    X_train, X_test, y_train, y_test, num_features, cat_features = get_feature_splits(df)
    
    # 2. Check tuning results to build best model pipeline
    tuning_file = os.path.join(METRICS_DIR, "tuning_results.json")
    if os.path.exists(tuning_file):
        with open(tuning_file, "r") as f:
            tuning_data = json.load(f)
        gb_params = {k.replace("classifier__", ""): v for k, v in tuning_data["GradientBoosting"]["best_params"].items()}
        clf = GradientBoostingClassifier(random_state=RANDOM_STATE, **gb_params)
        print(f"Instantiating tuned Gradient Boosting with params: {gb_params}")
    else:
        print("Tuning file not found; defaulting to tuned Gradient Boosting parameters.")
        clf = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=RANDOM_STATE)
        
    preprocessor = build_preprocessor(num_features, cat_features)
    best_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])
    
    # 3. Determine threshold on training set via out-of-fold cross-validation
    best_thresh, oof_probs = find_optimal_threshold_oof(best_pipe, X_train, y_train)
    
    # 4. Fit the pipeline on the full training set (leakage-free!)
    best_pipe.fit(X_train, y_train)
    
    # Save fitted model pipeline
    model_path = os.path.join(MODELS_DIR, "best_model_pipeline.joblib")
    joblib.dump(best_pipe, model_path)
    print(f"Saved fitted best model pipeline to {model_path}")
    
    # 5. Evaluate on untouched Test Set
    test_probs = best_pipe.predict_proba(X_test)[:, 1]
    test_roc_auc = roc_auc_score(y_test, test_probs)
    test_pr_auc = average_precision_score(y_test, test_probs)
    brier = brier_score_loss(y_test, test_probs)
    
    # Evaluation at default threshold 0.50
    metrics_050 = calculate_metrics_at_threshold(y_test, test_probs, threshold=0.50)
    metrics_050["roc_auc"] = float(test_roc_auc)
    metrics_050["pr_auc"] = float(test_pr_auc)
    metrics_050["brier_score"] = float(brier)
    
    # Evaluation at tuned threshold
    metrics_tuned = calculate_metrics_at_threshold(y_test, test_probs, threshold=best_thresh)
    metrics_tuned["roc_auc"] = float(test_roc_auc)
    metrics_tuned["pr_auc"] = float(test_pr_auc)
    metrics_tuned["brier_score"] = float(brier)
    
    print("\n================ FINAL TEST SET EVALUATION ================")
    print(f"ROC-AUC: {test_roc_auc:.4f} | PR-AUC: {test_pr_auc:.4f} | Brier Score: {brier:.4f}")
    print("\n--- At Default Threshold (0.50) ---")
    print(f"Accuracy: {metrics_050['accuracy']:.4f} | Precision: {metrics_050['precision']:.4f} | "
          f"Recall: {metrics_050['recall']:.4f} | F1: {metrics_050['f1']:.4f} | Specificity: {metrics_050['specificity']:.4f}")
    print(f"TP: {metrics_050['tp']}, FP: {metrics_050['fp']}, FN: {metrics_050['fn']}, TN: {metrics_050['tn']}")
    
    print(f"\n--- At Tuned Threshold ({best_thresh:.2f}) ---")
    print(f"Accuracy: {metrics_tuned['accuracy']:.4f} | Precision: {metrics_tuned['precision']:.4f} | "
          f"Recall: {metrics_tuned['recall']:.4f} | F1: {metrics_tuned['f1']:.4f} | Specificity: {metrics_tuned['specificity']:.4f}")
    print(f"TP: {metrics_tuned['tp']}, FP: {metrics_tuned['fp']}, FN: {metrics_tuned['fn']}, TN: {metrics_tuned['tn']}")
    
    # Save all test metrics
    final_metrics = {
        "model_name": "GradientBoostingClassifier (Tuned)",
        "default_threshold_0.50": metrics_050,
        "tuned_threshold": metrics_tuned
    }
    with open(os.path.join(METRICS_DIR, "test_metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=2)
        
    # -------------------------------------------------------------
    # 6. Evaluation Visualizations
    # -------------------------------------------------------------
    
    # Figure 10: Confusion Matrix (Default vs Tuned side by side)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    cm_050 = np.array(metrics_050["confusion_matrix"])
    cm_tuned = np.array(metrics_tuned["confusion_matrix"])
    
    for ax, cm, thresh_val, title in zip(
        [ax1, ax2], [cm_050, cm_tuned], [0.50, best_thresh],
        ["Default Threshold (0.50)", f"Tuned Threshold ({best_thresh:.2f})"]
    ):
        sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=ax,
                    annot_kws={"size": 14, "fontweight": "bold"}, linewidths=1.5)
        ax.set_title(f"Confusion Matrix: {title}", pad=12, fontweight="bold")
        ax.set_xlabel("Predicted Churn Status")
        ax.set_ylabel("Actual Churn Status")
        ax.set_xticklabels(["Retained (0)", "Churn (1)"])
        ax.set_yticklabels(["Retained (0)", "Churn (1)"])
        
        # Add percentage labels below counts
        total = cm.sum()
        for i in range(2):
            for j in range(2):
                pct = (cm[i, j] / total) * 100
                ax.text(j + 0.5, i + 0.75, f"({pct:.1f}%)", ha="center", va="center", color="gray", fontsize=10)
                
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FIG_DIR, "10_confusion_matrix.png"))
    plt.close(fig)
    print("Saved 10_confusion_matrix.png")
    
    # Figure 11: ROC Curve
    fig, ax = plt.subplots(figsize=(7, 6))
    fpr, tpr, _ = roc_curve(y_test, test_probs)
    ax.plot(fpr, tpr, color="#2b5c8f", lw=2.5, label=f"Gradient Boosting (AUC = {test_roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#999999", lw=1.5, linestyle="--", label="Random Classifier (AUC = 0.5000)")
    ax.set_xlim([-0.01, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve", pad=15, fontweight="bold")
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FIG_DIR, "11_roc_curve.png"))
    plt.close(fig)
    print("Saved 11_roc_curve.png")
    
    # Figure 12: Precision-Recall Curve
    fig, ax = plt.subplots(figsize=(7, 6))
    precs, recs, _ = precision_recall_curve(y_test, test_probs)
    baseline_rate = y_test.mean()
    ax.plot(recs, precs, color="#d9534f", lw=2.5, label=f"Gradient Boosting (PR-AUC = {test_pr_auc:.4f})")
    ax.axhline(baseline_rate, color="#999999", lw=1.5, linestyle="--", label=f"Baseline Churn Rate ({baseline_rate:.1%})")
    ax.set_xlim([0.0, 1.02])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("Recall (True Positive Rate)")
    ax.set_ylabel("Precision (Positive Predictive Value)")
    ax.set_title("Precision–Recall (PR) Curve", pad=15, fontweight="bold")
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FIG_DIR, "12_precision_recall_curve.png"))
    plt.close(fig)
    print("Saved 12_precision_recall_curve.png")
    
    # Figure 13: Model Comparison Chart
    plot_model_comparison()
    
    # Figure 14: Calibration Curve (Reliability Diagram)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    prob_true, prob_pred = calibration_curve(y_test, test_probs, n_bins=10, strategy="uniform")
    ax1.plot(prob_pred, prob_true, marker="o", color="#2b5c8f", lw=2, label=f"GB Model (Brier = {brier:.4f})")
    ax1.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    ax1.set_title("Calibration Curve (Reliability Diagram)", pad=12, fontweight="bold")
    ax1.set_xlabel("Mean Predicted Probability")
    ax1.set_ylabel("Fraction of Positives (Actual Churn)")
    ax1.legend(loc="upper left", frameon=True)
    
    # Predicted probability histogram
    sns.histplot(test_probs, bins=20, ax=ax2, color="#2b5c8f", kde=True)
    ax2.set_title("Predicted Churn Probability Distribution", pad=12, fontweight="bold")
    ax2.set_xlabel("Predicted Probability of Churn")
    ax2.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FIG_DIR, "14_calibration_curve.png"))
    plt.close(fig)
    print("Saved 14_calibration_curve.png")
    
    # -------------------------------------------------------------
    # 7. Model Interpretation: Permutation Importance
    # -------------------------------------------------------------
    print("\nComputing Permutation Feature Importance on Test Set...")
    perm_res = permutation_importance(
        best_pipe, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE, scoring="average_precision", n_jobs=-1
    )
    
    # Map back feature names from original columns
    sorted_importances_idx = perm_res.importances_mean.argsort()[::-1]
    top_n = 15
    top_indices = sorted_importances_idx[:top_n]
    top_features = [X_test.columns[i] for i in top_indices]
    top_means = perm_res.importances_mean[top_indices]
    top_stds = perm_res.importances_std[top_indices]
    
    fig, ax = plt.subplots(figsize=(10, 6.5))
    y_pos = np.arange(len(top_features))[::-1]
    ax.barh(y_pos, top_means, xerr=top_stds, align="center", color="#2b5c8f", edgecolor="black", alpha=0.85, ecolor="#d9534f", capsize=4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features, fontsize=11)
    ax.set_xlabel("Mean Decrease in PR-AUC (Permutation Importance)", fontsize=12)
    ax.set_title("Top 15 Permutation Feature Importances (Test Set, PR-AUC)", pad=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FIG_DIR, "15_feature_importance.png"))
    plt.close(fig)
    print("Saved 15_feature_importance.png")
    
    # Save feature importance values to JSON
    fi_dict = {feat: {"mean_importance": float(m), "std_importance": float(s)} 
               for feat, m, s in zip(top_features, top_means, top_stds)}
    with open(os.path.join(METRICS_DIR, "permutation_importance.json"), "w") as f:
        json.dump(fi_dict, f, indent=2)
        
    print("\nAll evaluation and interpretation figures generated successfully.")

if __name__ == "__main__":
    evaluate_and_interpret()
