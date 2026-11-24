"""
Exploratory Data Analysis (EDA) script for Telco Customer Churn.
Generates all mandatory EDA visualizations and saves them into outputs/figures/.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set overall aesthetic styling
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

OUTPUT_DIR = "outputs/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Custom color palette: Blue for No Churn (retained), Coral/Red for Churn
PALETTE = {"No": "#2b5c8f", "Yes": "#d9534f"}

def run_eda(filepath="dataset/Telco-Customer-Churn.csv"):
    df = pd.read_csv(filepath)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip(), errors="coerce").fillna(0.0)
    
    print("--- Generating EDA Figures ---")
    
    # -------------------------------------------------------------
    # 1. Target Class Distribution (01_churn_distribution.png)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    churn_counts = df["Churn"].value_counts()
    churn_pcts = df["Churn"].value_counts(normalize=True) * 100
    
    bars = ax.bar(["No Churn (Retained)", "Churn (Left)"], churn_counts.values, 
                  color=["#2b5c8f", "#d9534f"], width=0.55, edgecolor="black", linewidth=1.2)
    
    for bar, count, pct in zip(bars, churn_counts.values, churn_pcts.values):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 100, f"{count:,}\n({pct:.2f}%)", 
                ha="center", va="bottom", fontsize=11, fontweight="bold")
        
    ax.set_title("Customer Churn Target Distribution (N = 7,043)", pad=15, fontweight="bold")
    ax.set_ylabel("Customer Count")
    ax.set_ylim(0, 6000)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "01_churn_distribution.png"))
    plt.close(fig)
    print("Saved 01_churn_distribution.png")
    
    # -------------------------------------------------------------
    # 2. Tenure Distribution by Churn Status (02_tenure_distribution.png)
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram + KDE
    sns.histplot(data=df, x="tenure", hue="Churn", palette=PALETTE, kde=True, 
                 bins=30, ax=ax1, element="step", common_norm=False, alpha=0.4)
    ax1.set_title("Tenure Histogram & Density by Churn Status", pad=12, fontweight="bold")
    ax1.set_xlabel("Tenure (Months)")
    ax1.set_ylabel("Customer Count")
    
    # Boxplot
    sns.boxplot(data=df, x="Churn", y="tenure", hue="Churn", palette=PALETTE, ax=ax2, width=0.45, legend=False)
    ax2.set_title("Tenure Spread by Churn Status", pad=12, fontweight="bold")
    ax2.set_xlabel("Customer Churn")
    ax2.set_ylabel("Tenure (Months)")
    
    # Add median annotations
    medians = df.groupby("Churn")["tenure"].median()
    for i, (cat, med) in enumerate(medians.items()):
        color = "white" if med < 30 else "black"
        bg = "black" if med < 30 else "yellow"
        ax2.text(i, med + 1.5, f"Median: {med:.0f} mo", ha="center", va="bottom", 
                 fontweight="bold", color=color,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor=bg, alpha=0.7, edgecolor="none"))
    
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "02_tenure_distribution.png"))
    plt.close(fig)
    print("Saved 02_tenure_distribution.png")
    
    # -------------------------------------------------------------
    # 3. Churn Rate by Contract (03_churn_by_contract.png)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    contract_churn = df.groupby("Contract")["Churn"].apply(lambda s: (s == "Yes").mean() * 100).reindex(
        ["Month-to-month", "One year", "Two year"]
    )
    contract_counts = df.groupby("Contract")["Churn"].count().reindex(
        ["Month-to-month", "One year", "Two year"]
    )
    
    bars = ax.bar(contract_churn.index, contract_churn.values, 
                  color=["#d9534f", "#e8a838", "#2b5c8f"], width=0.55, edgecolor="black", linewidth=1.2)
    
    for bar, (cat, pct) in zip(bars, contract_churn.items()):
        total = contract_counts[cat]
        ax.text(bar.get_x() + bar.get_width()/2.0, pct + 1.0, 
                f"{pct:.1f}%\n(N={total:,})", ha="center", va="bottom", fontsize=11, fontweight="bold")
        
    ax.set_title("Customer Churn Rate by Contract Type", pad=15, fontweight="bold")
    ax.set_ylabel("Churn Rate (%)")
    ax.set_ylim(0, 55)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "03_churn_by_contract.png"))
    plt.close(fig)
    print("Saved 03_churn_by_contract.png")
    
    # -------------------------------------------------------------
    # 4. Churn Rate by Internet Service (04_churn_by_internet_service.png)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    is_churn = df.groupby("InternetService")["Churn"].apply(lambda s: (s == "Yes").mean() * 100)
    is_counts = df.groupby("InternetService")["Churn"].count()
    order = ["Fiber optic", "DSL", "No"]
    is_churn = is_churn.reindex(order)
    is_counts = is_counts.reindex(order)
    
    bars = ax.bar(is_churn.index, is_churn.values, 
                  color=["#d9534f", "#4da375", "#2b5c8f"], width=0.55, edgecolor="black", linewidth=1.2)
    
    for bar, (cat, pct) in zip(bars, is_churn.items()):
        total = is_counts[cat]
        ax.text(bar.get_x() + bar.get_width()/2.0, pct + 1.0, 
                f"{pct:.1f}%\n(N={total:,})", ha="center", va="bottom", fontsize=11, fontweight="bold")
        
    ax.set_title("Customer Churn Rate by Internet Service Type", pad=15, fontweight="bold")
    ax.set_ylabel("Churn Rate (%)")
    ax.set_ylim(0, 50)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "04_churn_by_internet_service.png"))
    plt.close(fig)
    print("Saved 04_churn_by_internet_service.png")
    
    # -------------------------------------------------------------
    # 5. Churn Rate by Payment Method (05_churn_by_payment_method.png)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5.5))
    pm_churn = df.groupby("PaymentMethod")["Churn"].apply(lambda s: (s == "Yes").mean() * 100).sort_values(ascending=False)
    pm_counts = df.groupby("PaymentMethod")["Churn"].count().loc[pm_churn.index]
    
    bars = ax.bar(pm_churn.index, pm_churn.values, 
                  color=["#d9534f", "#e8a838", "#2b5c8f", "#4da375"], width=0.6, edgecolor="black", linewidth=1.2)
    
    for bar, (cat, pct) in zip(bars, pm_churn.items()):
        total = pm_counts[cat]
        ax.text(bar.get_x() + bar.get_width()/2.0, pct + 1.0, 
                f"{pct:.1f}%\n(N={total:,})", ha="center", va="bottom", fontsize=10.5, fontweight="bold")
        
    ax.set_title("Customer Churn Rate by Payment Method", pad=15, fontweight="bold")
    ax.set_ylabel("Churn Rate (%)")
    ax.set_ylim(0, 55)
    plt.xticks(rotation=15, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "05_churn_by_payment_method.png"))
    plt.close(fig)
    print("Saved 05_churn_by_payment_method.png")
    
    # -------------------------------------------------------------
    # 6. Monthly Charges Distribution by Churn Status (06_monthly_charges_distribution.png)
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.histplot(data=df, x="MonthlyCharges", hue="Churn", palette=PALETTE, kde=True,
                 bins=30, ax=ax1, element="step", common_norm=False, alpha=0.4)
    ax1.set_title("Monthly Charges Distribution (KDE)", pad=12, fontweight="bold")
    ax1.set_xlabel("Monthly Charges ($)")
    ax1.set_ylabel("Customer Count")
    
    sns.boxplot(data=df, x="Churn", y="MonthlyCharges", hue="Churn", palette=PALETTE, ax=ax2, width=0.45, legend=False)
    ax2.set_title("Monthly Charges Comparison by Churn", pad=12, fontweight="bold")
    ax2.set_xlabel("Customer Churn")
    ax2.set_ylabel("Monthly Charges ($)")
    
    meds = df.groupby("Churn")["MonthlyCharges"].median()
    for i, (cat, med) in enumerate(meds.items()):
        ax2.text(i, med + 1.5, f"Median: ${med:.2f}", ha="center", va="bottom",
                 fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="gray"))
        
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "06_monthly_charges_distribution.png"))
    plt.close(fig)
    print("Saved 06_monthly_charges_distribution.png")
    
    # -------------------------------------------------------------
    # 7. Total Charges Distribution by Churn Status (07_total_charges_distribution.png)
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.histplot(data=df, x="TotalCharges", hue="Churn", palette=PALETTE, kde=True,
                 bins=30, ax=ax1, element="step", common_norm=False, alpha=0.4)
    ax1.set_title("Total Charges Distribution (KDE)", pad=12, fontweight="bold")
    ax1.set_xlabel("Total Charges ($)")
    ax1.set_ylabel("Customer Count")
    
    sns.boxplot(data=df, x="Churn", y="TotalCharges", hue="Churn", palette=PALETTE, ax=ax2, width=0.45, legend=False)
    ax2.set_title("Total Charges Comparison by Churn", pad=12, fontweight="bold")
    ax2.set_xlabel("Customer Churn")
    ax2.set_ylabel("Total Charges ($)")
    
    meds = df.groupby("Churn")["TotalCharges"].median()
    for i, (cat, med) in enumerate(meds.items()):
        ax2.text(i, med + 150, f"Median: ${med:.1f}", ha="center", va="bottom",
                 fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="gray"))
        
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "07_total_charges_distribution.png"))
    plt.close(fig)
    print("Saved 07_total_charges_distribution.png")
    
    # -------------------------------------------------------------
    # 8. Numerical Correlation Heatmap (08_numerical_correlation_heatmap.png)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    df_corr = df[num_cols + ["Churn"]].copy()
    df_corr["Churn_Numeric"] = (df_corr["Churn"] == "Yes").astype(int)
    corr_matrix = df_corr[["tenure", "MonthlyCharges", "TotalCharges", "Churn_Numeric"]].corr()
    
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    
    sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap=cmap, vmin=-1, vmax=1,
                square=True, linewidths=1.5, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Numerical Features & Target Correlation Matrix", pad=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "08_numerical_correlation_heatmap.png"))
    plt.close(fig)
    print("Saved 08_numerical_correlation_heatmap.png")
    
    # -------------------------------------------------------------
    # 9. Churn Rate by TechSupport, OnlineSecurity, and PaperlessBilling (09_churn_by_support_services.png)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    services = [
        ("TechSupport", "Tech Support"),
        ("OnlineSecurity", "Online Security"),
        ("PaperlessBilling", "Paperless Billing")
    ]
    
    for ax, (col, title) in zip(axes, services):
        rate = df.groupby(col)["Churn"].apply(lambda s: (s == "Yes").mean() * 100)
        cnts = df.groupby(col)["Churn"].count()
        
        bars = ax.bar(rate.index, rate.values, color=["#d9534f" if v > 30 else "#2b5c8f" for v in rate.values],
                      width=0.55, edgecolor="black", linewidth=1.1)
        for bar, (cat, pct) in zip(bars, rate.items()):
            tot = cnts[cat]
            ax.text(bar.get_x() + bar.get_width()/2.0, pct + 1.0, 
                    f"{pct:.1f}%\n(N={tot:,})", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_title(f"Churn Rate by {title}", pad=12, fontweight="bold")
        ax.set_ylabel("Churn Rate (%)")
        ax.set_ylim(0, 50)
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        if len(rate.index) > 2:
            ax.tick_params(axis='x', rotation=15)
            
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "09_churn_by_support_services.png"))
    plt.close(fig)
    print("Saved 09_churn_by_support_services.png")
    
    print("All EDA visualizations generated successfully.")

if __name__ == "__main__":
    run_eda()
