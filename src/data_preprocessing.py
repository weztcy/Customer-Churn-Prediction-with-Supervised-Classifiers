"""
Data preprocessing and pipeline creation module for Customer Churn Prediction.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42

def load_and_clean_data(filepath="dataset/Telco-Customer-Churn.csv", total_charges_imputation="domain_zero"):
    """
    Loads Telco Customer Churn dataset, trims whitespace, handles TotalCharges,
    encodes target, and returns cleaned DataFrame.
    
    Parameters:
    -----------
    filepath : str
        Path to Telco-Customer-Churn.csv.
    total_charges_imputation : str
        'domain_zero': Sets blank TotalCharges (all have tenure=0) to 0.0.
        'median': Replaces blank TotalCharges with median.
    
    Returns:
    --------
    df : pd.DataFrame
        Cleaned dataset.
    """
    df = pd.read_csv(filepath)
    
    # Strip whitespace across string columns
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype(str).str.strip()
    
    # Coerce TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    
    if total_charges_imputation == "domain_zero":
        # Domain rule: accounts with tenure == 0 have not completed any billing cycle, hence TotalCharges = 0.0
        df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    elif total_charges_imputation == "median":
        median_val = df["TotalCharges"].median()
        df["TotalCharges"] = df["TotalCharges"].fillna(median_val)
    
    # Target variable encoding: Yes = 1, No = 0
    df["Churn_Numeric"] = (df["Churn"] == "Yes").astype(int)
    
    return df

def get_feature_splits(df, test_size=0.2, random_state=RANDOM_STATE):
    """
    Separates predictors and target, drops customerID, and performs stratified split.
    
    Returns:
    --------
    X_train, X_test, y_train, y_test, num_features, cat_features
    """
    # Drop customerID and target columns from feature matrix X
    drop_cols = ["customerID", "Churn", "Churn_Numeric"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    X = df[feature_cols].copy()
    y = df["Churn_Numeric"].copy()
    
    # Programmatically identify numerical and categorical features
    num_features = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # Ensure SeniorCitizen is treated as categorical if desired, or keep as numeric
    # Here SeniorCitizen is binary (0/1), tenure, MonthlyCharges, TotalCharges are continuous
    # Let's keep tenure, MonthlyCharges, TotalCharges as numerical
    # and SeniorCitizen can remain numeric or be categorical.
    # Treating tenure, MonthlyCharges, TotalCharges as numerical:
    continuous_num = ["tenure", "MonthlyCharges", "TotalCharges"]
    # All others (including SeniorCitizen) as categorical
    cat_features = [col for col in feature_cols if col not in continuous_num]
    num_features = continuous_num
    
    # Convert cat_features to string type to avoid type mismatch in encoders
    for col in cat_features:
        X[col] = X[col].astype(str)
        
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, num_features, cat_features

def build_preprocessor(num_features, cat_features):
    """
    Constructs a ColumnTransformer for numerical scaling and categorical one-hot encoding.
    Fitted solely inside the pipeline to prevent data leakage.
    """
    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_features),
            ("cat", cat_transformer, cat_features)
        ],
        remainder="drop"
    )
    
    return preprocessor

if __name__ == "__main__":
    df = load_and_clean_data()
    X_train, X_test, y_train, y_test, num_features, cat_features = get_feature_splits(df)
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    print(f"Numerical features ({len(num_features)}): {num_features}")
    print(f"Categorical features ({len(cat_features)}): {cat_features}")
    print(f"Train churn distribution:\n{y_train.value_counts(normalize=True)}")
    print(f"Test churn distribution:\n{y_test.value_counts(normalize=True)}")
