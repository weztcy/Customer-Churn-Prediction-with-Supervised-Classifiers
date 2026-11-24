# Portfolio Project Data — Customer Churn Prediction with Supervised Classifiers

- **Category:** `machine-learning` (`Machine Learning`)
- **Accent Color:** `indigo`
- **Target Assets Folder:** `public/images/projects/customer_churn_prediction_with/`
- **Required Local Images:** `banner.webp`, `image01.webp` .. `image09.webp`

## TypeScript Data (`ProjectCaseStudy`)

```typescript
{
  "id": "customer-churn-prediction-with-supervised-classifiers",
  "code": "PRO-24 // ML-07",
  "title": "Customer Churn Prediction with Supervised Classifiers",
  "category": "machine-learning",
  "categoryLabel": "Supervised Learning",
  "year": "2024",
  "status": "Completed",
  "accentColor": "indigo",
  "repoUrl": "https://github.com/weztcy/Customer-Churn-Prediction-with-Supervised-Classifiers",
  "license": "MIT License",
  "banner": "/images/projects/customer_churn_prediction_with/banner.webp",
  "summary": "End-to-end supervised ML churn prediction pipeline using tuned Gradient Boosting, OOF threshold tuning, and a Streamlit decision-support app.",
  "projectDescription": "End-to-end supervised classification pipeline predicting telecom customer churn across 7,043 accounts. Benchmarks nine models under leakage-free preprocessing, tunes Gradient Boosting for PR-AUC, calibrates an out-of-fold 0.32 threshold, and serves risk playbooks in Streamlit.",
  "architectureSummary": "Raw CSV Cleaning -> Stratified 80/20 Split -> ColumnTransformer -> 5-Fold GridSearchCV (Gradient Boosting) -> OOF Threshold Tuning (0.32) -> Joblib Pipeline & Streamlit UI",
  "techStack": [
    "Python",
    "Scikit-Learn",
    "Pandas",
    "NumPy",
    "SciPy",
    "Matplotlib",
    "Seaborn",
    "Joblib",
    "Streamlit",
    "Jupyter Notebook"
  ],
  "telemetryMatrix": {
    "projectType": "Supervised Classification Pipeline",
    "targetPlatform": "Python CLI & Streamlit Web App",
    "primaryDomain": "Telecom Customer Churn Prediction",
    "architecturePipeline": "ColumnTransformer & Gradient Boosting",
    "testingAndAudit": "Verified",
    "projectStatus": "Completed",
    "projectYear": "2024"
  },
  "mlProjectInfo": {
    "projectType": "Predictive Classification Pipeline",
    "learningParadigm": "Supervised Learning",
    "modelDomain": "Telecom Customer Retention",
    "projectStatus": "Completed",
    "developmentYear": "2024",
    "capabilityBadges": [
      "Leakage-Free Pipeline",
      "OOF Threshold Tuning",
      "Streamlit Risk Scoring"
    ]
  },
  "mlProblemContext": "Telecommunications churn destroys recurring revenue, while replacing lost subscribers costs several times more than proactive retention. With only 26.54% positive churn cases, naive accuracy hides missed churners, requiring calibrated probability ranking and recall-focused thresholds to guide targeted retention outreach.",
  "mlDatasetOverview": {
    "profileBadge": "Telco Tabular Snapshot",
    "source": "Telco Customer Churn Dataset (dataset/original/Telco-Customer-Churn.csv)",
    "modality": "Tabular",
    "scale": "7,043 Customer Records (5,634 Train / 1,409 Test)",
    "inputShape": "N x 19 Predictors (3 Numerical, 16 Categorical)",
    "dataFormat": "CSV",
    "description": "Cross-sectional customer records spanning demographics, tenure, phone and internet subscriptions, security/support add-ons, contract terms, payment methods, and billing totals, partitioned via an 80/20 stratified train/test split.",
    "provenanceNote": "Zero duplicate rows or customerIDs; 11 blank TotalCharges entries (all tenure=0, Churn=No) cleaned via domain-consistent zero imputation."
  },
  "mlFeaturesAndDataCharacteristics": {
    "spaceBadge": "19-Feature Mixed Space",
    "spaceNote": "customerID excluded from features; SeniorCitizen cast as categorical alongside service and billing attributes.",
    "objectiveOutput": {
      "label": "Target Output",
      "name": "Churn (Churn_Numeric)",
      "type": "Binary Class",
      "distribution": "1,869 Churn (26.54%) vs 5,174 Retained (73.46%)"
    },
    "inputSignals": [
      {
        "name": "Contract",
        "type": "Categorical",
        "importanceOrRole": "Top Driver (0.0964 PR-AUC drop)",
        "description": "Billing commitment term (Month-to-month, One year, Two year); month-to-month accounts churn at 42.7% vs 2.8% for two-year plans."
      },
      {
        "name": "tenure",
        "type": "Numerical",
        "importanceOrRole": "Rank #2 (0.0878 PR-AUC drop)",
        "description": "Customer subscription duration in months (0-72); churned subscribers cluster in early months with a 10-month median vs 38 months retained."
      },
      {
        "name": "TotalCharges",
        "type": "Numerical",
        "importanceOrRole": "Rank #3 (0.0456 PR-AUC drop)",
        "description": "Cumulative customer billing in dollars; captures tenure-spending interactions after domain zero cleaning for new accounts."
      },
      {
        "name": "TechSupport & OnlineSecurity",
        "type": "Categorical",
        "importanceOrRole": "Protective Add-ons (0.0191 / 0.0188)",
        "description": "Value-added support and security subscriptions; accounts lacking these add-ons churn above 41% compared to roughly 15% with coverage."
      },
      {
        "name": "InternetService & PaymentMethod",
        "type": "Categorical",
        "importanceOrRole": "Service & Billing Signals",
        "description": "Fiber optic subscribers (41.9% churn) and electronic check payers (45.3% churn) show elevated attrition relative to DSL and auto-pay."
      },
      {
        "name": "MonthlyCharges",
        "type": "Numerical",
        "importanceOrRole": "Pricing Sensitivity (r = 0.193)",
        "description": "Recurring monthly bill in dollars; churners carry a higher median monthly charge ($79.65 vs $64.43 for retained customers)."
      }
    ],
    "dataCharacteristicsSummary": "The 19 predictors exhibit clear attrition gradients: churners average 10 months median tenure vs 38 months for retained users and higher median monthly bills ($79.65 vs $64.43). TotalCharges correlates strongly with tenure (r = 0.826), while month-to-month contracts, fiber optic, and electronic checks elevate churn."
  },
  "mlLearningApproach": {
    "paradigm": "Supervised Learning — Imbalanced Binary Classification",
    "methodologyDescription": "Evaluated nine supervised classifiers across 5-fold Stratified CV on the 5,634-row training set using PR-AUC, ROC-AUC, F1, and recall. Top candidates underwent GridSearchCV optimizing average_precision, followed by out-of-fold probability threshold sweeping (0.10 to 0.90) to select a recall-balanced cutoff.",
    "justification": "Sequential Gradient Boosting captures non-linear interactions between contract terms, tenure, and billing charges without the severe training memorization observed in default Random Forests (0.9999 train vs 0.8180 validation ROC-AUC), delivering the highest CV PR-AUC (0.6719) and calibrated probabilities."
  },
  "mlDataPreparation": [
    {
      "step": "01",
      "title": "Whitespace Stripping & Numeric Coercion",
      "description": "Trimmed all string columns and coerced TotalCharges from object strings to float, exposing 11 blank entries corresponding exclusively to new tenure=0 accounts.",
      "badge": "Data Cleaning"
    },
    {
      "step": "02",
      "title": "Domain Zero Imputation & Sensitivity Check",
      "description": "Assigned 0.0 to zero-tenure TotalCharges blanks and verified parity against median imputation via 5-fold Logistic Regression CV (0.8461 vs 0.8461 ROC-AUC).",
      "badge": "Domain Imputation"
    },
    {
      "step": "03",
      "title": "Stratified 80/20 Train/Test Partitioning",
      "description": "Dropped customerID, encoded Churn (Yes=1, No=0), and split 5,634 training and 1,409 untouched test rows with stratify=y and random_state=42.",
      "badge": "Leakage Control"
    },
    {
      "step": "04",
      "title": "ColumnTransformer Feature Encoding",
      "description": "Applied SimpleImputer(median) + StandardScaler to 3 numerical features and SimpleImputer(most_frequent) + OneHotEncoder(handle_unknown='ignore') to 16 categorical features.",
      "badge": "Pipeline Encoding"
    }
  ],
  "mlDataPreparationNote": "All preprocessing statistics (medians, means, variances, and one-hot categories) are fitted strictly within training folds inside scikit-learn Pipeline objects to eliminate data leakage.",
  "mlModelArchitectureAndTraining": {
    "algorithmName": "GradientBoostingClassifier (Tuned)",
    "architectureSummary": "200 shallow sequential regression trees (max_depth=3) with stochastic subsampling (0.8) and conservative shrinkage (0.03), chained behind a unified ColumnTransformer inside a single scikit-learn Pipeline.",
    "configurationBadge": "GridSearchCV PR-AUC",
    "hyperparameters": [
      {
        "name": "n_estimators",
        "value": "200",
        "description": "Number of sequential boosting stages fitted over residual gradients."
      },
      {
        "name": "learning_rate",
        "value": "0.03",
        "description": "Conservative step-size shrinkage regularizing each tree's contribution."
      },
      {
        "name": "max_depth",
        "value": "3",
        "description": "Limits individual tree depth to capture low-order feature interactions without overfitting."
      },
      {
        "name": "subsample",
        "value": "0.8",
        "description": "Stochastic gradient boosting fraction sampled per tree to reduce variance."
      },
      {
        "name": "decision_threshold",
        "value": "0.32",
        "description": "OOF F1-optimal probability cutoff selected strictly on 5-fold training predictions."
      }
    ],
    "objectiveCriterion": "Log-Loss (Binomial Deviance) with PR-AUC GridSearch Selection",
    "optimizationSolver": "Stagewise Gradient Descent over Regression Trees",
    "trainingProcess": "Training ran strictly inside 5-fold StratifiedKFold splits with random_state=42, fitting numerical scaling and categorical one-hot encoding per fold. GridSearchCV tuned Gradient Boosting across 36 parameter combinations for PR-AUC, followed by out-of-fold threshold selection (0.32) and a single full-train refit."
  },
  "mlEvaluationStrategy": {
    "validationMethod": "5-Fold Stratified CV & Untouched Holdout Test",
    "description": "Model comparison and hyperparameter tuning used 5-fold StratifiedKFold on the 5,634 training rows, threshold tuning swept 81 values on OOF training probabilities, and final metrics were measured once on the untouched 1,409-row test set.",
    "metrics": [
      {
        "label": "TEST ROC-AUC",
        "value": "0.8463",
        "benchmark": "Baseline: 0.5000"
      },
      {
        "label": "TEST PR-AUC",
        "value": "0.6608",
        "benchmark": "Baseline: 0.2654"
      },
      {
        "label": "BRIER SCORE",
        "value": "0.1351",
        "benchmark": "Target < 0.2000"
      },
      {
        "label": "TUNED RECALL",
        "value": "73.53%",
        "benchmark": "Default: 51.87%"
      },
      {
        "label": "TUNED F1-SCORE",
        "value": "0.6300",
        "benchmark": "Default: 0.5852"
      },
      {
        "label": "TUNED PRECISION",
        "value": "55.11%",
        "benchmark": "Baseline: 26.54%"
      },
      {
        "label": "TEST ACCURACY",
        "value": "77.08%",
        "benchmark": "Baseline: 73.46%"
      }
    ]
  },
  "mlPerformanceVisualizations": [
    {
      "title": "Streamlit Risk Assessment Console & Operating Controls",
      "badge": "Decision Support UI",
      "imageUrl": "/images/projects/customer_churn_prediction_with/image01.webp",
      "telemetryNote": "Interactive dark-mode interface displaying 0.32/0.50/custom threshold controls, risk presets, and 53.8% Elevated Risk scoring."
    },
    {
      "title": "Risk Stratification, Retention Playbook & Driver Breakdown",
      "badge": "Prescriptive Output",
      "imageUrl": "/images/projects/customer_churn_prediction_with/image02.webp",
      "telemetryNote": "Displays ELEVATED RISK classification, automated retention playbook guidance, itemized risk flags, and governance disclaimer."
    },
    {
      "title": "Tenure & Monthly Charges Distributions by Churn Status",
      "badge": "EDA Distributions",
      "imageUrl": "/images/projects/customer_churn_prediction_with/image03.webp",
      "telemetryNote": "Churned median tenure is 10 months vs 38 months retained; median monthly charges are $79.65 vs $64.43."
    },
    {
      "title": "Total Charges Distribution & Churn Comparison",
      "badge": "EDA Billing Profile",
      "imageUrl": "/images/projects/customer_churn_prediction_with/image04.webp",
      "telemetryNote": "Right-skewed cumulative billing distribution with median TotalCharges of $1,679.5 retained vs $703.5 churned."
    },
    {
      "title": "Probability Calibration, Score Distribution & CV Model Comparison",
      "badge": "Calibration & CV Benchmark",
      "imageUrl": "/images/projects/customer_churn_prediction_with/image05.webp",
      "telemetryNote": "Reliability curve achieves Brier Score 0.1351 alongside 5-fold CV comparison across 9 supervised classifiers."
    },
    {
      "title": "Test-Set ROC & Precision-Recall Curves",
      "badge": "Discrimination Curves",
      "imageUrl": "/images/projects/customer_churn_prediction_with/image06.webp",
      "telemetryNote": "Final tuned Gradient Boosting pipeline attains 0.8463 ROC-AUC and 0.6608 PR-AUC (2.49x over 26.5% baseline)."
    },
    {
      "title": "Churn Target Balance & Categorical Risk Segments",
      "badge": "EDA Categorical Drivers",
      "imageUrl": "/images/projects/customer_churn_prediction_with/image07.webp",
      "telemetryNote": "73.46% vs 26.54% class split with elevated churn in Month-to-month (42.7%), Fiber optic (41.9%), and Electronic check (45.3%)."
    },
    {
      "title": "Support Add-On Churn Rates & Threshold Confusion Matrices",
      "badge": "Confusion Matrix & Add-Ons",
      "imageUrl": "/images/projects/customer_churn_prediction_with/image08.webp",
      "telemetryNote": "Tuned 0.32 threshold raises test TP from 194 to 275 (+81 churners saved) and cuts FN from 180 to 99."
    },
    {
      "title": "Numerical Correlation Matrix & Permutation Feature Importance",
      "badge": "Explainability & Correlations",
      "imageUrl": "/images/projects/customer_churn_prediction_with/image09.webp",
      "telemetryNote": "Contract (0.0964), tenure (0.0878), and TotalCharges (0.0456) lead test-set PR-AUC permutation importance."
    }
  ],
  "mlVisualizationTelemetryNote": "Cross-validated Gradient Boosting (0.8482 ROC-AUC, 0.6719 tuned PR-AUC) generalizes consistently to the untouched test set (0.8463 ROC-AUC, 0.6608 PR-AUC) with well-calibrated probabilities (Brier Score 0.1351).",
  "mlModelInterpretationAndLimitations": {
    "interpretationSummary": "Test-set permutation importance (10 repeats) confirms contract commitment and customer longevity dominate model predictions, followed by cumulative charges and value-added support bundles. Calibration reliability (Brier 0.1351) aligns closely with observed churn frequencies while revealing non-linear risk interactions.",
    "keyDrivers": [
      {
        "factor": "Contract (Month-to-month vs Term)",
        "impact": "+0.0964 mean PR-AUC permutation importance; 42.7% month-to-month churn vs 2.8% two-year."
      },
      {
        "factor": "tenure (Account Longevity)",
        "impact": "+0.0878 mean PR-AUC permutation importance; early-tenure users (<12 mo) dominate attrition."
      },
      {
        "factor": "TotalCharges (Cumulative Spend)",
        "impact": "+0.0456 mean PR-AUC permutation importance; captures tenure-billing interactions (r = 0.826)."
      },
      {
        "factor": "TechSupport & OnlineSecurity",
        "impact": "+0.0191 and +0.0188 PR-AUC importance; absence of support add-ons aligns with >41% churn."
      },
      {
        "factor": "InternetService (Fiber optic)",
        "impact": "+0.0166 PR-AUC importance; Fiber optic plans exhibit 41.9% churn vs 19.0% for DSL."
      }
    ],
    "limitations": [
      {
        "title": "Cross-Sectional Snapshot Only",
        "description": "Lacks longitudinal usage trends, network outage logs, billing dispute records, and support ticket velocity."
      },
      {
        "title": "Unobserved Competitive Context",
        "description": "Regional competitor fiber rollouts and promotional price shocks are absent from the dataset, preventing causal attribution."
      },
      {
        "title": "Retention Outreach Cost Sensitivity",
        "description": "Operating at threshold 0.32 flags 35.4% of customers (224 false positives), requiring alignment with campaign incentive margins."
      }
    ],
    "futureImprovements": [
      {
        "title": "Longitudinal Behavioral Telemetry",
        "target": "Incorporate month-over-month usage deltas and support ticket frequency"
      },
      {
        "title": "Uplift & Treatment Effect Modeling",
        "target": "Estimate incremental retention lift rather than raw churn propensity"
      },
      {
        "title": "Automated Drift Monitoring",
        "target": "Track Population Stability Index and periodic pipeline recalibration"
      }
    ],
    "discoveryNote": "All feature importance rankings reflect statistical associations on historical snapshot data rather than proven causal mechanisms."
  },
  "mlModelArtifacts": [
    {
      "name": "best_model_pipeline.joblib",
      "format": "JOBLIB"
    },
    {
      "name": "cv_comparison_results.json",
      "format": "JSON"
    },
    {
      "name": "tuning_results.json",
      "format": "JSON"
    },
    {
      "name": "test_metrics.json",
      "format": "JSON"
    },
    {
      "name": "permutation_importance.json",
      "format": "JSON"
    },
    {
      "name": "sensitivity_analysis.json",
      "format": "JSON"
    }
  ],
  "mlMethodologySpecs": {
    "learningParadigm": "Supervised Binary Classification",
    "modelArchitecture": "Tuned Gradient Boosting Ensemble",
    "coreFramework": "Scikit-Learn",
    "validationMethod": "5-Fold Stratified Cross-Validation"
  }
}
```
