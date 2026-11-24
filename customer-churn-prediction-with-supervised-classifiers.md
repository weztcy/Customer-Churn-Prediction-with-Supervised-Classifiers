# Portfolio Project Data — Customer Churn Prediction with Supervised Classifiers

- **Category:** `machine-learning` (`Machine Learning`)
- **Accent Color:** `indigo`
- **Target Assets Folder:** `public/images/projects/customer_churn_prediction_with/`
- **Required Local Images:** `banner.webp`, `image01.webp` .. `image09.webp`

## TypeScript Data (`ProjectCaseStudy`)

```typescript
{
  "id": "customer-churn-prediction-with-supervised-classifiers",
  "code": "PRO-22 // ML-07",
  "title": "Customer Churn Prediction with Supervised Classifiers",
  "category": "machine-learning",
  "categoryLabel": "Supervised Learning",
  "year": "2023",
  "status": "Completed",
  "accentColor": "indigo",
  "repoUrl": "https://github.com/weztcy/Customer-Churn-Prediction-with-Supervised-Classifiers",
  "license": "MIT License",
  "banner": "/images/projects/customer_churn_prediction_with/banner.webp",
  "summary": "Supervised ML pipeline predicting telecom churn using tuned Gradient Boosting, OOF threshold tuning, and a Streamlit dashboard.",
  "projectDescription": "Built a leakage-free supervised classification pipeline on 7,043 Telco customer records, benchmarking nine classifiers and tuning Gradient Boosting via PR-AUC. Out-of-fold threshold optimization (0.32) raised churn recall to 73.53% and powers an interactive Streamlit risk-scoring dashboard.",
  "architectureSummary": "Telco CSV -> Domain-Zero Cleaning & Stratified 80/20 Split -> ColumnTransformer -> 5-Fold CV & GridSearchCV Gradient Boosting -> OOF Threshold (0.32) -> Joblib Pipeline & Streamlit UI",
  "techStack": [
    "Python",
    "Scikit-Learn",
    "pandas",
    "NumPy",
    "SciPy",
    "Matplotlib",
    "Seaborn",
    "joblib",
    "Streamlit",
    "Jupyter Notebook"
  ],
  "telemetryMatrix": {
    "projectType": "Supervised Binary Classification System",
    "targetPlatform": "Python CLI & Streamlit Web Dashboard",
    "primaryDomain": "Telecommunications Customer Retention",
    "architecturePipeline": "ColumnTransformer + Gradient Boosting",
    "testingAndAudit": "Verified",
    "projectStatus": "Completed",
    "projectYear": "2023"
  },
  "mlProjectInfo": {
    "projectType": "Supervised Classification Pipeline",
    "learningParadigm": "Supervised Learning",
    "modelDomain": "Telecom Customer Churn",
    "projectStatus": "Completed",
    "developmentYear": "2023",
    "capabilityBadges": [
      "Leakage-Free Preprocessing",
      "Threshold Optimization",
      "Interactive Risk Scoring"
    ]
  },
  "mlProblemContext": "Telecommunications providers lose recurring revenue when subscribers cancel unexpectedly, yet untargeted retention discounts waste outreach budgets on stable accounts. With a 26.54% churn rate, naive accuracy is misleading and requires calibrated risk probabilities paired with a recall-focused decision threshold.",
  "mlDatasetOverview": {
    "profileBadge": "Telco Tabular Snapshot",
    "source": "Telco Customer Churn Dataset (dataset/original/Telco-Customer-Churn.csv)",
    "modality": "Tabular",
    "scale": "7,043 customer records (5,634 train / 1,409 test; 0 duplicates)",
    "inputShape": "N x 19 Features (3 Continuous Numerical, 16 Categorical)",
    "dataFormat": "CSV",
    "description": "Cross-sectional telecommunications subscriber snapshot covering demographics, tenure, phone and internet services, add-on subscriptions, contract structure, payment channels, and billing amounts, split 80/20 with stratification on Churn.",
    "provenanceNote": "Public Telco Customer Churn benchmark dataset bundled in repository under MIT License code governance."
  },
  "mlFeaturesAndDataCharacteristics": {
    "spaceBadge": "19-Feature Mixed Space",
    "spaceNote": "customerID excluded; structural No internet/phone service categories preserved.",
    "objectiveOutput": {
      "label": "Target Output",
      "name": "Churn",
      "type": "Binary Class",
      "distribution": "No (0): 5,174 (73.46%) | Yes (1): 1,869 (26.54%), ~2.77:1 imbalance"
    },
    "inputSignals": [
      {
        "name": "Contract",
        "type": "Categorical",
        "importanceOrRole": "Top Driver (0.0964 PR-AUC drop)",
        "description": "Billing commitment term (Month-to-month, One year, Two year); month-to-month accounts churn at 42.7% versus 2.8% for two-year plans."
      },
      {
        "name": "tenure",
        "type": "Numerical",
        "importanceOrRole": "2nd Driver (0.0878 PR-AUC drop)",
        "description": "Account longevity in months (0 to 72); median tenure is 10 months for churners compared to 38 months for retained subscribers."
      },
      {
        "name": "TotalCharges",
        "type": "Numerical",
        "importanceOrRole": "3rd Driver (0.0456 PR-AUC drop)",
        "description": "Cumulative customer billing in dollars; strongly correlated with tenure (r = 0.826) and captures non-linear spending stability."
      },
      {
        "name": "TechSupport & OnlineSecurity",
        "type": "Categorical",
        "importanceOrRole": "Protective Add-ons (0.0191 / 0.0188)",
        "description": "Value-added support and security subscriptions; accounts lacking these add-ons churn above 41% versus ~15% when active."
      },
      {
        "name": "InternetService & PaymentMethod",
        "type": "Categorical",
        "importanceOrRole": "Service & Billing Signals (0.0166 / 0.0021)",
        "description": "Broadband tier and payment channel; Fiber optic (41.9% churn) and Electronic check (45.3% churn) show elevated attrition."
      }
    ],
    "dataCharacteristicsSummary": "Exploratory diagnostics reveal a 2.77:1 class imbalance, 11 zero-tenure records with blank TotalCharges strings, and strong churn concentration among month-to-month contracts, early-tenure accounts, fiber optic plans, and manual electronic check payers."
  },
  "mlLearningApproach": {
    "paradigm": "Supervised Binary Classification under Moderate Class Imbalance",
    "methodologyDescription": "Benchmark nine linear, decision-tree, bagging, and gradient-boosted classifiers via 5-fold stratified cross-validation inside a leakage-free ColumnTransformer pipeline, then optimize PR-AUC with GridSearchCV and tune the decision threshold out-of-fold.",
    "justification": "Sequential gradient boosting captures non-linear interactions between tenure, contract type, and cumulative charges without the severe training memorization observed in untuned Random Forests (0.9999 train vs. 0.8180 validation ROC-AUC)."
  },
  "mlDataPreparation": [
    {
      "step": "01",
      "title": "Whitespace Trimming & Numeric Coercion",
      "description": "Stripped whitespace across string columns and coerced TotalCharges to float, exposing 11 blank strings corresponding exclusively to tenure = 0, Churn = No accounts.",
      "badge": "Data Cleaning"
    },
    {
      "step": "02",
      "title": "Domain-Zero Imputation & Sensitivity Audit",
      "description": "Assigned TotalCharges = 0.0 for zero-tenure accounts and verified via 5-fold Logistic Regression CV that domain-zero (0.6615 PR-AUC) matches median imputation (0.6614).",
      "badge": "Imputation Audit"
    },
    {
      "step": "03",
      "title": "Stratified 80/20 Train-Test Split",
      "description": "Dropped customerID, encoded Churn_Numeric (Yes=1, No=0), cast SeniorCitizen and 15 categoricals to strings, and split 5,634 train / 1,409 test rows (random_state=42).",
      "badge": "Stratified Split"
    },
    {
      "step": "04",
      "title": "Leakage-Free ColumnTransformer Encoding",
      "description": "Applied median imputation plus StandardScaler to the 3 continuous features and most-frequent imputation plus OneHotEncoder(handle_unknown='ignore') to 16 categoricals.",
      "badge": "Pipeline Encoding"
    }
  ],
  "mlDataPreparationNote": "All imputers, scalers, and one-hot encoders fit strictly inside cross-validation training folds and on the full 5,634-sample training partition; the 1,409-sample test set remained untouched until final evaluation.",
  "mlModelArchitectureAndTraining": {
    "algorithmName": "Tuned GradientBoostingClassifier Pipeline",
    "architectureSummary": "Scikit-learn Pipeline chaining a ColumnTransformer (StandardScaler + OneHotEncoder) into a 200-stage GradientBoostingClassifier ensemble with depth-3 regression trees and 0.8 stochastic subsampling.",
    "configurationBadge": "GridSearchCV PR-AUC",
    "hyperparameters": [
      {
        "name": "n_estimators",
        "value": "200",
        "description": "Sequential boosting stages selected from [100, 150, 200] via 5-fold GridSearchCV."
      },
      {
        "name": "learning_rate",
        "value": "0.03",
        "description": "Shrinkage step size selected from [0.03, 0.05, 0.1] to regularize residual updates."
      },
      {
        "name": "max_depth",
        "value": "3",
        "description": "Maximum tree depth selected from [3, 4] to bound higher-order feature interactions."
      },
      {
        "name": "subsample",
        "value": "0.8",
        "description": "Stochastic subsampling fraction selected from [0.8, 1.0] to reduce variance."
      },
      {
        "name": "decision_threshold",
        "value": "0.32",
        "description": "Operating cutoff selected via out-of-fold probability sweep (0.10-0.90) maximizing F1 (0.6436)."
      }
    ],
    "objectiveCriterion": "Binomial Deviance Log-Loss (tuned for Average Precision / PR-AUC)",
    "optimizationSolver": "Stagewise Gradient Descent over Decision Trees",
    "trainingProcess": "Trained nine baseline classifiers via 5-fold StratifiedKFold on 5,634 training records, ran GridSearchCV optimizing average_precision (0.6719 CV PR-AUC), swept out-of-fold thresholds to select 0.32, and refitted the pipeline once on full training data."
  },
  "mlEvaluationStrategy": {
    "validationMethod": "5-Fold Stratified Cross-Validation & Untouched 20% Holdout Test",
    "description": "Candidate models were compared on 5-fold stratified training folds to monitor train-validation generalization gaps, threshold tuning used strictly out-of-fold training probabilities, and final calibrated metrics were measured once on 1,409 held-out test records.",
    "metrics": [
      {
        "label": "TEST ROC-AUC",
        "value": "0.8463",
        "benchmark": "Baseline > 0.5000"
      },
      {
        "label": "TEST PR-AUC",
        "value": "0.6608",
        "benchmark": "Baseline: 0.2654"
      },
      {
        "label": "BRIER SCORE",
        "value": "0.1351",
        "benchmark": "Target < 0.1500"
      },
      {
        "label": "TUNED RECALL (0.32)",
        "value": "73.53%",
        "benchmark": "Default: 51.87%"
      },
      {
        "label": "TUNED F1-SCORE",
        "value": "0.6300",
        "benchmark": "Default: 0.5852"
      },
      {
        "label": "TUNED ACCURACY",
        "value": "77.08%",
        "benchmark": "Default: 80.48%"
      }
    ]
  },
  "mlPerformanceVisualizations": [
    {
      "title": "Streamlit Risk Assessment Interface — Controls & Profile Form",
      "badge": "Decision Support UI",
      "imageUrl": "/images/projects/customer_churn_prediction_with/image01.webp",
      "telemetryNote": "Displays 53.8% Elevated Risk prediction above 0.32 threshold with prescriptive digital retention playbook."
    },
    {
      "title": "Streamlit Risk Assessment Interface — Risk Drivers & Governance",
      "badge": "Explainable UI",
      "imageUrl": "/images/projects/customer_churn_prediction_with/image02.webp",
      "telemetryNote": "Surfaces profile-level risk flags, protective retention drivers, and statistical governance disclaimer."
    },
    {
      "title": "Tenure & Monthly Charges Distributions by Churn Status",
      "badge": "EDA Distributions",
      "imageUrl": "/images/projects/customer_churn_prediction_with/image03.webp",
      "telemetryNote": "Churned accounts cluster at median 10 months tenure and $79.65 monthly charges vs. 38 months and $64.43 retained."
    },
    {
      "title": "Total Charges KDE & Boxplot Comparison",
      "badge": "EDA Billing",
      "imageUrl": "/images/projects/customer_churn_prediction_with/image04.webp",
      "telemetryNote": "Retained customers accumulate $1,679.5 median total charges compared to $703.5 for churners."
    },
    {
      "title": "Calibration Curve, Probability Histogram & 5-Fold CV Comparison",
      "badge": "CV & Calibration",
      "imageUrl": "/images/projects/customer_churn_prediction_with/image05.webp",
      "telemetryNote": "Reliability diagram achieves 0.1351 Brier score while Gradient Boosting leads 9 variants in CV ROC-AUC (0.8482)."
    },
    {
      "title": "Held-Out Test Set ROC & Precision–Recall Curves",
      "badge": "Test Discrimination",
      "imageUrl": "/images/projects/customer_churn_prediction_with/image06.webp",
      "telemetryNote": "Achieves 0.8463 ROC-AUC and 0.6608 PR-AUC (>2.49x the 26.5% baseline churn rate)."
    },
    {
      "title": "Churn Target Balance & Categorical Risk Breakdown",
      "badge": "EDA Cohorts",
      "imageUrl": "/images/projects/customer_churn_prediction_with/image07.webp",
      "telemetryNote": "Highlights 45.3% Electronic check, 42.7% Month-to-month, and 41.9% Fiber optic churn concentrations."
    },
    {
      "title": "Service Add-Ons, Paperless Billing & Confusion Matrix Shift",
      "badge": "Threshold Impact",
      "imageUrl": "/images/projects/customer_churn_prediction_with/image08.webp",
      "telemetryNote": "Tuning threshold from 0.50 to 0.32 increases true positives from 194 to 275 and cuts false negatives from 180 to 99."
    },
    {
      "title": "Numerical Correlation Matrix & Top 15 Permutation Importances",
      "badge": "Feature Attribution",
      "imageUrl": "/images/projects/customer_churn_prediction_with/image09.webp",
      "telemetryNote": "Test-set permutation ranks Contract (0.0964), tenure (0.0878), and TotalCharges (0.0456) as top PR-AUC drivers."
    }
  ],
  "mlVisualizationTelemetryNote": "Cross-validation ROC-AUC (0.8482) aligns closely with held-out test ROC-AUC (0.8463), confirming low variance and reliable calibration (0.1351 Brier score).",
  "mlModelInterpretationAndLimitations": {
    "interpretationSummary": "Test-set permutation importance confirms that contractual friction, account longevity, and cumulative billing dominate churn risk, reinforced by missing security or support add-ons. Lowering the cutoff to 0.32 trades 129 false positives to save 81 additional churners.",
    "keyDrivers": [
      {
        "factor": "Contract Type",
        "impact": "0.0964 mean PR-AUC decrease; Month-to-month subscribers churn at 42.7% versus 2.8% for Two-year contracts."
      },
      {
        "factor": "Customer Tenure",
        "impact": "0.0878 mean PR-AUC decrease; churners have a 10-month median tenure compared to 38 months for retained accounts."
      },
      {
        "factor": "TotalCharges",
        "impact": "0.0456 mean PR-AUC decrease; captures cumulative spend and non-linear longevity interactions (r = 0.826 with tenure)."
      },
      {
        "factor": "TechSupport & OnlineSecurity",
        "impact": "0.0191 and 0.0188 PR-AUC drops; customers without these sticky add-ons churn above 41% versus ~15% with coverage."
      }
    ],
    "limitations": [
      {
        "title": "Cross-Sectional Snapshot Only",
        "description": "Lacks longitudinal usage trends, billing dispute timestamps, network outage telemetry, and inbound support ticket counts."
      },
      {
        "title": "Missing Competitive & Causal Context",
        "description": "Unobserved competitor promotions or regional fiber rollouts can confound associations such as Fiber optic churn (41.9%)."
      },
      {
        "title": "Tenure Survivor Bias",
        "description": "Accounts exceeding 60 months represent naturally resilient cohorts, requiring careful interpretation against new signups."
      }
    ],
    "futureImprovements": [
      {
        "title": "Longitudinal Behavioral Features",
        "target": "Integrate month-over-month usage drops, late payments, and support escalation velocity."
      },
      {
        "title": "Causal Uplift Modeling",
        "target": "Estimate incremental retention treatment effect rather than raw churn probability to prevent incentive cannibalization."
      },
      {
        "title": "Automated MLOps Drift Monitoring",
        "target": "Add Population Stability Index tracking, scheduled retraining pipelines, and campaign A/B validation."
      }
    ],
    "discoveryNote": "All associations derived from 10-repeat test-set permutation importance on PR-AUC; predictive signals guide risk triage rather than proving causal root causes."
  },
  "mlModelArtifacts": [
    {
      "name": "best_model_pipeline.joblib",
      "format": "JOBLIB"
    },
    {
      "name": "test_metrics.json",
      "format": "JSON"
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
    "modelArchitecture": "Gradient Boosted Decision Trees",
    "coreFramework": "Scikit-Learn",
    "validationMethod": "5-Fold Stratified Cross-Validation"
  }
}
```
