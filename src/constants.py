COLUMN_NAME_MAP = {
    "Daytime/evening attendance\t": "Daytime/evening attendance",
    "Nacionality": "Nationality",
}

# Selected columns used in the analysis
COLUMNS_TO_DROP = [
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (without evaluations)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 2nd sem (enrolled)",
    "Tuition fees up to date",
]


KDE_DISTRIBUTION_COLS = [
    "Debtor",
    "Gender",
    "Age_at_enrollment",
    "Marital_status",
    "Tuition_fees_up_to_date",
    "Admission_grade",
]

PROTECTED_ATTRS = ["Debtor", "Gender", "Scholarship holder"]

# Evaluation and fairness metrics
EVAL_METRICS = ["Balanced Accuracy", "F1 Score", "ROC AUC"]

FAIRNESS_METRICS = [
    "Statistical Parity Difference",
    "Disparate Impact Ratio",
    "Equal Opportunity Difference",
    "Average Odds Difference",
]

# Hyperparameter grids for model tuning
XGB_PARAMS = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.1, 0.01, 0.001],
}

LFR_PARAMS = {
    "reconstruct_weight": [1e-2, 1e-3, 1e-4],
    "target_weight": [10, 100, 1000],
    "fairness_weight": [0, 10, 100, 1000],
    "n_prototypes": [10, 25, 50],
    "max_iter": [500, 1000, 2000],
}

ADVERSARIAL_PARAMS = {
    "adversary_loss_weight": [0.1, 0.5, 0.7],
    "num_epochs": [25, 50, 100],
    "batch_size": [32, 64, 128],
    "classifier_num_hidden_units": [100, 200],
}

EGR_PARAMS = {"eta0": [0.01, 0.1, 2.0], "eps": [0.01, 0.05, 0.1]}
