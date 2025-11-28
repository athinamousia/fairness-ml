import pandas as pd
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from typing import List, Tuple, Any
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

from aif360.sklearn.datasets.utils import standardize_dataset
from aif360.sklearn.metrics import (
    average_odds_error,
    statistical_parity_difference,
    disparate_impact_ratio,
    equal_opportunity_difference,
    average_odds_difference,
)

import lime.lime_tabular


# File I/O
def read_csv(file_path, sep=None):
    """
    Read a CSV file into a pandas DataFrame.
    """
    return pd.read_csv(file_path, sep=sep)


def save_to_csv(df, file_path):
    """
    Save the processed DataFrame to a CSV file.
    """
    df.to_csv(file_path, index=False)


def save_to_pickle(obj, file_path):
    """
    Save an object to a pickle file.
    """
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


def load_from_pickle(file_path):
    """
    Load an object from a pickle file.
    """
    with open(file_path, "rb") as f:
        return pickle.load(f)


# Modeling
def average_odds_scoring(y_true, y_pred, scoring, protected_attrs):
    """
    Custom scoring based on the absolute Average Odds Error.
    """
    return abs(average_odds_error(y_true, y_pred, prot_attr=protected_attrs))


def run_grid_search(model, params, X_train, y_train, scorer, weights=None, njobs=4):
    """Fit GridSearchCV with optional sample weights."""
    grid = GridSearchCV(model, params, scoring=scorer, cv=5, n_jobs=njobs)
    if weights is not None:
        grid.fit(X_train, y_train, sample_weight=weights)
    else:
        grid.fit(X_train, y_train)
    return grid


# Data Preparation
def split_data(
    df,
    protected_attrs,
    target_col,
    train_ratio=0.7,
    random_state=0,
):
    """
    Standardize the dataset, split into train/test sets, and scale the features.
    Protected attributes are kept in their original binary form (not standardized).
    """
    X, y = standardize_dataset(df=df, prot_attr=protected_attrs, target=target_col)

    X.index = y.index = pd.MultiIndex.from_arrays(X.index.codes, names=X.index.names)
    y = pd.Series(y.factorize(sort=True)[0], index=y.index, name=y.name)

    # Separate protected attributes from other features
    X_protected = X[protected_attrs].copy()  # Keep original binary values
    X_features = X.drop(columns=protected_attrs)

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_features, y, range(len(df)), train_size=train_ratio, random_state=random_state
    )
    X_train_prot, X_test_prot = train_test_split(
        X_protected, train_size=train_ratio, random_state=random_state
    )

    # Standardize only non-protected features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_df = pd.DataFrame(
        X_train_scaled, index=X_train.index, columns=X_train.columns
    )
    X_test_df = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)

    # Concatenate back with protected attributes (NOT standardized)
    X_train_df = pd.concat([X_train_df, X_train_prot], axis=1)
    X_test_df = pd.concat([X_test_df, X_test_prot], axis=1)

    return X_train_df, X_test_df, y_train, y_test, idx_train, idx_test


# Metrics Calculation
def calculate_performance_and_fairness_metrics(
    y_true, y_pred, y_scores, protected_attrs
):
    """
    Compute evaluation and fairness metrics for a binary classifier.
    """
    results = pd.DataFrame({"Protected Attribute": protected_attrs})

    results["Balanced Accuracy"] = balanced_accuracy_score(y_true, y_pred)
    results["F1 Score"] = f1_score(y_true, y_pred)
    results["ROC AUC"] = roc_auc_score(y_true, y_scores)

    results["Statistical Parity Difference"] = [
        statistical_parity_difference(y_true, y_pred, prot_attr=attr)
        for attr in protected_attrs
    ]
    results["Disparate Impact Ratio"] = [
        disparate_impact_ratio(y_true, y_pred, prot_attr=attr)
        for attr in protected_attrs
    ]
    results["Equal Opportunity Difference"] = [
        equal_opportunity_difference(y_true, y_pred, prot_attr=attr)
        for attr in protected_attrs
    ]
    results["Average Odds Difference"] = [
        average_odds_difference(y_true, y_pred, prot_attr=attr)
        for attr in protected_attrs
    ]

    return results.round(4)


def build_metrics_list(
    mitigation_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    eval_metrics,
    fairness_metrics,
    alg_name: str,
    protected_attrs,
):
    """
    Create a list of DataFrames: one for evaluation metrics, one per protected attr fairness.
    """
    result = []

    # ---- Evaluation metrics
    eval_df = pd.DataFrame(
        {
            "Metric": eval_metrics * 2,
            "Value": list(mitigation_df[eval_metrics].iloc[0, :])
            + list(baseline_df[eval_metrics].iloc[0, :]),
            "Algorithm": [alg_name] * len(eval_metrics)
            + ["XGBoost"] * len(eval_metrics),
        }
    )
    result.append(eval_df)

    # ---- Fairness metrics per protected attr
    for attr in protected_attrs:
        mit_vals = mitigation_df[mitigation_df["Protected Attribute"] == attr][
            fairness_metrics
        ].values[0]
        base_vals = baseline_df[baseline_df["Protected Attribute"] == attr][
            fairness_metrics
        ].values[0]

        fairness_df = pd.DataFrame(
            {
                "Metric": fairness_metrics * 2,
                "Value": list(abs(mit_vals)) + list(abs(base_vals)),
                "Algorithm": [alg_name] * len(fairness_metrics)
                + ["XGBoost"] * len(fairness_metrics),
            }
        )
        result.append(fairness_df)

    return result


# Plotting
def plot_metrics_grid(metrics_frames, plot_titles, algorithm_labels):
    """
    Plot evaluation & fairness metrics in a 2x2 subplot grid.
    """
    sns_palette = sns.color_palette("coolwarm")
    palette = [sns_palette[0], sns_palette[2]]
    fig, axes = plt.subplots(2, 2, figsize=(20, 18))

    for idx, metrics_df in enumerate(metrics_frames):
        row, col = divmod(idx, 2)
        ax = axes[row, col]

        sns.barplot(
            x="Metric",
            y="Value",
            hue="Algorithm",
            data=metrics_df,
            ax=ax,
            palette=palette,
        )
        ax.set_title(plot_titles[idx], fontweight="bold", fontsize=14)
        ax.set_xlabel("", fontsize=13)
        ax.set_ylabel("", fontsize=13)
        ax.tick_params(axis="both", labelsize=12)
        sns.despine(ax=ax, top=True, right=True, left=True, bottom=False)
        # ax.set_yticks([])
        ax.grid(True, linestyle="--", alpha=0.4)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.set_axisbelow(True)

        for bar in ax.patches:
            height = bar.get_height()
            if height > 0:
                ax.annotate(
                    f"{height:.2f}",
                    (bar.get_x() + bar.get_width() / 2, height),
                    ha="center",
                    va="center",
                    fontsize=12,
                    xytext=(0, 5),
                    textcoords="offset points",
                )
    fig.suptitle(
        f"{algorithm_labels[0]} - XGBoost baseline model",
        fontsize=20,
        fontweight="bold",
        y=1.02,
        color="#333333",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(
        f"docs/bias_mitigation/plot/{algorithm_labels[0].lower().replace(' ', '_')}.png",
        dpi=300,
        bbox_inches="tight",
    )


# LIME Explanation
def explain_prediction_with_lime(
    sample_idx: int,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    model: Any,
    num_features: int = 20,
):
    """
    Generate a LIME explanation for a single prediction.
    """
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        mode="classification",
    )

    instance = X_test.iloc[sample_idx].values
    explanation = explainer.explain_instance(
        instance, model.predict_proba, num_features=num_features
    )
    return explanation.as_pyplot_figure(label=explanation.available_labels()[0])
