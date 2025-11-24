import pandas as pd
from aif360.sklearn.datasets.utils import standardize_dataset
from aif360.sklearn.metrics import (
    average_odds_error,
    statistical_parity_difference,
    disparate_impact_ratio,
    equal_opportunity_difference,
    average_odds_difference,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    make_scorer,
)
import lime


def scoring_func(y_true, y_pred, scoring, pref_attr):
    return abs(average_odds_error(y_true, y_pred, prot_attr=pref_attr))


def standardized_dataset_output(data, prot_attr, target):
    standard_dataset = standardize_dataset(df=data, prot_attr=prot_attr, target=target)

    X = standard_dataset[0]
    y = standard_dataset[1]
    X.index = y.index = pd.MultiIndex.from_arrays(X.index.codes, names=X.index.names)
    y = pd.Series(y.factorize(sort=True)[0], index=y.index, name=y.name)
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
        X, y, range(len(data)), train_size=0.7, random_state=0
    )
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)
    X_train = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
    X_test = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)
    return X_train, X_test, y_train, y_test, indices_train, indices_test


def create_metrics_dataset(y_test, y_pred, y_proba, prot_attr):
    data = pd.DataFrame()
    data["Attribute"] = prot_attr
    data["Balanced Accuracy"] = round(balanced_accuracy_score(y_test, y_pred), 4)
    data["f1-score"] = round(f1_score(y_test, y_pred), 4)
    data["ROC-AUC"] = round(roc_auc_score(y_test, y_proba), 4)
    data[f"Statistical Parity Difference"] = [
        round(statistical_parity_difference(y_test, y_pred, prot_attr=attr), 4)
        for attr in prot_attr
    ]
    data[f"Disparate Impact Ratio"] = [
        round(disparate_impact_ratio(y_test, y_pred, prot_attr=attr), 4)
        for attr in prot_attr
    ]
    data[f"Equal Opportunity Difference"] = [
        round(equal_opportunity_difference(y_test, y_pred, prot_attr=attr), 4)
        for attr in prot_attr
    ]
    data[f"Average Odds Difference"] = [
        round(average_odds_difference(y_test, y_pred, prot_attr=attr), 4)
        for attr in prot_attr
    ]

    return data


# Function to plot evaluation and fairness metrics for multiple algorithms
def plot_2x2_metrics(metrics_list, labels, algorithms):
    # Define a consistent color palette
    palette = sns.color_palette("mako", n_colors=len(algorithms))
    # Create a figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(20, 18))

    # Iterate through each DataFrame and subplot
    for i, metrics in enumerate(metrics_list):
        row, col = i // 2, i % 2
        axs_current = axs[row, col]

        # Plot evaluation metrics (Plot 1)
        sns.barplot(
            x="Metric",
            y="Value",
            hue="Algorithm",
            data=metrics,
            ax=axs_current,
            palette=palette,
        )
        axs_current.set_xlabel("Metrics", fontsize=13)
        axs_current.set_ylabel("", fontsize=13)
        axs_current.set_title(f"{labels[i]}", fontweight="bold", fontsize=14)
        axs_current.tick_params(axis="both", labelsize=12)
        sns.despine(ax=axs_current, top=True, right=True, left=True, bottom=False)

        # Remove y-ticks
        axs_current.set_yticks([])

        # Add metric values above bars
        for p in axs_current.patches:
            if p.get_height() > 0.00:
                axs_current.annotate(
                    f"{p.get_height():.2f}",
                    (p.get_x() + p.get_width() / 2.0, p.get_height()),
                    ha="center",
                    va="center",
                    fontsize=13,
                    color="black",
                    xytext=(0, 5),
                    textcoords="offset points",
                )

    plt.tight_layout()
    plt.show()


def lime_explanation(idx, X_train, X_test, grid_search):
    # Initialize LIME explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        mode="classification",
    )

    # Explain a prediction

    instance_to_explain = X_test.iloc[idx].values
    exp = explainer.explain_instance(
        instance_to_explain, grid_search.predict_proba, num_features=20
    )
    exp.as_pyplot_figure(label=exp.available_labels()[0])
