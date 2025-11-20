import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)
import numpy as np
from constants import XGB_PARAMS, KDE_DISTRIBUTION_COLS
from utils import read_csv, save_to_pickle, save_to_csv, prepare_data, calculate_performance_and_fairness_metrics


# Pre-Modeling Bias
class PreModelingBias:
    """Analyse bias in data before model training."""
    def __init__(self, df, target="Target"):
        self.df = df
        self.target = target

    def correlation_analysis(self, save_path="output/plot/pre-modeling-detection/correlation_analysis.png"):
        """Compute Spearman correlation & p-values vs target results."""
        predictors = [c for c in self.df.columns if c != self.target]

        corrs, p_values = [], []
        for col in predictors:
            corr, p = stats.spearmanr(self.df[self.target], self.df[col])
            corrs.append(corr)
            p_values.append(p)

        results = pd.DataFrame({
            "Predictor": predictors,
            "Correlation": corrs,
            "p_value": p_values
        }).sort_values("Correlation", ascending=False)

        # --- Plot style ---
        sns.set_theme(style="whitegrid", font_scale=1.2)
        fig, ax = plt.subplots(figsize=(10, 0.5 * len(results)))

        # --- Color mapping by correlation strength ---
        norm = plt.Normalize(results["Correlation"].min(), results["Correlation"].max())
        sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
        colors = [sm.to_rgba(c) for c in results["Correlation"]]

        # --- Horizontal bar plot ---
        bars = ax.barh(results["Predictor"], results["Correlation"], color=colors)

        # --- Add correlation values and significance stars ---
        for i, (v, p) in enumerate(zip(results["Correlation"], results["p_value"])):
            ax.text(v + 0.02 if v >= 0 else v - 0.02,
                    i,
                    f"{v:.2f}",
                    va="center",
                    ha="left" if v >= 0 else "right",
                    fontsize=10)
            if p < 0.05:
                ax.text(1.02 if v >= 0 else -1.02,
                        i,
                        "*",
                        va="center",
                        ha="center",
                        fontsize=14,
                        weight="bold",
                        color="black")

        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linestyle="--", linewidth=0.6, color="#D9D9D9", alpha=0.6)
        ax.xaxis.grid(False)

        # --- Axis & title formatting ---
        ax.axvline(0, color="#888888", linewidth=1)
        ax.set_xlabel("Spearman Correlation Coefficient", fontsize=12, labelpad=10)
        ax.set_title("Spearman Correlation with Target", fontsize=14, weight="bold", pad=15)
        ax.set_xlim(-1, 1)
        sns.despine(ax=ax, left=True, bottom=True)

        # --- Colorbar ---
        cbar = fig.colorbar(sm, ax=ax, fraction=0.015, pad=0.04)
        cbar.set_label("Correlation Strength", rotation=270, labelpad=15)
        cbar.outline.set_visible(False)

        # --- Layout & save ---
        fig.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


    def plot_kde_distributions(
        self,
        cols=KDE_DISTRIBUTION_COLS,
        save_path="output/plot/pre-modeling-detection/kde_distributions.png"
    ):
        """Plot KDE distributions of target vs categorical columns with clear color labels."""
        n_cols = 2
        n_rows = int(np.ceil(len(cols) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
        axes = axes.flatten()

        sns.set_theme(style="whitegrid", font_scale=1.2)
        palette = sns.color_palette("coolwarm")

        for ax, col in zip(axes, cols):
            categories = self.df[col].dropna().unique()
            colors = palette[:len(categories)]  # assign one color per category

            for cat, color in zip(categories, colors):
                sns.kdeplot(
                    data=self.df[self.df[col] == cat],
                    x=self.target,
                    fill=True,
                    alpha=0.4,
                    linewidth=1.5,
                    color=color,
                    ax=ax,
                    label=str(cat)
                )

            # --- Formatting ---
            ax.set_title(f"{col} vs {self.target}", fontsize=13, weight="bold", pad=10)
            ax.set_xlabel(self.target, fontsize=11)
            ax.set_ylabel("Density", fontsize=11)

            # Show legend
            ax.legend(title=col, fontsize=9, title_fontsize=10, frameon=False, loc="upper right")

            ax.grid(True, linestyle="--", alpha=0.4)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)

        # Hide empty axes if any
        for ax in axes[len(cols):]:
            ax.axis("off")

        plt.suptitle(
            "KDE Distributions of Target by Categorical Variables",
            fontsize=15,
            weight="bold",
            y=1.02
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


# Post-Modeling Bias
class PostModelingBias:
    """Train models, evaluate, and compute fairness metrics."""

    def __init__(self, df: pd.DataFrame, target="Target"):
        self.df = df
        self.target = target

    def train_and_evaluate(
        self,
    ):
        """Train XGBoost, evaluate, plot, and return fairness metrics."""
        protected_attrs = list(self.df.columns)[:-1]

        X_train, X_test, y_train, y_test, *_ = prepare_data(
            self.df, protected_attrs, self.target
        )

        model = GridSearchCV(XGBClassifier(), XGB_PARAMS, cv=5, scoring="f1")
        model.fit(X_train, y_train)
        save_to_pickle(model.best_estimator_, "output/model/baseline_model.pkl")

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "Balanced Accuracy": round(balanced_accuracy_score(y_test, y_pred), 4),
            "F1-score": round(f1_score(y_test, y_pred), 4),
            "ROC-AUC": round(roc_auc_score(y_test, y_proba), 4),
        }

        self._plot_results(y_test, y_pred, y_proba, metrics)

        fairness_df = calculate_performance_and_fairness_metrics(y_test, y_pred, y_proba, protected_attrs)
        fairness_df.to_csv("output/csv/baseline_xgboost.csv", index=False)

        return fairness_df, metrics

    @staticmethod
    def _plot_results(y_true, y_pred, y_proba, metrics, save_path="output/plot/post-modeling-detection/baseline_model_evaluation.png"):
        """Plot confusion matrix, ROC curve, and evaluation metrics."""

        # --- Metrics ---
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = roc_auc_score(y_true, y_proba)

        # --- Style ---
        sns.set_theme(style="white", font_scale=1.2)
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        plt.subplots_adjust(wspace=0.3)

        # Palette
        main_color = "#4C72B0" 
        accent_color = "#55A868"  

        # --- Confusion Matrix ---
        sns.heatmap(
            cm_norm,
            annot=cm,
            fmt="d",
            cmap=sns.light_palette(main_color, as_cmap=True, reverse=False),
            cbar=False,
            ax=axs[0],
            linewidths=0.4,
            linecolor="white",
            annot_kws={"size": 12, "weight": "semibold", "color": "#333333"},
        )
        axs[0].set_title("Confusion Matrix", fontsize=14, weight="bold", pad=10)
        axs[0].set_xlabel("Predicted Label", fontsize=12)
        axs[0].set_ylabel("True Label", fontsize=12)
        axs[0].set_xticklabels(["Negative", "Positive"])
        axs[0].set_yticklabels(["Negative", "Positive"], rotation=0)
        axs[0].grid(False)

        # --- ROC Curve ---
        axs[1].plot(fpr, tpr, color=main_color, lw=2.5, label=f"AUC = {roc_auc:.3f}")
        axs[1].plot([0, 1], [0, 1], color="#999999", linestyle="--", lw=1)
        axs[1].set_xlim([0.0, 1.0])
        axs[1].set_ylim([0.0, 1.05])
        axs[1].set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
        axs[1].set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
        axs[1].set_title("ROC Curve", fontsize=14, weight="bold", pad=10)
        axs[1].legend(frameon=False, loc="lower right", fontsize=11)
        axs[1].grid(True, linestyle="--", alpha=0.3)

        # --- Evaluation Metrics ---
        bars = axs[2].bar(
            list(metrics.keys()),
            list(metrics.values()),
            color=sns.color_palette("crest", n_colors=len(metrics)),
            alpha=0.8,
        )
        axs[2].set_ylim(0, 1.05)
        axs[2].set_title("Evaluation Metrics", fontsize=14, weight="bold", pad=10)
        axs[2].set_ylabel("Score", fontsize=12)
        axs[2].set_xticklabels(list(metrics.keys()), rotation=0)

        for b in bars:
            axs[2].text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 0.03,
                f"{b.get_height():.2f}",
                ha="center",
                va="bottom",
                fontsize=11,
                weight="semibold",
                color="#333333",
            )

        for spine in ["top", "right", "left"]:
            axs[2].spines[spine].set_visible(False)
        axs[2].grid(axis="y", linestyle="--", alpha=0.3)

        # --- Final Layout ---
        plt.suptitle("Model Performance Summary", fontsize=16, weight="bold", y=1.03)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Run Fairness Pipelines")
    parser.add_argument(
        "--pipeline",
        type=str,
        choices=["pre-modeling", "post-modeling"],
        required=True,
        help="Choose which bias detection pipeline to run",
    )

    args = parser.parse_args()
    input_file = "data/standard_df.csv"
    df = read_csv(input_file)

    if args.pipeline == "pre-modeling":
        #--- Analyze bias before modeling ---
        pre_model_bias = PreModelingBias(df)

        # Save correlation and kde results
        pre_model_bias.correlation_analysis()
        pre_model_bias.plot_kde_distributions()

    elif args.pipeline == "post-modeling":
        #--- Train model and analyze bias after modeling ---
        post_model_bias = PostModelingBias(df)
        # Save evaluation and fairness metrics
        fairness_df, metrics = post_model_bias.train_and_evaluate()

if __name__ == "__main__":
    main()