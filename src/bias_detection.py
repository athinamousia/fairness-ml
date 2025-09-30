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
from constants import XGB_PARAMS, KDE_DISTRIBUTION_COLS
from utils import read_csv, save_to_csv, prepare_data, calculate_performance_and_fairness_metrics

# ----------------------------------------------------------------------
class PreModelingBias:
    """Analyse bias in data before model training."""
    def __init__(self, df, target="Target"):
        self.df = df
        self.target = target

    def correlation_analysis(self, save_path="output/plot/correlation_analysis.png"):
        """Compute Spearman correlation & p-values vs target, plot results."""
        predictors = [c for c in self.df.columns if c != self.target]

        p_values, corrs = [], []
        for col in predictors:
            corr, p = stats.spearmanr(self.df[self.target], self.df[col])
            corrs.append(corr)
            p_values.append(p)

        sns.set_palette("mako")
        plt.figure(figsize=(18, 8))
        ax = pd.Series(corrs, index=predictors).plot(kind="bar", width=0.9)

        for i, (v, p) in enumerate(zip(corrs, p_values)):
            ax.text(i, v + 0.02 if v >= 0 else v - 0.06, f"{v:.2f}", ha="center")
            if p < 0.05:
                ax.text(i, v + 0.04 if v >= 0 else v - 0.12, "*", ha="center", fontsize=16, weight="bold")

        ax.set_title("Spearman Correlation with Target", fontsize=14, weight="bold")
        ax.set_ylim([-0.45, 1])
        ax.set_yticks([])
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        plt.tight_layout()
        plt.savefig(save_path)

        results = pd.DataFrame({"Predictor": predictors, "Correlation": corrs, "p_value": p_values})
        save_to_csv(results, "output/csv/correlation_analysis.csv")
        

    def plot_kde_distributions(self, cols=KDE_DISTRIBUTION_COLS, save_path="output/plot/kde_distributions.png"):
        """Plot KDE distributions of target vs categorical columns."""
        fig, axes = plt.subplots(3, 2, figsize=(15, 8))
        axes = axes.flatten()

        for ax, col in zip(axes, cols):
            sns.kdeplot(
                data=self.df,
                x=self.target,
                hue=col,
                common_norm=False,
                palette="mako",
                ax=ax,
            )
            ax.set_xlabel(self.target)
            ax.set_ylabel("Density")

        plt.tight_layout()
        plt.savefig(save_path)


# ----------------------------------------------------------------------
# Post-Modeling Bias
# ----------------------------------------------------------------------
class PostModelingBias:
    """Train models, evaluate, and compute fairness metrics."""

    def __init__(self, df: pd.DataFrame, target="Target"):
        self.df = df
        self.target = target

    def train_and_evaluate(
        self,
    ):
        """Train XGBoost, evaluate, plot, and return fairness metrics."""
        param_grid = XGB_PARAMS

        protected_attrs = list(self.df.columns)[:-1]

        X_train, X_test, y_train, y_test, *_ = prepare_data(
            self.df, protected_attrs, self.target
        )

        model = GridSearchCV(XGBClassifier(), param_grid, cv=3, scoring="f1")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "Balanced Accuracy": round(balanced_accuracy_score(y_test, y_pred), 4),
            "F1-score": round(f1_score(y_test, y_pred), 4),
            "ROC-AUC": round(roc_auc_score(y_test, y_proba), 4),
        }

        self._plot_results(y_test, y_pred, y_proba, metrics)

        fairness_df = calculate_performance_and_fairness_metrics(y_test, y_pred, y_proba, protected_attrs)
        fairness_df.to_csv("output/csv/xgboost_df.csv", index=False)

        return fairness_df, metrics

    @staticmethod
    def _plot_results(y_true, y_pred, y_proba, metrics):
        """Draw confusion matrix, ROC curve and metric bars."""
        cm = confusion_matrix(y_true, y_pred)
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = roc_auc_score(y_true, y_proba)

        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        palette = sns.color_palette("mako", n_colors=2)

        # Confusion Matrix
        sns.heatmap(cm, annot=True, fmt="d", cmap="mako", ax=axs[0])
        axs[0].set_title("Confusion Matrix")
        axs[0].set_xlabel("Predicted")
        axs[0].set_ylabel("True")

        # ROC Curve
        axs[1].plot(fpr, tpr, color=palette[0], lw=2, label=f"AUC = {roc_auc:.2f}")
        axs[1].plot([0, 1], [0, 1], "k--")
        axs[1].legend(loc="lower right")
        axs[1].set_title("ROC Curve")
        axs[1].set_xlabel("FPR")
        axs[1].set_ylabel("TPR")

        # Metric bars
        bars = axs[2].bar(metrics.keys(), metrics.values(), color=palette)
        axs[2].set_ylim(0, 1)
        axs[2].set_title("Evaluation Metrics")
        for b in bars:
            axs[2].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"{b.get_height():.2f}", ha="center")
        for side in ["top", "right", "left", "bottom"]:
            axs[2].spines[side].set_visible(False)

        plt.tight_layout()
        plt.savefig("output/plot/post_model_bias_evaluation.png")

def main():
    input_file = "data/standard_df.csv"
    df = read_csv(input_file)
    #--- Analyze bias before modeling ---
    pre_model_bias = PreModelingBias(df)

    # Save correlation and kde results
    pre_model_bias.correlation_analysis()
    pre_model_bias.plot_kde_distributions()

    #--- Train model and analyze bias after modeling ---
    post_model_bias = PostModelingBias(df)

    # Save evaluation and fairness metrics
    fairness_df, metrics = post_model_bias.train_and_evaluate()

if __name__ == "__main__":
    main()