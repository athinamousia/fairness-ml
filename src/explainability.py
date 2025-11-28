import os
import pickle
import pandas as pd
import lime
import lime.lime_tabular
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import read_csv, split_data
from src.constants import PROTECTED_ATTRS


class Explainability:
    def __init__(self):
        self.input_file = "data/standard_df.csv"
        self.protected_attrs = PROTECTED_ATTRS
        self.target = "Target"

    def load_data_and_model(self):
        self.df = read_csv(self.input_file)
        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
            *_,
        ) = split_data(self.df, self.protected_attrs, self.target)
        with open("output/model/baseline_model.pkl", "rb") as f:
            self.model = pickle.load(f)

    def get_shap_values(self, max_samples=100):
        X_sample = self.X_test.sample(
            n=min(max_samples, len(self.X_test)), random_state=42
        )
        explainer = shap.Explainer(self.model, X_sample)
        shap_values = explainer(X_sample)
        shap_df = pd.DataFrame(shap_values.values, columns=X_sample.columns)
        shap_df["sample_index"] = X_sample.index
        shap_df.to_csv("output/csv/shap_values.csv", index=False)
        return shap_values, shap_df, X_sample

    def plot_bar(self, shap_values, plot_dir):
        plt.figure()
        shap.plots.bar(shap_values, show=False)
        plt.title("SHAP Feature Importance (Bar)")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "shap_bar.png"))
        plt.close()

    def plot_beeswarm(self, shap_values, plot_dir):
        plt.figure()
        shap.summary_plot(shap_values, plot_type="dot", show=False)
        plt.title("SHAP Summary (Beeswarm)")
        plt.savefig(os.path.join(plot_dir, "shap_beeswarm.png"))
        plt.close()

    def plot_dependence_subplots(
        self, shap_df, shap_values, X_sample, plot_dir, top_features=5
    ):
        feature_importance = (
            shap_df.drop("sample_index", axis=1)
            .abs()
            .mean()
            .sort_values(ascending=False)
        )
        top_feats = feature_importance.head(top_features).index
        fig, axes = plt.subplots(1, top_features, figsize=(5 * top_features, 5))
        if top_features == 1:
            axes = [axes]
        for i, feat in enumerate(top_feats):
            shap.dependence_plot(
                feat, shap_values.values, X_sample, ax=axes[i], show=False
            )
            axes[i].set_title(f"Dependence: {feat}")
        plt.suptitle("SHAP Dependence Plots (Top Features)")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(plot_dir, "shap_dependence_subplots.png"))
        plt.close()

    def run_pipeline(self, max_samples: int = 100, top_features: int = 5):
        self.load_data_and_model()
        shap_values, shap_df, X_sample = self.get_shap_values(max_samples)
        plot_dir = "docs/explainability/plot"
        os.makedirs(plot_dir, exist_ok=True)
        self.plot_bar(shap_values, plot_dir)
        self.plot_beeswarm(shap_values, plot_dir)
        self.plot_dependence_subplots(
            shap_df, shap_values, X_sample, plot_dir, top_features
        )
        return shap_values
