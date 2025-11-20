import os
import pickle
import pandas as pd
import lime
import lime.lime_tabular
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from utils import read_csv, prepare_data


class Explainability:
    def __init__(self, model_file, input_file, protected_attrs, target='Target'):
        self.model_file = model_file
        self.input_file = input_file
        self.protected_attrs = protected_attrs
        self.target = target


    def load_and_split(self):
        self.df = read_csv(self.input_file)
        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
            *_,
        ) = prepare_data(self.df, self.protected_attrs, self.target)

        # Load trained model
        with open(self.model_file, "rb") as f:
            self.model = pickle.load(f)

    def run_shap(self, max_samples: int = 100, top_features: int = 5):
        """Run SHAP explainability, save plots and SHAP values CSV."""
        X_sample = self.X_test.sample(n=min(max_samples, len(self.X_test)), random_state=42)

        explainer = shap.Explainer(self.model, X_sample)
        shap_values = explainer(X_sample)

        # Convert SHAP values to DataFrame
        shap_df = pd.DataFrame(shap_values.values, columns=X_sample.columns)
        shap_df["sample_index"] = X_sample.index
        shap_df.to_csv('output/csv/shap_values.csv', index=False)

        # Plot 1: Bar plot
        plt.figure()
        shap.plots.bar(shap_values, show=False)
        plt.tight_layout()
        plt.savefig("output/plot/shap_bar.png")
        plt.close()

        # Plot 2: Summary violin
        plt.figure()
        shap.summary_plot(shap_values, plot_type="violin", show=False)
        plt.savefig("output/plot/shap_violin.png")
        plt.close()

        # Plot 3: Beeswarm
        plt.figure()
        shap.summary_plot(shap_values, plot_type="dot", show=False)
        plt.savefig("output/plot/shap_beeswarm.png")
        plt.close()

        # Plot 4: Dependence plots (top N features)
        feature_importance = shap_df.abs().mean().sort_values(ascending=False)
        top_feats = feature_importance.head(top_features).index

        for feat in top_feats:
            plt.figure()
            shap.dependence_plot(feat, shap_values.values, X_sample, show=False)
            plt.savefig(f"output/plot/shap_dependence_{feat}.png")
            plt.close()
        return shap_values
