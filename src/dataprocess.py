import pandas as pd
import numpy as np
import re
from aif360.datasets import StandardDataset
import seaborn as sns
import matplotlib.pyplot as plt
from src.utils import read_csv, save_to_csv
from src.constants import COLUMNS_TO_DROP, COLUMN_NAME_MAP


class DataPreprocessorPipeline:
    def __init__(self):
        """
        Initialize the DataPreprocessorPipeline with a CSV file path.
        """
        input_file = "data/academic_retention_dataset.csv"
        self.df = pd.read_csv(input_file, sep=";")

    def preprocess_columns(self):
        """
        1. Remove leakage-prone columns
        2. Clean and standardize column names (lowercase, underscores, remove parentheses)
        3. Capitalize first letter of each column
        4. Replace 'nacionality' with 'Nationality'
        5. Select only the desired columns
        """
        df = self.df.copy()
        # Remove leakage columns
        df = df.drop(columns=[col for col in COLUMNS_TO_DROP if col in df.columns])
        # Rename only columns in COLUMN_NAME_MAP, keep others unchanged
        df = df.rename(columns=COLUMN_NAME_MAP)
        return df

    def filter_target_values(self):
        """
        Keep only 'Dropout' and 'Graduate' values in the 'Target' column and replace them with 0 and 1.
        """
        self.df = self.df[self.df["Target"].isin(["Dropout", "Graduate"])]
        self.df["Target"] = self.df["Target"].replace({"Dropout": 0, "Graduate": 1})

    def bin_column(self, column_name, bins, labels):
        """
        Bin a column into specified ranges and assign labels.
        """
        self.df[column_name] = pd.cut(
            self.df[column_name], bins=bins, labels=labels, right=False
        ).astype(int)

    def plot_distribution_data(
        self,
        output_path="docs/data_processing/plot/variable_distributions.png",
    ):
        """
        Generate plots to visualize the processed data.
        """
        # Automatically determine numerical and categorical columns
        numerical_vars = self.df.select_dtypes(include=["number"]).columns.tolist()
        categorical_vars = self.df.select_dtypes(exclude=["number"]).columns.tolist()

        total_vars = numerical_vars + categorical_vars
        n_total = len(total_vars)
        n_cols = 4
        n_rows = int(np.ceil(n_total / n_cols))

        sns.set_theme(style="whitegrid", font_scale=1.2)
        palette = sns.color_palette("coolwarm")
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        axes = axes.flatten()
        fig.suptitle("Distribution of All Variables", fontsize=15, weight="bold")

        for i, var in enumerate(total_vars):
            if var in numerical_vars:
                sns.histplot(
                    self.df[var], kde=True, ax=axes[i], color=palette[i % len(palette)]
                )
            else:
                categories = self.df[var].dropna().unique()
                n_colors = len(categories)
                cat_palette = palette[:n_colors]
                sns.countplot(
                    x=var,
                    hue=var,
                    data=self.df,
                    ax=axes[i],
                    palette=cat_palette,
                    legend=False,
                )
            axes[i].set_title(var)
            axes[i].tick_params(axis="x")

        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(output_path)
        plt.close()

    def plot_outliers(self, output_path="docs/data_processing/plot/outliers.png"):
        """
        Boxplots for all numerical columns to detect outliers.
        """

        num_cols = self.df.select_dtypes(include=["number"]).columns.tolist()
        n_cols = 4
        n_rows = int(np.ceil(len(num_cols) / n_cols))
        palette = sns.color_palette("coolwarm")
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        axes = axes.flatten()
        for i, col in enumerate(num_cols):
            sns.boxplot(x=self.df[col], ax=axes[i], color=palette[i % len(palette)])
            axes[i].set_title(col)
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")
        plt.suptitle("Numerical Feature Outliers", fontsize=15, weight="bold")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(output_path)
        plt.close()

    def get_privileged_groups(self):
        """
        Create a nested list with the most frequent value in each column for rows where 'Target' is 1.
        """
        total_priv_attr = []
        total_columns = list(self.df.columns)[:-1]
        for col in total_columns:
            # Calculate percentage of Target==1 for each value in col
            value_counts = self.df.groupby(col)["Target"].mean()
            # Select value with highest percentage of Target==1
            priv_value = value_counts.idxmax()
            total_priv_attr.append([priv_value])
            print(f"Privileged group for {col}: {priv_value}")
        return total_priv_attr, total_columns

    def preprocess_standard_dataset(self, privileged_groups, total_columns):
        """
        Preprocess the dataset using the StandardDataset class.
        """
        categorical_features = list(
            set(self.df.select_dtypes(include="object").columns) - set(["Target"])
        )
        standard_df = (
            StandardDataset(
                df=self.df,
                label_name="Target",
                favorable_classes=[1],
                protected_attribute_names=total_columns,
                privileged_classes=privileged_groups,
                categorical_features=categorical_features,
            )
            .convert_to_dataframe()[0]
            .reset_index(drop=True)
        )
        return standard_df

    def run_pipeline(self, output_path="data/standard_df.csv"):
        """
        Run the entire preprocessing pipeline and save the final dataset to a CSV file.
        """

        self.df = self.preprocess_columns()
        self.filter_target_values()
        self.plot_distribution_data()
        self.plot_outliers()
        privileged_groups, total_columns = self.get_privileged_groups()
        standard_df = self.preprocess_standard_dataset(privileged_groups, total_columns)
        standard_df.columns = [
            COLUMN_NAME_MAP.get(col, col) for col in standard_df.columns
        ]
        save_to_csv(standard_df, output_path)
