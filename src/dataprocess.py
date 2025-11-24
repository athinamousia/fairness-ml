import pandas as pd
import numpy as np
import re
from aif360.datasets import StandardDataset
import seaborn as sns
import matplotlib.pyplot as plt
from src.utils import read_csv, save_to_csv
from src.constants import COLUMNS_TO_DROP


class DataPreprocessorPipeline:
    def __init__(self, df):
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
        """
        df = self.df.copy()
        # Remove leakage columns

        df = df.drop(columns=[col for col in COLUMNS_TO_DROP if col in df.columns])
        # Clean column names
        cols = [re.sub(r"\s+", "_", col.lower()) for col in df.columns]
        cols = [re.sub(r"[()\[\]{}]", "", col) for col in cols]
        cols = [re.sub(r"__+", "_", col) for col in cols]
        # Capitalize first letter, fix 'nacionality'
        new_columns = []
        for col in cols:
            if col == "nacionality":
                new_columns.append("Nationality")
            else:
                new_columns.append(col[:1].upper() + col[1:])
        df.columns = new_columns
        return df

    def filter_target_values(self):
        """
        Keep only 'Dropout' and 'Graduate' values in the 'Target' column and replace them with 0 and 1.
        """
        self.df = self.df[self.df["Target"].isin(["Dropout", "Graduate"])]
        self.df["Target"] = self.df["Target"].replace({"Dropout": 0, "Graduate": 1})

    def clean_column_names(self):
        """
        Ensure column names are consistent by replacing invalid characters with underscores.
        """
        self.df.columns = self.df.columns.str.replace(
            r"\/s+|[^a-zA-Z0-9]", "_", regex=True
        )

    def bin_column(self, column_name, bins, labels):
        """
        Bin a column into specified ranges and assign labels.
        """
        self.df[column_name] = pd.cut(
            self.df[column_name], bins=bins, labels=labels, right=False
        ).astype(int)

    def preprocess_age_and_admission(self):
        """
        Preprocess 'Age_at_enrollment' and 'Admission_grade' columns by binning them into groups.
        """
        age_bins = [0, 21, 30, 45, 60, float("inf")]
        age_labels = [1, 2, 3, 4, 5]
        self.bin_column("Age_at_enrollment", age_bins, age_labels)

        admission_bins = [0.0, 114.0, 133.0, 152.0, 171.0, float("inf")]
        admission_labels = [1, 2, 3, 4, 5]
        self.bin_column("Admission_grade", admission_bins, admission_labels)

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
            total_priv_attr.append(
                [self.df[self.df["Target"] == 1][col].value_counts().head(1).index[0]]
            )
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
        self.clean_column_names()
        self.plot_distribution_data()
        self.plot_outliers()
        self.preprocess_age_and_admission()
        privileged_groups, total_columns = self.get_privileged_groups()
        standard_df = self.preprocess_standard_dataset(privileged_groups, total_columns)
        save_to_csv(standard_df, output_path)
