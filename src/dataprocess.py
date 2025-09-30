import pandas as pd
from aif360.datasets import StandardDataset
import seaborn as sns
import matplotlib.pyplot as plt
from utils import read_csv, save_to_csv

class DataPreprocessorPipeline:
    def __init__(self, df):
        """
        Initialize the DataPreprocessorPipeline with a CSV file path.
        """
        self.df = df

    def filter_target_values(self):
        """
        Keep only 'Dropout' and 'Graduate' values in the 'Target' column and replace them with 0 and 1.
        """
        self.df = self.df[self.df['Target'].isin(['Dropout', 'Graduate'])]
        self.df['Target'].replace({'Dropout': 0, 'Graduate': 1}, inplace=True)

    def clean_column_names(self):
        """
        Ensure column names are consistent by replacing invalid characters with underscores.
        """
        self.df.columns = self.df.columns.str.replace(r'\/s+|[^a-zA-Z0-9]', '_', regex=True)

    def bin_column(self, column_name, bins, labels):
        """
        Bin a column into specified ranges and assign labels.
        """
        self.df[column_name] = pd.cut(self.df[column_name], bins=bins, labels=labels, right=False).astype(int)

    def preprocess_age_and_admission(self):
        """
        Preprocess 'Age_at_enrollment' and 'Admission_grade' columns by binning them into groups.
        """
        age_bins = [0, 21, 30, 45, 60, float('inf')]
        age_labels = [1, 2, 3, 4, 5]
        self.bin_column('Age_at_enrollment', age_bins, age_labels)

        admission_bins = [0.0, 114.0, 133.0, 152.0, 171.0, float('inf')]
        admission_labels = [1, 2, 3, 4, 5]
        self.bin_column('Admission_grade', admission_bins, admission_labels)

    def plot_distibution_data(self, output_path='output/plot/variable_distributions.png'):
        """
        Generate plots to visualize the processed data.
        """
        numerical_vars = ['Previous_qualification', 'Admission_grade', 'Age_at_enrollment', 'Unemployment_rate']
        categorical_vars = ['Marital_status', 'Gender', 'Scholarship_holder', 'Target']

        # Setting up the plot
        fig, axes = plt.subplots(2, 4, figsize=(15, 8))
        fig.suptitle('Distribution of Selected Variables')

        # Plotting histograms for numerical variables
        for i, var in enumerate(numerical_vars):
            sns.histplot(self.df[var], kde=True, ax=axes[0, i])
            axes[0, i].set_title(var)

        # Plotting bar plots for categorical variables
        for i, var in enumerate(categorical_vars):
            sns.countplot(x=var, data=self.df, ax=axes[1, i])
            axes[1, i].set_title(var)
            axes[1, i].tick_params(axis='x', rotation=45)

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
            total_priv_attr.append([self.df[self.df['Target'] == 1][col].value_counts().head(1).index[0]])
        return total_priv_attr, total_columns

    def preprocess_standard_dataset(self, privileged_groups, total_columns):
        """
        Preprocess the dataset using the StandardDataset class.
        """
        categorical_features = list(set(self.df.select_dtypes(include="object").columns) - set(['Target']))
        standard_df = StandardDataset(
            df=self.df,
            label_name='Target',
            favorable_classes=[1],
            protected_attribute_names=total_columns,
            privileged_classes=privileged_groups,
            categorical_features=categorical_features
        ).convert_to_dataframe()[0].reset_index(drop=True)
        return standard_df
    def run_pipeline(self, output_path='data/standard_df.csv'):
        """
        Run the entire preprocessing pipeline and save the final dataset to a CSV file.
        """
        self.filter_target_values()
        self.clean_column_names()
        self.plot_distibution_data()
        self.preprocess_age_and_admission()
        privileged_groups, total_columns = self.get_privileged_groups()
        standard_df = self.preprocess_standard_dataset(privileged_groups, total_columns)
        save_to_csv(standard_df, output_path)



def main():
    input_file = 'data/academic_retention_dataset.csv'
    df = read_csv(input_file, sep=';') 
    data_processor = DataPreprocessorPipeline(df=df)
    data_processor.run_pipeline()

if __name__ == "__main__":
    main()