## Dataset Overview

The dataset analyzed in this study is provided by the SATDAP program (Capacitação da Administração Pública, grant POCI-05-5762-FSE-000191, Portugal). Sourced from a higher education institution, it is compiled from multiple databases and covers student records across a range of undergraduate programs, including agronomy, design, education, nursing, journalism, management, social service, and technology fields.

This dataset contains information available at the time of student enrollment, such as academic history, demographic attributes, and socio-economic indicators. It also includes academic performance data from the end of the first and second semesters.

In total, the dataset comprises 4,424 rows and 37 variables, offering a comprehensive view of student trajectories and outcomes for fairness and retention analysis.

---

## Data Preprocessing Steps
1. **Column Cleaning**: Standardize column names, remove leakage-prone columns, and fix naming inconsistencies (e.g., 'nacionality' to 'Nationality').
2. **Target Filtering**: Keep only 'Dropout' and 'Graduate' values in the target column, mapping them to 0 and 1.
3. **Column Name Consistency**: Replace invalid characters in column names with underscores for uniformity.
4. **Feature Binning**: Bin 'Age_at_enrollment' and 'Admission_grade' into categorical groups for easier analysis.
5. **Outlier Detection**: Use boxplots to detect outliers in all numerical features.
6. **Feature Distributions**: Plot distributions for all variables (numerical and categorical) to understand their spread and balance.
7. **Standard Dataset Creation**: Prepare the dataset for fairness analysis using the `StandardDataset` class.

---

## Outlier Analysis

![Outlier Boxplots](plot/outliers.png)

**Key Results:**

- Several features show significant outliers, especially in 'Admission_grade' and 'Age_at_enrollment'.
- Outliers may indicate data entry errors or rare cases; consider further investigation or robust modeling techniques.

---

## Variable Distributions

![Variable Distributions](plot/variable_distributions.png)

**Key Results:**

- Most categorical features (e.g., 'Gender', 'Scholarship_holder', 'Marital_status') are imbalanced, with one category dominating.
- Numerical features like 'Admission_grade' and 'Age_at_enrollment' are right-skewed, suggesting most students are younger and have moderate admission grades.
- The target variable is relatively balanced, but some class imbalance exists.

---

These EDA steps provide a comprehensive overview of the dataset, highlight potential issues (missing values, outliers, imbalance), and guide further modeling and fairness analysis. Addressing these findings will improve model robustness and fairness.