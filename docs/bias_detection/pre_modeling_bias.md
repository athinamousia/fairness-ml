Before training machine learning models, it is important to detect potential biases in the dataset. Identifying bias early improves the fairness, reliability, and interpretability of the resulting model.
Common statistical techniques for pre-modeling bias analysis include:

- **Correlation analysis**
- **Kernel Density Estimation (KDE)**
- **Hypothesis testing**

### Spearman Correlation
Spearman correlation coefficient (ρ) is used to measure the strength and direction of the relationship between two variables based on ranks, not raw numerical values.
Spearman correlation is especially useful when the relationship between variables is not linear, the data is not normally distributed and need to capture monotonic relationships.

**Interpretation**


| ρ value | Meaning                   |
|---------|---------------------------|
| +1      | Perfect positive correlation |
| 0       | No correlation              |
| -1      | Perfect negative correlation |


Along with the Spearman correlation coefficient, a p-value is used to determine whether the observed correlation is statistically meaningful. 

- *p < 0.05* is considered statistically significant. 
- *p ≥ 0.05* suggests that the observed correlation could be due to random variation.

Spearman correlation is useful for understanding how variables relate to both the target and to sensitive attributes. Strong relationships with the target can highlight which variables are influential for prediction, while strong relationships with sensitive attributes may indicate potential sources of bias. 

![Correlation Analysis](plot/correlation_analysis.png)


### Kernel Density Estimation (KDE)
Kernel Density Estimation (KDE) is a method used to show how data values are distributed. 

A KDE plot makes it easy to see whether two groups have similar or different distributions.

If the curves are clearly different, the variable may be important for prediction — and it may also indicate potential bias, especially when the variable relates to sensitive information.

![KDE](plot/kde_distributions.png)
