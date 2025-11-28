# Pre-Modeling Bias Detection

Pre-modeling bias detection examines the dataset before model training to identify potential sources of unfairness. Early detection enables proactive mitigation strategies, improving model fairness, reliability, and interpretability.

This analysis employs statistical techniques to uncover relationships between features and sensitive attributes that may lead to discriminatory outcomes:

- **Correlation analysis** - Quantifies relationships between variables
- **Kernel Density Estimation (KDE)** - Visualizes distributional differences across groups
- **Hypothesis testing** - Assesses statistical significance of observed patterns

---

## Spearman Correlation Analysis

The Spearman correlation coefficient (ρ) measures monotonic relationships between variables using rank-based comparisons rather than raw values. This approach is robust to non-linear relationships, non-normal distributions, and outliers, making it ideal for fairness analysis.

| ρ value | Interpretation |
|---------|----------------|
| +1      | Perfect positive monotonic relationship |
| 0       | No monotonic relationship |
| -1      | Perfect negative monotonic relationship |

**Statistical Significance:**

- **p < 0.05**: Statistically significant correlation (unlikely due to chance)
- **p ≥ 0.05**: Not statistically significant (may be due to random variation)


![Correlation Analysis](plot/correlation_analysis.png)


The Spearman correlation analysis with the target variable (1 = graduation, 0 = dropout) reveals key predictive relationships. Statistically significant correlations (marked with *) indicate features most associated with student outcomes:

**Key Findings:**

- **Scholarship holder** (ρ = 0.31)*: Strongest positive predictor. Scholarship recipients have significantly higher graduation rates.
- **Age at enrollment** (ρ = -0.32)*: Strongest negative predictor. Older students face substantially higher dropout risk, with non-traditional students experiencing greater challenges
- **Debtor** (ρ = -0.27)*: Being in debt strongly correlates with dropout, highlighting financial hardship as a critical barrier to completion.
- **Gender** (ρ = -0.25)*: Male studentsshow higher dropout rates compared to female students, revealing gender-based disparities in academic retention.
- **Application mode** (ρ = -0.24)*: Enrollment pathway influences retention, with certain admission routes associated with higher dropout risk
- **Admission grade** (ρ = 0.13)* and **Previous qualification (grade)** (ρ = 0.12)*: Academic preparation positively predicts graduation

Notably, economic indicators (*Unemployment rate*, *Inflation rate*, *GDP*), parental factors, and demographic variables (*Nationality*, *International*) show minimal correlation, suggesting limited direct impact on individual student retention.

The strong correlations with *Age*, *Debtor*, *Scholarship holder* and *Gender* represent potential bias sources, as models trained on these features may systematically disadvantage certain demographic groups. 

---

## Kernel Density Estimation (KDE)

Kernel Density Estimation visualizes the probability distribution of variables across different demographic groups. By comparing distribution curves between groups (e.g., graduates vs. dropouts), KDE plots reveal whether certain features exhibit systematic differences.

### Interpreting KDE Plots

- **Overlapping distributions**: Feature values are similar across groups, suggesting lower bias risk
- **Separated distributions**: Distinct patterns between groups indicate the feature may contribute to differential treatment

Significant distributional differences, particularly for features correlated with sensitive attributes, signal potential fairness concerns that need further investigation.

![KDE Distributions](plot/kde_distributions.png)

The KDE distributions compare feature distributions between graduates and dropouts (1 = graduation, 0 = dropout), revealing key patterns with fairness implications:

**Key Findings:**

- **Debtor**: Clear distributional separation shows non-debtors (0) concentrate heavily in the graduation group, while debtors (1) are more prevalent among dropouts, confirming financial hardship as a critical barrier.
- **Gender**: Male students (1) show higher concentration in the dropout distribution compared to females (0), aligning with the negative correlation observed.
- **Scholarship holder**: Scholarship recipients (1) demonstrate stronger concentration in the graduation distribution, while non-recipients (0) are more prevalent among dropouts.
- **Age at enrollment**: Graduates tend to be younger at enrollment, with the dropout distribution showing a rightward shift toward older ages, consistent with the negative correlation.
- **Application mode** and **Course**: Different enrollment pathways and course selections show distinct distributional patterns between graduates and dropouts, indicating these institutional factors significantly influence retention.
- **Parental background** (qualification/occupation): Distributions reveal socioeconomic stratification, with students from higher socioeconomic backgrounds showing greater concentration in the graduation group.

These distributional differences, particularly for *Debtor*, *Gender*, *Scholarship holder*, and *Age*, represent potential bias sources that models may learn and perpetuate, warranting careful consideration in fairness-aware modeling.
