Post-modeling bias detection takes place after the predictive model has been trained. The goal is to check whether the model makes fair decisions for different demographic groups. 
To evaluate fairness, several metrics are commonly used:

- **Statistical Parity Difference**: Shows the difference in the probability of receiving a positive outcome between the privileged and unprivileged groups. A value of zero means both groups have the same chance, while values far from zero suggest bias.

- **Disparate Impact Ratio**: Compares the rate of positive outcomes between unprivileged and privileged groups. A value of 1 means no disparate impact. A value below 1 indicates bias against the unprivileged group, while a value above 1 indicates bias against the privileged group.

- **Equal Opportunity Difference**:  Measures the difference in true positive rates between the two groups. A value close to zero means both groups have an equal chance of being correctly identified.

- **Average Odds Difference**: Looks at differences in both false positive and true positive rates across groups. A value close to zero suggests the model is fair in terms of both types of errors.


| **Metric**                              | **Equation**                                                                  | **Meaning of Symbols**                                                                    |
| --------------------------------------- | ----------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| **Statistical Parity Difference (SPD)** | `SPD = P(Ŷ = 1 | A = 1) - P(Ŷ = 1 | A = 0)`        | `A`: protected attribute (1 = privileged, 0 = unprivileged); `Ŷ`: model prediction |
| **Disparate Impact Ratio (DIR)**        | `DIR = P(Ŷ = 1 | A = 0) / P(Ŷ = 1 | A = 1)` | Ratio of positive outcomes for unprivileged vs privileged group                           |
| **Equal Opportunity Difference (EOD)**  | `EOD = TPR_{A=1} - TPR_{A=0}`                                        | `TPR`: True Positive Rate = `P(Ŷ = 1 | Y = 1)`                                   |
| **Average Odds Difference (AOD)**       | `AOD = [(TPR_{A=1} - TPR_{A=0}) + (FPR_{A=1} - FPR_{A=0})] / 2` | `FPR`: False Positive Rate = `P(Ŷ = 1 | Y = 0)`                                  |





![Evaluation](plot/baseline_model_evaluation.png)

