# Bias Mitigation

Bias mitigation techniques address fairness violations by reducing discriminatory patterns that machine learning models learn from biased training data. Without intervention, models can perpetuate and amplify existing societal inequalities, leading to systematically unfair outcomes for protected demographic groups.

## Mitigation Strategies

Bias mitigation approaches been used in the next sections are summarized below:

| Category            | Timing | Purpose                                  | Model Techniques                         |
| ------------------- | ------ | ---------------------------------------- | ------------------------------------------ |
| **Pre-processing**  | Before training | Transform data to remove bias while preserving predictive information | Reweighing, Learning Fair Representations |
| **In-processing**   | During training | Incorporate fairness constraints directly into model optimization | Adversarial Debiasing, Exponentiated Gradient Reduction |
| **Post-processing** | After training | Adjust model outputs to satisfy fairness criteria without retraining | Calibrated Equalized Odds, Reject Option Classification |


This project implements bias mitigation techniques using [AI Fairness 360 (AIF360)](https://github.com/Trusted-AI/AIF360), an open-source toolkit developed by IBM Research. AIF360 provides comprehensive algorithms for detecting and mitigating bias across the machine learning lifecycle, along with standardized fairness metrics for evaluation.

The following sections present each mitigation approach, demonstrate its application to the student retention dataset, evaluate performance using fairness metrics, and compare results against the baseline model to quantify improvements in both fairness and predictive accuracy.

