# Bias Mitigation

## 1. Pre-processing
Pre-processing mitigation techniques address bias before model training by transforming the data. Common methods include:
- **Reweighing**: Adjusts weights of training samples to balance protected groups.
- **Learning Fair Representations (LFR)**: Learns new feature representations that remove bias.

## 2. In-processing
In-processing mitigation techniques modify the learning algorithm to reduce bias during model training. Examples include:
- **Adversarial Debiasing**: Uses adversarial networks to minimize bias.
- **Exponentiated Gradient Reduction (EGR)**: Optimizes model parameters for fairness constraints.

## 3. Post-processing
Post-processing mitigation techniques adjust model predictions to improve fairness after training. Methods include:
- **Calibrated Equalized Odds**: Modifies predictions to equalize odds across groups.
- **Reject Option Classification**: Allows the model to abstain from making predictions in uncertain cases to reduce bias.
