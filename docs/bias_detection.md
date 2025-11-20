# Bias Detection

## 1. Pre-Modeling Bias Detection
Before training machine learning models, it is important to detect potential biases in the dataset. Identifying bias early improves the fairness, reliability, and interpretability of the resulting model.
Common statistical techniques for pre-modeling bias analysis include:

- **Correlation analysis**
- **Kernel Density Estimation (KDE)**
- **Hypothesis testing**

## 2. Post-modeling
Post-modeling bias detection evaluates bias after model training. It includes:
- **Model Training & Evaluation**: Trains an XGBoost classifier and evaluates its performance using metrics such as Balanced Accuracy, F1-score, and ROC-AUC.
- **Fairness Metrics**: Computes fairness metrics and visualizes results with confusion matrices, ROC curves, and metric bar plots.
