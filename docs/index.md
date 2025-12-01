# Introduction

## Overview

Machine learning algorithms increasingly drive critical decisions in finance, healthcare, criminal justice, and education. While these systems offer powerful predictive capabilities, they also raise fundamental concerns about fairness and bias that must be addressed for responsible AI deployment.

Fairness in machine learning means that algorithms provide equitable outcomes across demographic groups without systematically disadvantaging certain populations.   

Bias enters ML systems through two primary pathways: **data bias** (skewed or unrepresentative training data reflecting historical inequities, incomplete diversity, or measurement errors) and **model bias** (algorithmic choices and feature selections that amplify existing biases).   

Sensitive attributes like race, gender, or socioeconomic status can become entangled with other features, causing models to discriminate even when these attributes aren't explicitly used.  

This project addresses algorithmic fairness using predictive models to identify students' droppout and academic success.This analysis investigates how bias appears in student success prediction and evaluates 7 fairness intervention techniques across pre-processing, in-processing, and post-processing stages to mitigate discriminatory patterns while maintaining predictive accuracy.

## Dataset

 The analysis uses the [Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) dataset from the UCI Machine Learning Repository, containing real-world academic retention data with demographic, socioeconomic, and academic performance features.

## Documentation Structure

This documentation is organized into four main sections:

1. **[Exploratory Data Analysis](data_processing/eda.md)**: Statistical analysis of data distributions, correlations, and patterns that may contribute to model bias before training begins.
2. **[Bias Detection](bias_detection/bias_detection.md)**: Pre-modeling and post-modeling bias identification
3. **[Bias Mitigation](bias_mitigation/bias_mitigation.md)**: Detailed evaluation of seven fairness intervention methods
4. **[Explainability](explainability/explainability.md)**: SHAP (SHapley Additive exPlanations) analysis to understand feature contributions and prediction rationale.

Navigate through the sections using the sidebar to explore the complete analysis, methodology, results, and interpretations.

## Framework and Tools

| Technology | Purpose |
|------------|---------|
| **AIF360** | IBM's AI Fairness 360 toolkit for bias detection and mitigation |
| **XGBoost** | Gradient boosting baseline classifier |
| **TensorFlow** | Deep learning framework for Adversarial Debiasing |
| **SHAP** | Model explainability and interpretability |
| **scikit-learn** | Machine learning utilities and evaluation metrics |



---

**Author**: Athina Mousia  
UoM - Business Analytics and Data Science Master's Thesis  
**Repository**: [github.com/athinamousia/fairness-ml](https://github.com/athinamousia/fairness-ml)