# Fairness-Aware Machine Learning for Student Success Prediction

A comprehensive framework for detecting and mitigating algorithmic bias in student retention prediction systems. This project implements pre-processing, in-processing, and post-processing fairness interventions using the AIF360 toolkit and model explainability analysis.

📚 **[View Full Documentation](https://athinamousia.github.io/fairness-ml/)**

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [License](#license)

## Overview

This project addresses algorithmic fairness in predictive models for student academic outcomes. Using a real-world academic retention dataset, we:

- **Detect bias** through pre-modeling statistical analysis and post-modeling fairness metrics
- **Mitigate bias** using 7 different fairness-aware machine learning techniques
- **Explain predictions** with SHAP for model interpretability
- **Document findings** in a comprehensive MkDocs-based knowledge base


## Project Structure

```
fairness-ml/
├── data/                              # Dataset files
│   ├── academic_retention_dataset.csv # Raw academic data
│   └── standard_df.csv               # Preprocessed standardized data
├── src/                              # Source code modules
│   ├── bias_detection.py            # Pre/post-modeling bias detection
│   ├── bias_mitigation.py           # Fairness intervention methods
│   ├── dataprocess.py               # Data preprocessing pipeline
│   ├── explainability.py            # SHAP/LIME explainability
│   ├── utils.py                     # Utility functions
│   └── constants.py                 # Configuration constants
├── docs/                            # MkDocs documentation
│   ├── bias_detection/             # Bias detection analysis
│   ├── bias_mitigation/            # Mitigation method descriptions
│   ├── explainability/             # Model interpretability
│   └── data_processing/            # EDA and preprocessing docs
├── output/                          # Generated results
│   ├── csv/                        # Fairness metrics CSV files
│   └── model/                      # Trained model artifacts
├── main.py                         # CLI entry point
├── mkdocs.yml                      # Documentation configuration
└── requirements.txt                # Python dependencies
```

## Setup

1. **Clone the repository:**
```bash
git clone https://github.com/athinamousia/fairness-ml.git
cd fairness-ml
```

2. **Create and activate a virtual environment:**

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux/macOS:**
```bash
python -m venv .venv
source .venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

**Core Dependencies:**
- `aif360` - AI Fairness 360 toolkit for bias mitigation
- `xgboost` - Baseline classification model
- `scikit-learn` - Machine learning utilities
- `tensorflow` - Required for Adversarial Debiasing
- `pandas`, `numpy` - Data manipulation
- `matplotlib`, `seaborn` - Visualization
- `shap`, `lime` - Model explainability

## Usage

The project uses a command-line interface with pipeline-specific execution modes.

### General Syntax

```bash
python main.py --pipeline <pipeline_name>
```


## License

This project is part of a Master's thesis in UoM university on algorithmic fairness in educational predictive analytics.

---

**Author**: Athina Mousia  
**Repository**: [github.com/athinamousia/fairness-ml](https://github.com/athinamousia/fairness-ml)  

For questions or issues, please open an issue on the GitHub repository.