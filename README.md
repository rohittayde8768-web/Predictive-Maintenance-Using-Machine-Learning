# Predictive Maintenance for Industrial Equipment using Machine Learning

An end-to-end Machine Learning project that predicts industrial machine failures using the **AI4I 2020 Predictive Maintenance Dataset**. The project demonstrates the complete machine learning pipeline, including data understanding, exploratory data analysis (EDA), preprocessing, model training, hyperparameter tuning, feature importance analysis, model persistence, and prediction.

---

# Project Overview

Unexpected machine failures can lead to production downtime, increased maintenance costs, and reduced operational efficiency. Predictive maintenance aims to identify potential failures before they occur using historical machine sensor data.

In this project, multiple machine learning models were trained and compared to classify whether a machine is likely to fail. The best-performing model was selected, tuned, and saved for future predictions.

---

# Problem Statement

Develop a machine learning model capable of predicting industrial machine failures using operational sensor data.

The project covers:

- Dataset Understanding
- Exploratory Data Analysis (EDA)
- Data Preprocessing
- Feature Engineering
- Model Training
- Model Comparison
- Hyperparameter Tuning
- Feature Importance Analysis
- Model Persistence
- Prediction on New Data

---

# Dataset

**Dataset:** AI4I 2020 Predictive Maintenance Dataset

The dataset contains **10,000 machine observations** with operational sensor readings and machine information.

## Features

| Feature | Description |
|----------|-------------|
| Type | Machine quality (L, M, H) |
| Air Temperature | Ambient air temperature (Kelvin) |
| Process Temperature | Internal process temperature (Kelvin) |
| Rotational Speed | Rotational speed (RPM) |
| Torque | Applied torque (Nm) |
| Tool Wear | Tool wear duration (minutes) |
| Machine Failure | Target variable |

The dataset also contains individual failure-type columns (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`), which were removed during preprocessing to prevent **data leakage**.

---

# Repository Structure

```text
Predictive-Maintenance/
│
├── data/
│   └── ai4i2020.csv
│
├── images/
│   ├── correlation_heatmap.png
│   ├── feature_boxplots.png
│   ├── feature_histograms.png
│   ├── feature_importance.png
│   ├── machine_type_distribution.png
│   ├── model_comparison.png
│   ├── target_distribution.png
│   └── xgboost_confusion_matrix.png
│
├── Predictive_Maintenance.ipynb
├── predictive_maintenance_xgboost.pkl
├── README.md
├── requirements.txt
└── .gitignore
```

---

# Exploratory Data Analysis

## Target Distribution

The dataset is highly imbalanced, with machine failures accounting for only **3.39%** of all observations.

<p align="center">
<img src="images/target_distribution.png" width="500">
</p>

---

## Machine Type Distribution

Most observations belong to **Low (L)** quality machines, followed by **Medium (M)** and **High (H)** quality machines.

<p align="center">
<img src="images/machine_type_distribution.png" width="500">
</p>

---

## Feature Distributions

Histograms were used to understand the distribution of the numerical sensor features.

<p align="center">
<img src="images/feature_histograms.png" width="850">
</p>

---

## Boxplots

Boxplots were used to identify outliers and compare feature distributions.

<p align="center">
<img src="images/feature_boxplots.png" width="850">
</p>

---

## Correlation Heatmap

A correlation heatmap was generated to analyze relationships between numerical features and machine failure.

<p align="center">
<img src="images/correlation_heatmap.png" width="650">
</p>

---

# Data Preprocessing

The following preprocessing steps were performed:

- Removed identifier columns (`UDI`, `Product ID`)
- Removed failure-type columns (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) to prevent data leakage
- One-Hot Encoding of categorical features
- Stratified Train-Test Split (80:20)
- Standardization using StandardScaler

---

# Machine Learning Models

The following classification models were trained and evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

---

# Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|--------|---------:|----------:|--------:|---------:|---------:|
| Logistic Regression | 82.45% | 14.18% | 82.35% | 24.19% | 90.69% |
| Decision Tree | 97.80% | 68.18% | 66.18% | 67.16% | 82.54% |
| Random Forest | 98.15% | 87.80% | 52.94% | 66.06% | 96.16% |
| Gradient Boosting | 98.60% | 88.46% | 67.65% | 76.67% | 96.98% |
| **XGBoost** | **98.75%** | **89.09%** | **72.06%** | **79.67%** | **97.29%** |

---

## Model Comparison

<p align="center">
<img src="images/model_comparison.png" width="600">
</p>

---

# Best Model – XGBoost

Among all evaluated models, **XGBoost** achieved the best balance between Precision, Recall, F1-Score, and ROC-AUC, making it the final model selected for deployment.

---

## Confusion Matrix

<p align="center">
<img src="images/xgboost_confusion_matrix.png" width="500">
</p>

---

# Feature Importance

The XGBoost model identified the following features as the most influential:

1. Torque
2. Air Temperature
3. Tool Wear
4. Process Temperature
5. Rotational Speed
6. Machine Type (Medium)
7. Machine Type (Low)

<p align="center">
<img src="images/feature_importance.png" width="600">
</p>

---

# Hyperparameter Tuning

RandomizedSearchCV was used to optimize the XGBoost classifier.

The tuned model achieved higher precision but slightly lower recall than the original model. Therefore, the original XGBoost model was retained as it provided the best overall balance.

---

# Model Persistence

The final XGBoost model was saved using **Joblib** for future use.

```python
import joblib

joblib.dump(xgb_model, "predictive_maintenance_xgboost.pkl")
```

The saved model can later be loaded without retraining.

```python
loaded_model = joblib.load("predictive_maintenance_xgboost.pkl")
```

---

# Results

- Successfully built a complete predictive maintenance pipeline.
- Compared five classification models.
- XGBoost achieved the best overall performance.
- Identified the most influential machine parameters using feature importance.
- Saved the trained model for future inference.

---

# Future Improvements

- Perform k-fold cross-validation for more robust evaluation.
- Explore additional ensemble models such as LightGBM and CatBoost.
- Deploy the model using Streamlit or Flask.
- Integrate real-time industrial sensor data.
- Automate periodic model retraining.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/soneesh_07/Predictive-Maintenance.git
```

Navigate to the project:

```bash
cd Predictive-Maintenance
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

---

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Joblib
- Jupyter Notebook

---

# References

1. AI4I 2020 Predictive Maintenance Dataset
2. Scikit-learn Documentation
3. XGBoost Documentation
4. Pandas Documentation
5. NumPy Documentation
6. Matplotlib Documentation
7. Seaborn Documentation

---

# Author

**Soneesh Dega**

Electrical Engineering Undergraduate  
Indian Institute of Technology Kharagpur

**Areas of Interest**

- Machine Learning
- Embedded Systems
- Digital Design
- Predictive Analytics
- Data-Driven Engineering

---

⭐ If you found this project useful, consider giving it a **Star**!