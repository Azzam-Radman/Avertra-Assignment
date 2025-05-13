# 🚀 Avertra Assignment

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-%23150458?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Numerical-%23013243?logo=numpy)
![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-%230C4A6E?logo=xgboost)
![CatBoost](https://img.shields.io/badge/CatBoost-Gradient%20Boosting-yellow?logo=catboost)
![LightGBM](https://img.shields.io/badge/LightGBM-Fast%20Boosting-%23008DFF?logo=lightgbm)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![MIT License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Overview

This repository contains a machine learning solution for Avertra's forecasting assignment. The objective is to predict a target variable using various regression models, data analysis, and feature engineering techniques.

---

## 📁 Repository Structure

```
Avertra-Assignment/
├── dataset.csv # Raw dataset
├── lr_model.pkl # Final trained Linear Regression model
├── models.py # Script containing model training logic
├── requirements.txt # List of dependencies
├── LICENSE # License file
├── README.md # Project documentation
└── Notebooks/
├── 1_EDA_Preprocessing.ipynb
├── 2_Model_Development.ipynb
└── 3_Inference.ipynb
```


---

## 🧠 Models Evaluated

- Linear Regression ✅
- Multilayer Perceptron (MLP)
- XGBoost
- CatBoost
- LightGBM
- GRU (Gated Recurrent Unit)
- LSTM (Long Short-Term Memory)

> 🏆 **Best Model:** Surprisingly, the **Linear Regression** model achieved the best performance, suggesting the dataset was not complex enough for more sophisticated architectures.

---

## 📊 Notebooks Description

1. **Exploratory Data Analysis & Preprocessing**  
   Data visualization, cleaning, and feature engineering.

2. **Model Development**  
   Trains and compares different machine learning models.  
   🔗 [Run it on Kaggle](https://www.kaggle.com/code/azzamradman/avertra-assignment-model-development)

3. **Inference**  
   Demonstrates model loading and prediction on new/unseen data.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip
- Jupyter Notebook

### Installation

```bash
git clone https://github.com/Azzam-Radman/Avertra-Assignment.git
cd Avertra-Assignment
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

### 📈 Results
The Linear Regression model performed best on this dataset. All evaluation metrics, visualizations, and model comparisons are available in the notebooks.

## 📜 License
This project is licensed under the MIT License. See LICENSE for more information.

