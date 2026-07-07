# UAS-Machine-Learning
Kelompok 18

Fadhillah Rizqia Arfin
Muhammad Jalallullail
Raditya Raihan

Komparasi Algoritma Boosting (XGBoost, LightGBM, CatBoost) untuk Prediksi Dropout Mahasiswa dengan Interpretabilitas SHAP

# Student Dropout Prediction using Boosting Algorithms with SHAP Interpretability

This repository contains the source code and experimental results for predicting student dropout and academic success. The study implements and compares three state-of-the-art tree-based boosting algorithms: **XGBoost**, **LightGBM**, and **CatBoost**, integrated with **SHAP (SHapley Additive exPlanations)** for model interpretability.

---

## 🚀 Experimental Results & Model Evaluation

The models were evaluated using **10-fold Cross-Validation** on the *Predict Students Dropout and Academic Success* dataset from the UCI Machine Learning Repository. Below is the summary of the performance metrics achieved by each model:

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LightGBM** | **88.47%** | 85.04% | 78.13% | **81.45%** | 0.9315 |
| **XGBoost** | 88.10% | 83.92% | **78.55%** | 81.15% | **0.9328** |
| **CatBoost** | 87.90% | **85.34%** | 78.12% | 81.08% | 0.9310 |

### Key Takeaways:
* **LightGBM** achieved the highest overall **Accuracy (88.47%)** and **F1-Score (81.45%)**, making it highly reliable for balanced prediction.
* **XGBoost** demonstrated the best discriminative capacity with the highest **ROC-AUC of 0.9328**.
* **CatBoost** showed competitive performance, yielding the highest **Precision** score.

---

## 🔍 Model Interpretability via SHAP

To avoid the "black-box" nature of machine learning models, we utilized **SHAP global and local explanations** to uncover the main driving factors behind student dropouts. 

### Global Feature Importance
Based on the SHAP Summary Plot (Beeswarm), the most critical features influencing the model's predictions are:
1. **Curricular units 2nd sem (approved):** The number of approved courses in the second semester is the strongest predictor. Lower approved units drastically increase the risk of dropout.
2. **Tuition fees up to date:** Financial stability plays a vital role. Students with unpaid tuition fees are significantly more likely to drop out.
3. **Curricular units 1st sem (approved):** Similar to the second semester, early academic performance in the first semester heavily determines retention.

---

## 🛠️ Tech Stack & Methodology

- **Language:** Python 3.x
- **Libraries Used:** `xgboost`, `lightgbm`, `catboost`, `shap`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`
- **Validation Strategy:** 10-Fold Cross-Validation to prevent overfitting and ensure generalization.
- **Handling Imbalance:** SMOTE (Synthetic Minority Over-sampling Technique) was implemented to handle class imbalance in student outcomes.

---

## 📂 Repository Structure

```text
├── data/                  # Dataset directory
├── notebooks/             # Jupyter Notebooks containing data exploration & modeling
│   └── student_dropout_analysis.ipynb
├── src/                   # Python source scripts
├── README.md              # Project documentation
└── requirements.txt       # Project dependencies
