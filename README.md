# 🫀 10-Year Coronary Heart Disease (CHD) Risk Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

An end-to-end Machine Learning classification project built using the *Framingham Heart Study* dataset to predict whether a patient is at risk of developing *Coronary Heart Disease (CHD)* within a 10-year window.

This project addresses a critical real-world clinical challenge: *the Accuracy Paradox in imbalanced medical datasets. By transitioning from an unweighted baseline model to a cost-sensitive Logistic Regression model (class_weight='balanced'), patient risk detection (Recall) improved from **8.00% to 72.00%*.

---

## 📌 Table of Contents
1. [Project Overview](#-project-overview)
2. [Dataset Summary](#-dataset-summary)
3. [Data Preprocessing & Pipeline](#-data-preprocessing--pipeline)
4. [Model Comparison: Before vs. After](#-model-comparison-before-vs-after)
5. [Confusion Matrix & Classification Report](#-confusion-matrix--classification-report)
6. [Tech Stack](#-tech-stack)
7. [How to Run Locally](#-how-to-run-locally)
8. [Author](#-author)

---

## 🔍 Project Overview

In medical diagnostics, missing an at-risk patient (*False Negative) carries a much higher cost than flagging a healthy patient for a routine follow-up checkup (False Positive*).

* *The Challenge (Class Imbalance):* In the Framingham dataset, *~84.75%* of individuals do not develop heart disease. A standard Logistic Regression model achieves *84.90% accuracy* simply by predicting "No Heart Disease" for almost everyone—missing *92%* of actual heart disease patients.
* *The Solution (Cost-Sensitive Learning):* By applying class_weight='balanced' and leakage-free feature scaling (StandardScaler), the model penalizes misclassifications of the minority class, boosting positive case detection by *9x* (catching *126 out of 175* high-risk patients in the test set).

---

## 📂 Dataset Summary

The dataset originates from an ongoing cardiovascular study on residents of the town of Framingham, Massachusetts (framingham.csv).

* *Initial Dataset Size:* 4,240 rows × 16 columns
* *Cleaned Dataset Size:* 3,751 rows × 15 columns (after dropping missing records and non-clinical education column)
* *Target Variable (TenYearCHD):*
  * 0 (No 10-Year CHD Risk): *3,179 patients (84.75%)*
  * 1 (10-Year CHD Risk Present): *572 patients (15.25%)*

### Selected Clinical & Behavioral Features (X)

| Feature Name | Clinical Description | Data Type |
| :--- | :--- | :--- |
| *age* | Age of the patient at the time of examination (years) | Numerical |
| *Sex_male* | Gender of the patient (1 = Male, 0 = Female) | Binary |
| *cigsPerDay* | Number of cigarettes smoked per day | Numerical |
| *totChol* | Total serum cholesterol level (mg/dL) | Numerical |
| *sysBP* | Systolic blood pressure measurement (mmHg) | Numerical |
| *glucose* | Fasting blood glucose level (mg/dL) | Numerical |

---

## ⚙️ Data Preprocessing & Pipeline

1. *Data Cleaning:*
   * Dropped the irrelevant education feature.
   * Renamed male to Sex_male for clarity.
   * Removed rows containing NaN values (dropna), leaving *3,751 complete records*.
2. *Train-Test Split:*
   * Partitioned the dataset into *70% Training (2,625 samples)* and *30% Testing (1,126 samples)* using random_state=4.
3. *Leakage-Free Feature Scaling:*
   * Fitted StandardScaler() strictly on X_train (fit_transform) and applied the learned transformation to X_test (transform) to prevent train-test data leakage.
4. *Exploratory Data Analysis (EDA):*
   * Plotted target class frequencies and percentage distributions using Seaborn and Matplotlib to highlight the 85:15 class imbalance.
5. *Model Training:*
   * *Model 1 (Baseline):* LogisticRegression(random_state=4)
   * *Model 2 (Balanced):* LogisticRegression(class_weight='balanced', random_state=4)

---

## 📊 Model Comparison: Before vs. After

| Evaluation Metric | Baseline Model (Unweighted) | Balanced Model (class_weight='balanced') | Key Clinical Takeaway |
| :--- | :---: | :---: | :--- |
| *Overall Accuracy* | 84.90% | *67.14%* | Accuracy adjusts as the model stops blindly predicting Class 0 |
| *CHD Recall (Sensitivity)* | 8.00% | *72.00%* | *+64.00% increase* in identifying true high-risk patients |
| *Sick Patients Detected (TP)* | 14 / 175 | *126 / 175* | *9x more* positive CHD cases successfully caught |
| *Sick Patients Missed (FN)* | 161 / 175 | *49 / 175* | Reduced dangerous missed diagnoses by *~70%* |

---

## 🧮 Confusion Matrix & Classification Report

### Confusion Matrix (Balanced Model on Test Set = 1,126 Patients)

| | *Predicted: No CHD (0)* | *Predicted: CHD (1)* |
| :--- | :---: | :---: |
| *Actual: No CHD (0)* | *630* (True Negatives) | *321* (False Positives) |
| *Actual: CHD (1)* | *49* (False Negatives) | *126* (True Positives) |

### Detailed Classification Metrics (Balanced Model)

| Class / Metric | Precision | Recall | F1-Score | Support (Patients) |
| :--- | :---: | :---: | :---: | :---: |
| *Class 0 (No Heart Disease)* | 0.93 | 0.66 | 0.77 | 951 |
| *Class 1 (Heart Disease Risk)* | 0.28 | *0.72* | 0.41 | 175 |
| *Macro Average* | 0.60 | 0.69 | 0.59 | 1,126 |
| *Weighted Average* | 0.83 | 0.67 | 0.72 | 1,126 |

> *Clinical Interpretation:* When the balanced model predicts a patient does *not* have heart disease (Class 0), it achieves a *93% precision rate (0.93). Simultaneously, it captures **72% (0.72)* of all patients who actually develop CHD over a 10-year period, making it a strong first-tier screening tool.

---

## 🛠️ Tech Stack

* *Programming Language:* Python 3
* *Data Manipulation:* pandas, numpy
* *Machine Learning:* scikit-learn (StandardScaler, train_test_split, LogisticRegression, accuracy_score, recall_score, confusion_matrix, classification_report)
* *Data Visualization:* matplotlib, seaborn
* *Development Environment:* Jupyter Notebook

---

## 🚀 How to Run Locally

1. *Clone this repository:*

       git clone https://github.com/singamalaomsai/Heart_Disease_Prediction_Using_Logistic_Regression.git
       cd Heart_Disease_Prediction_Using_Logistic_Regression

2. *Install the required Python libraries:*

       pip install pandas numpy scikit-learn matplotlib seaborn jupyter

3. *Launch the Jupyter Notebook:*

       jupyter notebook

4. Open the .ipynb file and run all cells sequentially (Kernel -> Restart & Run All).

---

## 👤 Author

*Singamala Omsai*
* *GitHub:* [@singamalaomsai](https://github.com/singamalaomsai)
* *Project Link:* [Heart_Disease_Prediction_Using_Logistic_Regression](https://github.com/singamalaomsai/Heart_Disease_Prediction_Using_Logistic_Regression)

---
If you found this project useful or insightful, please consider giving this repository a ⭐ on GitHub!
