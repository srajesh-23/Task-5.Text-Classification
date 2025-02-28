# Consumer Complaint Classification using Machine Learning

This project focuses on **text classification** of consumer complaints into predefined categories using **machine learning**. The dataset contains consumer complaints related to financial products and services, and the goal is to classify them into four categories:

- **0** - Credit Reporting, Repair, or Other  
- **1** - Debt Collection  
- **2** - Consumer Loan  
- **3** - Mortgage  

---

## Project Workflow

### 1. **Exploratory Data Analysis (EDA) & Feature Engineering**
- Analyzed class distribution and balanced the dataset using **SMOTE**.
- Converted text data into numerical representations using **TF-IDF Vectorization**.

### 2️.**Text Pre-Processing**
- Tokenization, stopword removal, and stemming.
- TF-IDF vectorization to convert text into numerical format.

### 3️. **Model Selection**
We experimented with multiple classification models:
- **Logistic Regression**
- **Random Forest (Slow Performance)**
- **XGBoost (Best Model with 95% Accuracy)**

### 4️. **Hyperparameter Tuning**
- Used `GridSearchCV` to optimize XGBoost parameters.
- Best parameters:
  - `n_estimators`: 200
  - `max_depth`: 6
  - `learning_rate`: 0.1

### 5️. **Model Evaluation**
- Accuracy & F1-Score were used as evaluation metrics.
- **XGBoost outperformed other models** with an accuracy of **95.12%**.

---

## **Results Summary**

| Model               | Accuracy  | F1-Score |
|---------------------|----------|----------|
| Logistic Regression | 61.47%   | 0.62     |
| XGBoost (Tuned)    | **95.12%** | **0.95** |

🔹 **Final Model:** XGBoost  
🔹 **Best Accuracy Achieved:** 95.12%  
🔹 **Key Challenge:** Correcting class label issues (`3` mistakenly introduced in dataset).  

---

## **How to Run the Project?**
### 1️. Install Dependencies
```bash
pip install pandas numpy sklearn xgboost imbalanced-learn
