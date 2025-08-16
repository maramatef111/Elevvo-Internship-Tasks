# Elevvo-Internship-Tasks


This repository contains the tasks completed during my internship at **Elevoo**.  
Each task demonstrates practical applications of **Machine Learning** concepts.  

---

## 📂 Repository Structure
- `Task1/` → Task 1 implementation
- `Task2/` → Task 2 implementation
- `Task3/` → Task 3 implementation
- `datasets/` → Datasets used for tasks
  
---
## 📝 Tasks Overview

### 🔹 Task 1: [Loan Approval Prediction]  
**Goal:** 
The goal of this project is to build a machine learning model that predicts whether a loan application will be approved or rejected, based on applicant and financial attributes.
This project simulates a real-world financial application where accurate predictions can help banks and institutions reduce risks and improve decision-making.
**Steps Taken:**  
1. Data Loading & Inspection

Loaded loan_approval_dataset.csv into Pandas.

Checked data shape, column info, and missing values.

Performed duplication checks and standardized column names.

2. Data Cleaning & Preprocessing

Fixed categorical inconsistencies (e.g., Self_Employed mapped to boolean values).

Handled missing values and formatted data.

Outliers detected and capped using the IQR method.

Applied Label Encoding for categorical variables.

Applied StandardScaler to normalize numerical features.

3. Exploratory Data Analysis (EDA)

Univariate Analysis: Plotted distributions for numerical (income, loan amount, CIBIL score, etc.) and categorical features.

Bivariate Analysis: Explored relationships (e.g., income vs. loan amount with loan status).

Correlation Analysis: Heatmap of numerical features using Spearman correlation.

4. Data Splitting & Imbalance Handling

Split dataset into train (80%) and test (20%) sets.

Addressed class imbalance with SMOTE (Synthetic Minority Oversampling Technique) to avoid biased predictions.

5. Model Training

Trained two models for comparison:

Logistic Regression

Decision Tree Classifier

6. Model Evaluation

Evaluated with confusion matrix, precision, recall, F1-score, and accuracy.

Visualized results with seaborn heatmaps and sklearn’s confusion matrix display.

7. Model Comparison

Logistic Regression: Baseline model, worked but lower recall/precision.

Decision Tree: Outperformed Logistic Regression in recall and precision, making it the better choice for this dataset. 

**Results:**  
- F1-Score: 96.6%  
- Precision: 96.3%  
- Recall: 96.9%  

-----
## 📊 Dataset Section

### 📂 Dataset Sources
- **Task 1 Dataset**: https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset/data
- **Task 2 Dataset**: [Name or link]  
- **Task 3 Dataset**: [Name or link] 
