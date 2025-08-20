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

### 🔹 Task 1 : [Customer Segmentation]
**Goal:** 
The goal of this project is to segment customers based on annual income and spending score to identify distinct groups of shopping behavior. This segmentation helps businesses design targeted marketing strategies and improve customer relationship management.

**Steps Taken:** 
1- Data Preparation

- Cleaned the dataset, focusing on Annual Income and Spending Score.

- Standardized the features to ensure balanced clustering.

2- K-Means Clustering

- Applied the elbow method  to determine the optimal number of clusters.

- Chose 5 clusters as the best fit.

3- DBSCAN Clustering

- Applied DBSCAN with default parameters (eps=0.5, min_samples=5).

- Identified clusters and noise points.

4-Cluster Analysis

- Compared average spending scores across clusters.

- Interpreted each group’s behavior (high/low income vs. high/low spending).
  
**Resuls : (Decision Tree)**
1- K-Means (5 Clusters)

- Successfully segmented customers into 5 distinct groups:

       High income / High spending

       Low income / High spending

       High income / Low spending

      Low income / Low spending

      Average income / Average spending

Average spending scores per cluster ranged from 17 → 82, showing clear behavioral differences.

2- DBSCAN

- Cluster 0: Customers with lower income and lower spending (avg spending ≈ 43).

- Cluster 1: Smaller group with higher spending (avg spending ≈ 83).

- Noise points (-1): Customers with irregular patterns (avg spending ≈ 47).
---
### 🔹 Task 2: [Loan Approval Prediction]  
**Goal:** 
The goal of this project is to build a machine learning model that predicts whether a loan application will be approved or rejected, based on applicant and financial attributes.
This project simulates a real-world financial application where accurate predictions can help banks and institutions reduce risks and improve decision-making.

**Steps Taken:**  
1. Data Loading & Inspection :

- Loaded loan_approval_dataset.csv into Pandas.

- Checked data shape, column info, and missing values.

- Performed duplication checks and standardized column names.

2. Data Cleaning & Preprocessing :

- Fixed categorical inconsistencies (e.g., Self_Employed mapped to boolean values).

- Handled missing values and formatted data.

- Outliers detected and capped using the IQR method.

- Applied Label Encoding for categorical variables.

- Applied StandardScaler to normalize numerical features.

3. Exploratory Data Analysis (EDA) :

- Univariate Analysis: Plotted distributions for numerical (income, loan amount, CIBIL score, etc.) and categorical features.

- Bivariate Analysis: Explored relationships (e.g., income vs. loan amount with loan status).

- Correlation Analysis: Heatmap of numerical features using Spearman correlation.

4. Data Splitting & Imbalance Handling :

- Split dataset into train (80%) and test (20%) sets.

- Addressed class imbalance with SMOTE (Synthetic Minority Oversampling Technique) to avoid biased predictions.

5. Model Training :

- Trained two models for comparison:

  Logistic Regression

  Decision Tree Classifier

6. Model Evaluation :

- Evaluated with confusion matrix, precision, recall, F1-score, and accuracy.

- Visualized results with seaborn heatmaps and sklearn’s confusion matrix display.

7. Model Comparison :

- Logistic Regression: Baseline model, worked but lower recall/precision.

- Decision Tree: Outperformed Logistic Regression in recall and precision, making it the better choice for this dataset. 

**Resuls : (Decision Tree)**
- F1-Score: 96.6%  
- Precision: 96.3%  
- Recall: 96.9%  

-----
### 🔹 Task 3: Sales Forecasting Description
**Goal:** 
The goal of this project is to predict weekly sales for Walmart stores using historical sales, store information, economic indicators, promotions, and seasonal effects.
Accurate forecasting can help optimize inventory, staffing, and promotional strategies.
**Steps Taken:**  
1- Data Preparation & Merging

- Combined train, features, and store data into a single DataFrame.

- Resolved missing values in MarkDown columns and consolidated duplicate IsHoliday columns.

2- Feature Engineering

- Extracted time-based features: Year, Month, DayOfWeek.

- Created lag feature: Weekly_Sales_Lag_1.

- Added rolling averages (Weekly_Sales_Roll4, Weekly_Sales_Roll12) for trend smoothing.

- Performed seasonal decomposition for trend/seasonality analysis.

- Encoded categorical variable Type using one-hot encoding.

3-Handling Missing Values

- Filled missing MarkDown values and lagged sales with 0.

- Consolidated duplicate IsHoliday into a single binary column.

4- Model Training

- Split dataset by time (pre-2012 = training, post-2012 = testing).

- Trained three regression models:

         Linear Regression (baseline)

         XGBoost (gradient boosting)

         LightGBM (optimized gradient boosting)

4- Model Evaluation

Compared models using MAE (Mean Absolute Error) and RMSE (Root Mean Squared Error).

Conducted seasonal decomposition visualizations for deeper insights.

**Results :**
- Linear Regression - MAE: 2484.09, RMSE: 4406.50
- XGBoost - MAE: 1811.48, RMSE: 3990.33
- LightGBM - MAE: 1805.68, RMSE: 3979.01
## 📊 Dataset Section

### 📂 Dataset Sources
- **Task 1 Dataset**: https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python
- **Task 2 Dataset**: https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset/data
- **Task 3 Dataset**: https://www.kaggle.com/datasets/aslanahmedov/walmart-sales-forecast/data
