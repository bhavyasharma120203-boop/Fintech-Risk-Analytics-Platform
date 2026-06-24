# Fintech Risk Analytics Platform

## Overview

The Fintech Risk Analytics Platform is an end-to-end data analytics and machine learning project designed to assess loan default risk, identify high-risk customers, forecast portfolio performance, and support data-driven lending decisions.

This project simulates a real-world fintech lending environment where financial institutions must evaluate borrower risk, reduce loan defaults, and improve profitability through advanced analytics.

By integrating Python, SQL, Machine Learning, Forecasting, and Business Intelligence techniques, the project demonstrates the complete analytics lifecycle from raw data processing to actionable business insights.

---

## Business Problem

Loan defaults are a major challenge for banks and fintech companies, leading to substantial financial losses and increased credit risk.

The objective of this project is to answer key business questions:

* Which customers are most likely to default on their loans?
* What factors contribute most to default risk?
* How can customers be segmented into meaningful risk categories?
* How are loan disbursements and default rates changing over time?
* What future default trends can lenders expect?

The insights generated can help financial institutions improve lending strategies, optimize risk management, and enhance customer targeting.

---

## Dataset

The project uses synthetic fintech customer and loan data representing real-world lending scenarios.

The dataset includes:

### Customer Information

* Age
* Annual Income
* Employment Type
* Customer Tenure
* Credit Score

### Loan Information

* Loan Amount
* Interest Rate
* Loan Tenure
* Existing Debt

### Financial Behavior Metrics

* Monthly Inflow
* Monthly Outflow
* Monthly Balance
* Debt-to-Income Ratio
* Loan-to-Income Ratio

### Target Variable

* Loan Default Status (Default / Non-Default)

---

## Tools & Technologies

### Programming & Analytics

* Python
* SQL

### Data Processing

* Pandas
* NumPy

### Data Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-Learn
* Logistic Regression
* Random Forest
* K-Means Clustering

### Forecasting

* Prophet

### Business Intelligence

* Power BI

---

## Project Workflow

### 1. Data Generation & Preparation

* Generated customer, loan, repayment, and transaction datasets.
* Performed data cleaning and preprocessing.
* Handled missing values and transformed variables.
* Created analytical features for risk assessment.

### 2. Exploratory Data Analysis (EDA)

* Analyzed customer demographics and loan characteristics.
* Identified default patterns across income groups and employment categories.
* Evaluated debt-to-income and loan-to-income relationships.

### 3. Customer Risk Segmentation

* Applied K-Means Clustering to group customers into risk categories.
* Identified:

  * Prime Customers
  * Stable Customers
  * Watchlist Customers
  * High-Risk Customers

### 4. Loan Default Prediction

* Developed classification models to predict customer default risk.
* Compared Logistic Regression and Random Forest performance.
* Evaluated models using:

  * Accuracy
  * Precision
  * Recall
  * ROC-AUC

### 5. Forecasting

* Built forecasting models using Prophet.
* Predicted future default rates and portfolio trends.
* Identified potential future risk patterns.

### 6. Business Intelligence Preparation

* Generated KPI-ready datasets.
* Prepared outputs for Power BI dashboard development.

---

## Key Findings

### Risk Drivers

* Customers with lower credit scores exhibited significantly higher default rates.
* Higher debt-to-income ratios were strongly associated with loan defaults.
* Unemployed and self-employed customers demonstrated elevated risk levels.
* Lower-income customer segments experienced higher default probabilities.

### Customer Segmentation

* Four distinct customer risk groups were identified.
* High-risk customers generally exhibited lower credit scores and higher financial leverage.
* Prime customers maintained stronger repayment profiles and lower risk exposure.

### Predictive Modeling

* Logistic Regression achieved an ROC-AUC score of approximately 0.81.
* Credit score emerged as the most important predictor of loan default.
* Debt-to-income ratio was identified as the second most influential factor.

### Forecasting

* Future default rates are expected to remain relatively stable.
* Portfolio monitoring remains important to prevent emerging risk concentrations.

---

## Project Visualizations

### Risk Driver Analysis

![Risk Driver Analysis](images/risk_driver_analysis.jpg)

---

### Customer Risk Segmentation

![Customer Risk Segmentation](images/customer_segments.jpg)

---

### Loan Default Prediction Model

![Default Prediction Model](images/default_prediction_model.jpg)

---

### Default Rate Forecast

![Default Rate Forecast](images/default_rate_forecast.jpg)

---

### Monthly Loan Disbursement & Default Trend

![Monthly Loan Disbursement Trend](images/monthly_disbursement_trend.jpg)

---

## Repository Structure

```text
Fintech-Risk-Analytics-Platform

│
├── data/
│
├── 01_generate_data.py
├── 02_clean_merge_sql.py
├── 03_eda.py
├── 04_segmentation_kmeans.py
├── 05_default_prediction_model.py
├── 06_forecasting_prophet.py
├── 07_powerbi_export.py
│
├── images/
│
└── README.md
```

## Skills Demonstrated

* Data Cleaning & Preprocessing
* Exploratory Data Analysis
* SQL Querying
* Feature Engineering
* Customer Segmentation
* Machine Learning
* Predictive Analytics
* Forecasting
* Data Visualization
* Business Intelligence
* Business Problem Solving

---

## Business Impact

This project demonstrates how analytics can be leveraged to:

* Improve lending decisions.
* Reduce loan default risk.
* Identify high-risk borrowers early.
* Optimize portfolio management.
* Support data-driven business strategies.

---

## Future Enhancements

* Integration with real-world lending datasets.
* Deployment of interactive Power BI dashboards.
* Implementation of XGBoost and Gradient Boosting models.
* Development of real-time risk monitoring systems.
* Automated credit scoring framework.

---

### Author

Bhavya Sharma

Master's in Economics | Data Analytics | SQL | Power BI | Python | Machine Learning
