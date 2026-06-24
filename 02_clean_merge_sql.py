"""
Fintech Customer Risk & Loan Default Analytics Platform
Step 2: Data Cleaning, SQL Merging & Unified Analytical Dataset

- Loads raw CSVs into a SQLite database (simulates a real DB / data warehouse layer)
- Uses SQL to join customers + loans + transaction summary + repayment aggregates
- Uses Pandas to handle missing values, outliers, and feature engineering
- Outputs a single clean analytical table: master_dataset.csv
"""

import pandas as pd
import numpy as np
import sqlite3

DATA = "/home/claude/fintech_risk/data"
OUT = "/home/claude/fintech_risk/outputs"

# ------------------------------------------------------------------
# 1. LOAD RAW DATA
# ------------------------------------------------------------------
customers = pd.read_csv(f"{DATA}/customers.csv")
loans = pd.read_csv(f"{DATA}/loans.csv")
repayments = pd.read_csv(f"{DATA}/repayments.csv")
transactions = pd.read_csv(f"{DATA}/transactions.csv")

print("Raw shapes:", customers.shape, loans.shape, repayments.shape, transactions.shape)

# ------------------------------------------------------------------
# 2. LOAD INTO SQLITE (simulates a SQL warehouse) & RUN SQL AGGREGATIONS
# ------------------------------------------------------------------
conn = sqlite3.connect(":memory:")
customers.to_sql("customers", conn, index=False)
loans.to_sql("loans", conn, index=False)
repayments.to_sql("repayments", conn, index=False)
transactions.to_sql("transactions", conn, index=False)

# SQL: repayment-level behavioral aggregates per loan
repayment_agg_sql = """
SELECT
    loan_id,
    COUNT(*)                                            AS total_installments_due,
    SUM(CASE WHEN payment_status = 'Missed' THEN 1 ELSE 0 END) AS missed_installments,
    ROUND(AVG(days_late), 2)                            AS avg_days_late,
    MAX(days_late)                                       AS max_days_late
FROM repayments
GROUP BY loan_id
"""
repay_agg = pd.read_sql(repayment_agg_sql, conn)

# SQL: the core analytical join (customers + loans + transactions)
join_sql = """
SELECT
    l.loan_id,
    l.customer_id,
    c.age,
    c.gender,
    c.state,
    c.city_tier,
    c.employment_type,
    c.education,
    c.annual_income,
    c.credit_score,
    c.existing_loans_count,
    c.existing_debt,
    c.account_open_date,
    l.loan_amount,
    l.interest_rate,
    l.tenure_months,
    l.emi_amount,
    l.loan_purpose,
    l.origination_date,
    l.dti_ratio,
    l.loan_status,
    l.is_default,
    t.avg_monthly_balance,
    t.avg_monthly_inflow,
    t.avg_monthly_outflow,
    t.num_bounced_payments_last_year,
    t.num_active_credit_lines
FROM loans l
LEFT JOIN customers c ON l.customer_id = c.customer_id
LEFT JOIN transactions t ON l.customer_id = t.customer_id
"""
master = pd.read_sql(join_sql, conn)
master = master.merge(repay_agg, on="loan_id", how="left")
conn.close()

print("Joined master shape:", master.shape)

# ------------------------------------------------------------------
# 3. DATA CLEANING (Pandas)
# ------------------------------------------------------------------

# --- Missing value handling ---
missing_before = master.isna().sum()
print("\nMissing values before cleaning:\n", missing_before[missing_before > 0])

# Numeric: median imputation, grouped by employment_type where sensible
master["annual_income"] = master.groupby("employment_type")["annual_income"].transform(
    lambda x: x.fillna(x.median())
)
master["credit_score"] = master["credit_score"].fillna(master["credit_score"].median())
master["education"] = master["education"].fillna("Unknown")
master["missed_installments"] = master["missed_installments"].fillna(0)
master["avg_days_late"] = master["avg_days_late"].fillna(0)
master["max_days_late"] = master["max_days_late"].fillna(0)
master["total_installments_due"] = master["total_installments_due"].fillna(0)

# --- Outlier handling: cap extreme values using IQR capping (not deletion, to preserve data) ---
def cap_outliers(s):
    q1, q3 = s.quantile([0.01, 0.99])
    return s.clip(q1, q3)

for col in ["annual_income", "loan_amount", "dti_ratio", "avg_monthly_balance"]:
    master[col] = cap_outliers(master[col])

# --- Type fixes ---
master["origination_date"] = pd.to_datetime(master["origination_date"])
master["account_open_date"] = pd.to_datetime(master["account_open_date"])

# --- Duplicate check ---
dupes = master.duplicated(subset=["loan_id"]).sum()
print(f"Duplicate loan_id rows: {dupes}")
master = master.drop_duplicates(subset=["loan_id"])

# ------------------------------------------------------------------
# 4. FEATURE ENGINEERING
# ------------------------------------------------------------------
master["loan_to_income_ratio"] = np.round(master["loan_amount"] / master["annual_income"], 3)
master["missed_installment_rate"] = np.where(
    master["total_installments_due"] > 0,
    master["missed_installments"] / master["total_installments_due"],
    0,
)
master["customer_tenure_years"] = np.round(
    (master["origination_date"] - master["account_open_date"]).dt.days / 365, 2
)
master["income_band"] = pd.cut(
    master["annual_income"],
    bins=[0, 300000, 600000, 1200000, np.inf],
    labels=["Low (<3L)", "Mid (3-6L)", "High (6-12L)", "Very High (12L+)"]
)
master["risk_band"] = pd.cut(
    master["credit_score"],
    bins=[0, 600, 700, 750, 900],
    labels=["High Risk", "Moderate Risk", "Low Risk", "Prime"]
)

# ------------------------------------------------------------------
# 5. SAVE OUTPUTS
# ------------------------------------------------------------------
master.to_csv(f"{OUT}/master_dataset.csv", index=False)
print("\nFinal cleaned master dataset shape:", master.shape)
print("Missing values after cleaning:", master.isna().sum().sum())
print(f"\nSaved -> {OUT}/master_dataset.csv")
