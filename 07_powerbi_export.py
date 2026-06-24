"""
Step 7: Power BI Export Layer
Prepares a clean star-schema-style export (fact + dimension tables) ready to import
directly into Power BI, plus pre-aggregated tables so the dashboard loads fast.
"""
import pandas as pd
import numpy as np

OUT = "/home/claude/fintech_risk/outputs"
df = pd.read_csv(f"{OUT}/master_dataset.csv")
risk_scores = pd.read_csv(f"{OUT}/loan_risk_scores.csv")
segments = pd.read_csv(f"{OUT}/customer_segments.csv")

df = df.merge(risk_scores[["loan_id","default_probability","predicted_risk_tier"]], on="loan_id", how="left")

# ---- FACT TABLE: Loans ----
fact_loans = df[[
    "loan_id","customer_id","origination_date","loan_amount","interest_rate",
    "tenure_months","emi_amount","dti_ratio","loan_to_income_ratio",
    "missed_installment_rate","is_default","default_probability","predicted_risk_tier",
    "loan_status","loan_purpose"
]]
fact_loans.to_csv(f"{OUT}/powerbi_fact_loans.csv", index=False)

# ---- DIMENSION TABLE: Customers (with segment) ----
dim_customers = df[[
    "customer_id","age","gender","state","city_tier","employment_type",
    "education","annual_income","credit_score","income_band","risk_band"
]].drop_duplicates(subset="customer_id")
dim_customers = dim_customers.merge(segments[["customer_id","segment"]], on="customer_id", how="left")
dim_customers.to_csv(f"{OUT}/powerbi_dim_customers.csv", index=False)

# ---- DIMENSION TABLE: Date ----
dates = pd.to_datetime(df["origination_date"]).drop_duplicates().sort_values()
dim_date = pd.DataFrame({"date": dates})
dim_date["year"] = dim_date["date"].dt.year
dim_date["month"] = dim_date["date"].dt.month
dim_date["month_name"] = dim_date["date"].dt.strftime("%b")
dim_date["quarter"] = dim_date["date"].dt.quarter
dim_date["year_month"] = dim_date["date"].dt.strftime("%Y-%m")
dim_date.to_csv(f"{OUT}/powerbi_dim_date.csv", index=False)

print("Power BI export tables created:")
print(f" - powerbi_fact_loans.csv      {fact_loans.shape}")
print(f" - powerbi_dim_customers.csv   {dim_customers.shape}")
print(f" - powerbi_dim_date.csv        {dim_date.shape}")
