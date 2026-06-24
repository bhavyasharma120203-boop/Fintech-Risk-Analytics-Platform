"""
Fintech Customer Risk & Loan Default Analytics Platform
Step 1: Synthetic Data Generation

Generates a realistic multi-table fintech dataset:
 - customers.csv      (demographics, KYC, credit bureau-style features)
 - loans.csv           (loan applications/originations)
 - repayments.csv       (monthly installment-level repayment history)
 - transactions.csv     (bank-account transaction behavior, pre-loan)

Total scale: ~50,000 customers, ~70,000 loans, ~700,000+ repayment records.
"""

import numpy as np
import pandas as pd
from faker import Faker
import random

np.random.seed(42)
random.seed(42)
fake = Faker()
Faker.seed(42)

N_CUSTOMERS = 50_000
N_LOANS = 70_000  # some customers have multiple loans

# ----------------------------------------------------------------------
# 1. CUSTOMERS TABLE
# ----------------------------------------------------------------------
print("Generating customers...")

states = ["Maharashtra", "Delhi", "Karnataka", "Tamil Nadu", "Uttar Pradesh",
          "West Bengal", "Gujarat", "Rajasthan", "Telangana", "Punjab"]
employment_types = ["Salaried", "Self-Employed", "Business Owner", "Unemployed", "Retired"]
education_levels = ["High School", "Bachelor's", "Master's", "PhD", "Diploma"]

customer_ids = [f"CUST{100000+i}" for i in range(N_CUSTOMERS)]

ages = np.clip(np.random.normal(38, 11, N_CUSTOMERS), 21, 70).astype(int)
annual_income = np.round(np.random.lognormal(mean=12.6, sigma=0.55, size=N_CUSTOMERS), -2)
annual_income = np.clip(annual_income, 120000, 8_000_000)

credit_score = np.clip(np.random.normal(680, 95, N_CUSTOMERS), 300, 900).astype(int)
existing_loans_count = np.random.poisson(0.8, N_CUSTOMERS)
existing_debt = np.round(np.random.exponential(scale=80000, size=N_CUSTOMERS) * (existing_loans_count > 0), -2)

customers = pd.DataFrame({
    "customer_id": customer_ids,
    "name": [fake.name() for _ in range(N_CUSTOMERS)],
    "age": ages,
    "gender": np.random.choice(["Male", "Female", "Other"], N_CUSTOMERS, p=[0.55, 0.43, 0.02]),
    "state": np.random.choice(states, N_CUSTOMERS),
    "employment_type": np.random.choice(employment_types, N_CUSTOMERS, p=[0.45, 0.25, 0.15, 0.10, 0.05]),
    "education": np.random.choice(education_levels, N_CUSTOMERS, p=[0.25, 0.40, 0.20, 0.05, 0.10]),
    "annual_income": annual_income,
    "credit_score": credit_score,
    "existing_loans_count": existing_loans_count,
    "existing_debt": existing_debt,
    "account_open_date": [fake.date_between(start_date="-10y", end_date="-1y") for _ in range(N_CUSTOMERS)],
    "city_tier": np.random.choice(["Tier 1", "Tier 2", "Tier 3"], N_CUSTOMERS, p=[0.4, 0.35, 0.25]),
})

# Inject some realistic missingness
for col, frac in [("annual_income", 0.02), ("credit_score", 0.015), ("education", 0.01)]:
    idx = customers.sample(frac=frac, random_state=1).index
    customers.loc[idx, col] = np.nan

customers.to_csv("/home/claude/fintech_risk/data/customers.csv", index=False)
print(f"customers.csv -> {customers.shape}")

# ----------------------------------------------------------------------
# 2. LOANS TABLE
# ----------------------------------------------------------------------
print("Generating loans...")

loan_customer_ids = np.random.choice(customer_ids, N_LOANS,
                                      p=None, replace=True)
# bias so most customers have 1 loan, some have 2-3
loan_purposes = ["Personal", "Home Renovation", "Auto", "Education", "Medical", "Business", "Debt Consolidation"]
loan_ids = [f"LOAN{200000+i}" for i in range(N_LOANS)]

cust_lookup = customers.set_index("customer_id")

loan_amount = np.round(np.random.lognormal(mean=11.2, sigma=0.7, size=N_LOANS), -2)
loan_amount = np.clip(loan_amount, 10000, 2_000_000)

interest_rate = np.round(np.random.normal(13.5, 3.2, N_LOANS), 2)
interest_rate = np.clip(interest_rate, 6, 28)

tenure_months = np.random.choice([6, 12, 18, 24, 36, 48, 60], N_LOANS,
                                  p=[0.05, 0.20, 0.15, 0.20, 0.20, 0.12, 0.08])

origination_date = [fake.date_between(start_date="-4y", end_date="-2m") for _ in range(N_LOANS)]

loans = pd.DataFrame({
    "loan_id": loan_ids,
    "customer_id": loan_customer_ids,
    "loan_amount": loan_amount,
    "interest_rate": interest_rate,
    "tenure_months": tenure_months,
    "loan_purpose": np.random.choice(loan_purposes, N_LOANS,
                                      p=[0.30, 0.12, 0.13, 0.10, 0.10, 0.15, 0.10]),
    "origination_date": origination_date,
})

# EMI calculation
r = loans["interest_rate"] / 1200
n = loans["tenure_months"]
P = loans["loan_amount"]
loans["emi_amount"] = np.round(P * r * (1 + r) ** n / ((1 + r) ** n - 1), 2)

# Merge customer risk drivers to simulate a REALISTIC default probability
loans = loans.merge(customers[["customer_id", "credit_score", "annual_income",
                                "existing_debt", "employment_type", "age"]],
                     on="customer_id", how="left")

# Debt-to-income ratio (key risk driver)
loans["dti_ratio"] = np.round(
    (loans["existing_debt"].fillna(0) + loans["emi_amount"] * 12) / loans["annual_income"].fillna(loans["annual_income"].median()), 3
)

# ---- Simulate default probability via logistic function of real drivers ----
emp_risk = loans["employment_type"].map({
    "Salaried": -0.3, "Self-Employed": 0.1, "Business Owner": 0.0,
    "Unemployed": 0.9, "Retired": -0.1
}).fillna(0)

z = (
    -2.6                                              # base intercept -> ~8-12% default rate
    - 0.012 * (loans["credit_score"].fillna(680) - 650)
    + 1.6 * loans["dti_ratio"].clip(0, 3)
    + emp_risk
    + 0.02 * (loans["interest_rate"] - 13)
    - 0.008 * (loans["age"] - 38)
    + np.random.normal(0, 0.55, N_LOANS)   # noise
)
default_prob = 1 / (1 + np.exp(-z))
loans["is_default"] = (np.random.rand(N_LOANS) < default_prob).astype(int)

# Loan status reflects default / active / closed
def assign_status(row):
    if row["is_default"] == 1:
        return "Defaulted"
    return np.random.choice(["Closed - Paid", "Active"], p=[0.55, 0.45])

loans["loan_status"] = loans.apply(assign_status, axis=1)

loans_out = loans[["loan_id", "customer_id", "loan_amount", "interest_rate",
                    "tenure_months", "emi_amount", "loan_purpose",
                    "origination_date", "dti_ratio", "loan_status", "is_default"]]
loans_out.to_csv("/home/claude/fintech_risk/data/loans.csv", index=False)
print(f"loans.csv -> {loans_out.shape}, default rate = {loans_out['is_default'].mean():.3f}")

# ----------------------------------------------------------------------
# 3. REPAYMENTS TABLE (installment-level history)
# ----------------------------------------------------------------------
print("Generating repayments (this creates the bulk of records)...")

repayment_rows = []
for _, loan in loans_out.iterrows():
    n_installments = loan["tenure_months"]
    default_flag = loan["is_default"]
    due_date = pd.to_datetime(loan["origination_date"])
    missed_streak = 0
    for i in range(1, n_installments + 1):
        due_date_i = due_date + pd.DateOffset(months=i)
        if due_date_i > pd.Timestamp("2026-06-24"):
            break  # don't generate future installments
        # simulate late/missed payments more often as default approaches
        if default_flag:
            late_prob = min(0.15 + i * 0.02, 0.85)
        else:
            late_prob = 0.05
        is_late = np.random.rand() < late_prob
        days_late = int(np.random.exponential(10)) if is_late else 0
        paid = not (default_flag and i > n_installments * 0.6 and np.random.rand() < 0.5)
        repayment_rows.append((
            loan["loan_id"], i, due_date_i.date(), loan["emi_amount"],
            "Paid" if paid else "Missed", days_late
        ))

repayments = pd.DataFrame(repayment_rows, columns=[
    "loan_id", "installment_no", "due_date", "emi_amount", "payment_status", "days_late"
])
repayments.to_csv("/home/claude/fintech_risk/data/repayments.csv", index=False)
print(f"repayments.csv -> {repayments.shape}")

# ----------------------------------------------------------------------
# 4. TRANSACTIONS TABLE (pre-loan bank behavior signal)
# ----------------------------------------------------------------------
print("Generating account transaction summary...")

txn_summary = pd.DataFrame({
    "customer_id": customer_ids,
    "avg_monthly_balance": np.round(np.random.lognormal(9.5, 1.0, N_CUSTOMERS), -2),
    "avg_monthly_inflow": np.round(np.random.lognormal(10.2, 0.6, N_CUSTOMERS), -2),
    "avg_monthly_outflow": np.round(np.random.lognormal(10.1, 0.6, N_CUSTOMERS), -2),
    "num_bounced_payments_last_year": np.random.poisson(0.4, N_CUSTOMERS),
    "num_active_credit_lines": np.random.poisson(1.5, N_CUSTOMERS),
})
txn_summary.to_csv("/home/claude/fintech_risk/data/transactions.csv", index=False)
print(f"transactions.csv -> {txn_summary.shape}")

print("\nAll raw tables generated in /home/claude/fintech_risk/data/")
total_records = len(customers) + len(loans_out) + len(repayments) + len(txn_summary)
print(f"TOTAL RECORDS ACROSS TABLES: {total_records:,}")
