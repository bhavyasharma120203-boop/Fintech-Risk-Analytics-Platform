"""
Step 3: Exploratory Data Analysis (EDA)
Generates key business charts as PNGs for the dashboard/report.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
OUT = "/home/claude/fintech_risk/outputs"
df = pd.read_csv(f"{OUT}/master_dataset.csv")

print("=== BUSINESS SUMMARY ===")
print(f"Total loans: {len(df):,}")
print(f"Total disbursed: ₹{df['loan_amount'].sum():,.0f}")
print(f"Overall default rate: {df['is_default'].mean()*100:.2f}%")
print(f"Avg credit score: {df['credit_score'].mean():.0f}")
print(f"Avg DTI ratio: {df['dti_ratio'].mean():.2f}")

fig, axes = plt.subplots(2, 3, figsize=(20, 11))

# 1. Default rate by risk band
d1 = df.groupby("risk_band")["is_default"].mean().sort_values()
sns.barplot(x=d1.index, y=d1.values*100, ax=axes[0,0], palette="Reds_r")
axes[0,0].set_title("Default Rate by Credit Risk Band")
axes[0,0].set_ylabel("Default Rate (%)")

# 2. Default rate by employment type
d2 = df.groupby("employment_type")["is_default"].mean().sort_values(ascending=False)*100
sns.barplot(x=d2.values, y=d2.index, ax=axes[0,1], palette="Blues_r")
axes[0,1].set_title("Default Rate by Employment Type")
axes[0,1].set_xlabel("Default Rate (%)")

# 3. Loan amount distribution by default status
sns.boxplot(data=df, x="is_default", y="loan_amount", ax=axes[0,2], palette="Set2")
axes[0,2].set_title("Loan Amount: Defaulters vs Non-Defaulters")
axes[0,2].set_xticklabels(["Non-Default", "Default"])

# 4. DTI ratio vs default
sns.boxplot(data=df, x="is_default", y="dti_ratio", ax=axes[1,0], palette="Set3")
axes[1,0].set_title("Debt-to-Income Ratio vs Default")
axes[1,0].set_ylim(0, 3)
axes[1,0].set_xticklabels(["Non-Default", "Default"])

# 5. Default rate by income band
d5 = df.groupby("income_band")["is_default"].mean()*100
sns.barplot(x=d5.index, y=d5.values, ax=axes[1,1], palette="Greens_r")
axes[1,1].set_title("Default Rate by Income Band")
axes[1,1].tick_params(axis='x', rotation=20)

# 6. Default rate by state (top 10)
d6 = df.groupby("state")["is_default"].mean().sort_values(ascending=False)*100
sns.barplot(x=d6.values, y=d6.index, ax=axes[1,2], palette="Oranges_r")
axes[1,2].set_title("Default Rate by State")
axes[1,2].set_xlabel("Default Rate (%)")

plt.tight_layout()
plt.savefig(f"{OUT}/eda_overview.png", dpi=120)
print(f"Saved -> {OUT}/eda_overview.png")

# Correlation heatmap of key numeric risk drivers
plt.figure(figsize=(9,7))
num_cols = ["credit_score","dti_ratio","loan_amount","interest_rate","annual_income",
            "missed_installment_rate","avg_days_late","num_bounced_payments_last_year","is_default"]
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap: Risk Drivers vs Default")
plt.tight_layout()
plt.savefig(f"{OUT}/correlation_heatmap.png", dpi=120)
print(f"Saved -> {OUT}/correlation_heatmap.png")

# Monthly origination trend & default rate over time
df["origination_month"] = pd.to_datetime(df["origination_date"]).dt.to_period("M").astype(str)
monthly = df.groupby("origination_month").agg(
    loans_originated=("loan_id","count"),
    total_disbursed=("loan_amount","sum"),
    default_rate=("is_default","mean")
).reset_index()
monthly.to_csv(f"{OUT}/monthly_trends.csv", index=False)

fig, ax1 = plt.subplots(figsize=(16,6))
ax2 = ax1.twinx()
ax1.bar(monthly["origination_month"], monthly["total_disbursed"]/1e6, color="steelblue", alpha=0.6, label="Disbursed (₹M)")
ax2.plot(monthly["origination_month"], monthly["default_rate"]*100, color="crimson", marker="o", label="Default Rate (%)")
ax1.set_ylabel("Total Disbursed (₹ Millions)")
ax2.set_ylabel("Default Rate (%)")
ax1.set_title("Monthly Loan Disbursement & Default Rate Trend")
plt.xticks(rotation=90)
fig.tight_layout()
plt.savefig(f"{OUT}/monthly_trend.png", dpi=120)
print(f"Saved -> {OUT}/monthly_trend.png")

print("\nEDA complete.")
