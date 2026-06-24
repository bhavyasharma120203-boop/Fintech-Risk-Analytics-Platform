"""
Step 6: Time-Series Forecasting
Forecasts future monthly loan disbursement volume and default rate using Prophet.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from prophet import Prophet

OUT = "/home/claude/fintech_risk/outputs"
df = pd.read_csv(f"{OUT}/master_dataset.csv")
df["origination_date"] = pd.to_datetime(df["origination_date"])

monthly = df.groupby(pd.Grouper(key="origination_date", freq="MS")).agg(
    loans_originated=("loan_id","count"),
    total_disbursed=("loan_amount","sum"),
    default_rate=("is_default","mean")
).reset_index()
monthly = monthly[monthly["origination_date"] < monthly["origination_date"].max()]  # drop partial last month

print("Monthly series length:", len(monthly))

# --- Forecast 1: Total Disbursement Volume ---
disb = monthly.rename(columns={"origination_date":"ds", "total_disbursed":"y"})[["ds","y"]]
m1 = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
             changepoint_prior_scale=0.1)
m1.fit(disb)
future1 = m1.make_future_dataframe(periods=6, freq="MS")
fc1 = m1.predict(future1)

fig1 = m1.plot(fc1)
plt.title("Forecast: Monthly Loan Disbursement Volume (₹)")
plt.tight_layout()
fig1.savefig(f"{OUT}/forecast_disbursement.png", dpi=120)

# --- Forecast 2: Default Rate ---
defr = monthly.rename(columns={"origination_date":"ds", "default_rate":"y"})[["ds","y"]]
m2 = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
             changepoint_prior_scale=0.1)
m2.fit(defr)
future2 = m2.make_future_dataframe(periods=6, freq="MS")
fc2 = m2.predict(future2)

fig2 = m2.plot(fc2)
plt.title("Forecast: Monthly Default Rate")
plt.tight_layout()
fig2.savefig(f"{OUT}/forecast_default_rate.png", dpi=120)

# Save forecast tables
fc1[["ds","yhat","yhat_lower","yhat_upper"]].tail(12).to_csv(f"{OUT}/forecast_disbursement_table.csv", index=False)
fc2[["ds","yhat","yhat_lower","yhat_upper"]].tail(12).to_csv(f"{OUT}/forecast_default_rate_table.csv", index=False)

print("\n=== Next 6-month disbursement forecast ===")
print(fc1[["ds","yhat"]].tail(6).to_string(index=False))
print("\n=== Next 6-month default rate forecast ===")
print(fc2[["ds","yhat"]].tail(6).to_string(index=False))

print(f"\nSaved -> {OUT}/forecast_disbursement.png, forecast_default_rate.png")
