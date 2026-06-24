# Fintech Customer Risk & Loan Default Analytics Platform

End-to-end analytics project covering the full lifecycle: data engineering → EDA →
customer risk segmentation → machine learning → time-series forecasting → BI dashboard.

## Dataset
Synthetic but realistic fintech lending dataset (Faker + statistically calibrated risk logic):
- **customers.csv** — 50,000 customers (demographics, income, credit score, KYC)
- **loans.csv** — 70,000 loan accounts (amount, rate, tenure, purpose, default flag)
- **repayments.csv** — 1.2M+ installment-level repayment records
- **transactions.csv** — 50,000 bank-behavior summaries (balances, bounced payments, credit lines)

Overall portfolio default rate: **15.5%** (realistic for unsecured/personal lending).

## Pipeline (run in order)
| Script | Purpose |
|---|---|
| `01_generate_data.py` | Synthetic multi-table data generation |
| `02_clean_merge_sql.py` | SQL (SQLite) joins + Pandas cleaning → `master_dataset.csv` |
| `03_eda.py` | EDA — default rate by risk band, employment, income, state, geography, trends |
| `04_segmentation_kmeans.py` | Customer risk segmentation (Prime / Stable / Watchlist / High-Risk) via K-Means |
| `05_default_prediction_model.py` | Logistic Regression + Random Forest default prediction, AUC ~0.81 |
| `06_forecasting_prophet.py` | Prophet forecasts for disbursement volume & default rate (6-month horizon) |
| `07_powerbi_export.py` | Star-schema export (fact_loans, dim_customers, dim_date) for Power BI |

## Key Findings
- **Credit score and DTI ratio are the dominant default drivers** (confirmed via both correlation
  analysis and Random Forest feature importance) — consistent with real-world credit risk theory.
- Customers segment cleanly into 4 risk tiers: **Prime (5.7% default)**, **Stable (24%)**,
  **Watchlist (35%)**, **High-Risk (100%, by construction of the worst cluster)**.
- Unemployed applicants default at roughly **2x** the rate of salaried applicants.
- The default prediction model (Logistic Regression, AUC = 0.814) outperformed Random Forest
  slightly while staying interpretable — important for regulatory/credit-decision use cases.
- 6-month forecast shows default rate trending from ~14.5% to ~16% — an early warning signal
  for portfolio risk management.

## Outputs (in `/outputs`)
- `master_dataset.csv` — cleaned, feature-engineered analytical dataset
- `eda_overview.png`, `correlation_heatmap.png`, `monthly_trend.png` — EDA visuals
- `customer_segments.csv`, `segments_scatter.png`, `segment_summary.csv` — risk segmentation
- `model_evaluation.png`, `model_comparison.csv`, `loan_risk_scores.csv` — ML model results
- `forecast_disbursement.png`, `forecast_default_rate.png` + forecast tables — time-series forecasts
- `powerbi_fact_loans.csv`, `powerbi_dim_customers.csv`, `powerbi_dim_date.csv` — BI-ready star schema
- `POWERBI_BUILD_GUIDE.md` — DAX measures + dashboard page layout guide
- `/models/default_prediction_model.pkl` — trained, serialized model

## Why this matters (role relevance)
This mirrors real credit-risk / fintech analytics work: cleaning messy multi-source data,
quantifying risk drivers, building an underwriting-grade prediction model, forecasting portfolio
health, and packaging it for business stakeholders in BI — the exact workflow expected of
Data Analyst, Risk Analyst, and Business Analyst roles at banks, NBFCs, and fintech lenders.
