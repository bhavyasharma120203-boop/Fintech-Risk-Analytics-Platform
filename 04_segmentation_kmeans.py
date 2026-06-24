"""
Step 4: Customer Risk Segmentation (Fintech analogue of RFM)
Instead of Recency/Frequency/Monetary (retail), we use the fintech equivalent:
  - Credit Score (creditworthiness)
  - DTI Ratio (leverage / repayment burden)
  - Missed Installment Rate (repayment behavior)
  - Loan-to-Income Ratio (exposure)
K-Means clusters customers into segments: Prime, Stable, Watchlist, High-Risk.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

OUT = "/home/claude/fintech_risk/outputs"
df = pd.read_csv(f"{OUT}/master_dataset.csv")

# Aggregate to customer level (a customer may have multiple loans)
cust = df.groupby("customer_id").agg(
    credit_score=("credit_score","mean"),
    dti_ratio=("dti_ratio","mean"),
    missed_installment_rate=("missed_installment_rate","mean"),
    loan_to_income_ratio=("loan_to_income_ratio","mean"),
    total_loans=("loan_id","count"),
    total_loan_amount=("loan_amount","sum"),
    any_default=("is_default","max"),
).reset_index()

features = ["credit_score","dti_ratio","missed_installment_rate","loan_to_income_ratio"]
X = cust[features].copy()
X["dti_ratio"] = X["dti_ratio"].clip(0,3)
X["loan_to_income_ratio"] = X["loan_to_income_ratio"].clip(0,5)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow method
inertias = []
K_range = range(2,9)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
    inertias.append(km.inertia_)

plt.figure(figsize=(7,5))
plt.plot(list(K_range), inertias, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal k")
plt.tight_layout()
plt.savefig(f"{OUT}/kmeans_elbow.png", dpi=120)

# Final clustering with k=4
k = 4
km = KMeans(n_clusters=k, random_state=42, n_init=10)
cust["cluster"] = km.fit_predict(X_scaled)

# Label clusters by mean credit score (higher score + lower risk metrics = better segment)
cluster_profile = cust.groupby("cluster")[features + ["any_default"]].mean()
cluster_profile["risk_rank"] = cluster_profile["any_default"].rank()
order = cluster_profile.sort_values("any_default").index.tolist()

label_map = {}
names = ["Prime Customers", "Stable Customers", "Watchlist Customers", "High-Risk Customers"]
for i, c in enumerate(order):
    label_map[c] = names[i]

cust["segment"] = cust["cluster"].map(label_map)

print("=== Cluster Profiles (mean values) ===")
print(cluster_profile.round(3))
print("\n=== Segment sizes ===")
print(cust["segment"].value_counts())
print("\n=== Default rate by segment ===")
print(cust.groupby("segment")["any_default"].mean().round(3).sort_values())

cust.to_csv(f"{OUT}/customer_segments.csv", index=False)
print(f"\nSaved -> {OUT}/customer_segments.csv")

# Visualization
plt.figure(figsize=(9,7))
sns.scatterplot(data=cust, x="credit_score", y="dti_ratio", hue="segment",
                 palette="viridis", alpha=0.4, s=15)
plt.title("Customer Risk Segments: Credit Score vs DTI Ratio")
plt.ylim(0,3)
plt.tight_layout()
plt.savefig(f"{OUT}/segments_scatter.png", dpi=120)
print(f"Saved -> {OUT}/segments_scatter.png")

# Segment summary bar chart
seg_summary = cust.groupby("segment").agg(
    customers=("customer_id","count"),
    avg_credit_score=("credit_score","mean"),
    default_rate=("any_default","mean"),
    avg_loan_amount=("total_loan_amount","mean"),
).reindex(names)
seg_summary.to_csv(f"{OUT}/segment_summary.csv")
print(seg_summary.round(2))
