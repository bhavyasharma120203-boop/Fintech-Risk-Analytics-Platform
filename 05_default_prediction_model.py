"""
Step 5: Loan Default Prediction Model
Trains Logistic Regression (interpretable baseline) and Random Forest (higher performance),
compares them on standard classification metrics, and saves the best model + feature importances.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, roc_auc_score, roc_curve,
                              confusion_matrix, precision_recall_curve)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

OUT = "/home/claude/fintech_risk/outputs"
MODELS = "/home/claude/fintech_risk/models"
df = pd.read_csv(f"{OUT}/master_dataset.csv")

# ------------------------------------------------------------------
# Feature selection
# ------------------------------------------------------------------
numeric_features = [
    "age", "annual_income", "credit_score", "existing_loans_count", "existing_debt",
    "loan_amount", "interest_rate", "tenure_months", "dti_ratio",
    "avg_monthly_balance", "avg_monthly_inflow", "avg_monthly_outflow",
    "num_bounced_payments_last_year", "num_active_credit_lines",
    "loan_to_income_ratio", "customer_tenure_years"
]
categorical_features = ["gender", "state", "city_tier", "employment_type",
                         "education", "loan_purpose"]

X = df[numeric_features + categorical_features]
y = df["is_default"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}, Default rate train: {y_train.mean():.3f}")

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
])

# ------------------------------------------------------------------
# Model 1: Logistic Regression
# ------------------------------------------------------------------
logreg = Pipeline([
    ("prep", preprocessor),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
])
logreg.fit(X_train, y_train)
y_pred_lr = logreg.predict(X_test)
y_prob_lr = logreg.predict_proba(X_test)[:,1]
auc_lr = roc_auc_score(y_test, y_prob_lr)

print("\n=== Logistic Regression ===")
print(classification_report(y_test, y_pred_lr))
print(f"AUC-ROC: {auc_lr:.4f}")

# ------------------------------------------------------------------
# Model 2: Random Forest
# ------------------------------------------------------------------
rf = Pipeline([
    ("prep", preprocessor),
    ("clf", RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=20,
        class_weight="balanced", random_state=42, n_jobs=-1
    ))
])
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:,1]
auc_rf = roc_auc_score(y_test, y_prob_rf)

print("\n=== Random Forest ===")
print(classification_report(y_test, y_pred_rf))
print(f"AUC-ROC: {auc_rf:.4f}")

# ------------------------------------------------------------------
# Comparison plots
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# ROC curves
for name, y_prob, auc in [("Logistic Regression", y_prob_lr, auc_lr), ("Random Forest", y_prob_rf, auc_rf)]:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    axes[0].plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
axes[0].plot([0,1],[0,1],"k--", alpha=0.4)
axes[0].set_title("ROC Curve Comparison")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].legend()

# Confusion matrix (RF - best model assumed)
cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1],
            xticklabels=["No Default","Default"], yticklabels=["No Default","Default"])
axes[1].set_title("Confusion Matrix - Random Forest")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

# Feature importance (RF)
ohe_cols = rf.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(categorical_features)
all_feature_names = numeric_features + list(ohe_cols)
importances = rf.named_steps["clf"].feature_importances_
fi = pd.Series(importances, index=all_feature_names).sort_values(ascending=False).head(15)
sns.barplot(x=fi.values, y=fi.index, ax=axes[2], color="teal")
axes[2].set_title("Top 15 Feature Importances - Random Forest")

plt.tight_layout()
plt.savefig(f"{OUT}/model_evaluation.png", dpi=120)
print(f"\nSaved -> {OUT}/model_evaluation.png")

# ------------------------------------------------------------------
# Save best model + predictions
# ------------------------------------------------------------------
best_model = rf if auc_rf >= auc_lr else logreg
best_name = "random_forest" if auc_rf >= auc_lr else "logistic_regression"
joblib.dump(best_model, f"{MODELS}/default_prediction_model.pkl")
print(f"Best model: {best_name} -> saved to {MODELS}/default_prediction_model.pkl")

# Score the full dataset and export risk scores for the dashboard
df["default_probability"] = best_model.predict_proba(X)[:,1]
df["predicted_risk_tier"] = pd.cut(
    df["default_probability"], bins=[0,0.1,0.25,0.5,1.0],
    labels=["Low","Medium","High","Very High"]
)
df[["loan_id","customer_id","is_default","default_probability","predicted_risk_tier"]].to_csv(
    f"{OUT}/loan_risk_scores.csv", index=False
)
print(f"Saved -> {OUT}/loan_risk_scores.csv")

# Model comparison summary
comparison = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest"],
    "AUC_ROC": [auc_lr, auc_rf],
})
comparison.to_csv(f"{OUT}/model_comparison.csv", index=False)
print("\n", comparison)
