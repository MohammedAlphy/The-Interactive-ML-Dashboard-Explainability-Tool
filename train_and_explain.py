import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from pipeline import engineer_features, load_and_clean_data, print_dataset_insights

# ==========================================
# 1. Load Data & Prepare Target (Churn)
# ==========================================
raw_df = load_and_clean_data("ecommerce_transactions.csv")

print_dataset_insights("ecommerce_transactions.csv")


# Synthetic Churn Label: 1 if customer's last purchase was > 90 days from latest dataset date
latest_date = raw_df["Transaction_Date"].max()
user_last_purchase = (
    raw_df.groupby("User_Name")["Transaction_Date"].max().reset_index()
)
user_last_purchase["days_since_last"] = (
    latest_date - user_last_purchase["Transaction_Date"]
).dt.days
user_last_purchase["Churn"] = (
    user_last_purchase["days_since_last"] > 90
).astype(int)

# Merge Churn target into featured data
featured_df = engineer_features(raw_df)

# For demonstration, assume user-level aggregated dataset for training
df_model = raw_df.merge(
    user_last_purchase[["User_Name", "Churn"]], on="User_Name"
)
df_model = engineer_features(df_model)

X = df_model.drop(columns=["Churn"])
y = df_model["Churn"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================================
# 2. Preprocessing & Model Setup
# ==========================================
num_cols = [
    "Age",
    "Purchase_Amount",
    "tenure_months",
    "transaction_count",
    "avg_spending_per_trans",
]
cat_cols = ["Country", "Payment_Method", "Product_Category"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            cat_cols,
        ),
    ]
)

# Transform data
X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep = preprocessor.transform(X_test)

# Get feature names after OneHotEncoding
cat_encoder = preprocessor.named_transformers_["cat"]
encoded_cat_cols = cat_encoder.get_feature_names_out(cat_cols).tolist()
feature_names = num_cols + encoded_cat_cols

# Convert transformed arrays back to DataFrame for SHAP readability
X_train_prep_df = pd.DataFrame(
    X_train_prep, columns=feature_names, index=X_train.index
)
X_test_prep_df = pd.DataFrame(
    X_test_prep, columns=feature_names, index=X_test.index
)

# Train XGBoost Model
model = XGBClassifier(
    n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42
)
model.fit(X_train_prep_df, y_train)

# ==========================================
# 3. SHAP Explainability (XAI)
# ==========================================
# Initialize Tree Explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test_prep_df)

# --- Global Feature Importance ---
print("Generating Global Feature Importance Plot...")
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_prep_df, show=False)
plt.title("Global Feature Importance (Churn Drivers)", fontsize=14)
plt.tight_layout()
plt.savefig("shap_global_importance.png")
plt.close()

# --- Local Feature Importance (e.g., Customer #1042 or index 0) ---
customer_idx = 0  # Replace with specific index or customer ID lookup
print(f"Generating Local Waterfall Plot for Customer Index {customer_idx}...")

plt.figure(figsize=(10, 6))
shap.plots.waterfall(shap_values[customer_idx], show=False)
plt.title(f"Churn Prediction Breakdown for Sample Customer", fontsize=14)
plt.tight_layout()
plt.savefig("shap_local_customer_breakdown.png")
plt.close()





# ==========================================
# 4. Save Model and Preprocessor to Disk
# ==========================================
artifacts = {
    "model": model,
    "preprocessor": preprocessor,
    "feature_names": feature_names,
}

joblib.dump(artifacts, "churn_model_pipeline.pkl")
print("Saved model pipeline to 'churn_model_pipeline.pkl'")

