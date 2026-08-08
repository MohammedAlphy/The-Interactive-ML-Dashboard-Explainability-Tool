import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st



# Force Streamlit to display immediate text to verify execution
st.write("### App is initializing...")


@st.cache_resource
def load_artifacts():
    # Make sure this matches the exact filename saved by joblib
    artifacts = joblib.load("churn_model_pipeline.pkl")
    return (
        artifacts["model"],
        artifacts["preprocessor"],
        artifacts["feature_names"],
    )


# Debug block: Print exact error to browser instead of calling st.stop() silently
try:
    model, preprocessor, feature_names = load_artifacts()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load 'churn_model_pipeline.pkl': {e}")
    # Comment out st.stop() temporarily to inspect if the rest of the UI renders
    # st.stop()


# Import existing helper functions from pipeline.py
from pipeline import engineer_features, load_and_clean_data

# Set page configuration
st.set_page_config(
    page_title="Customer Churn & Explainability Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================
# 1. Load Pre-trained Model Artifacts
# ==========================================
@st.cache_resource
def load_artifacts():
    artifacts = joblib.load("churn_model_pipeline.pkl")
    return (
        artifacts["model"],
        artifacts["preprocessor"],
        artifacts["feature_names"],
    )


try:
    model, preprocessor, feature_names = load_artifacts()
    explainer = shap.TreeExplainer(model)
except Exception as e:
    st.error(
        f"Error loading model artifacts. Make sure 'churn_model_pipeline.pkl' exists. Details: {e}"
    )
    st.stop()

# ==========================================
# 2. Sidebar Controls
# ==========================================
st.sidebar.title("Dashboard Controls")

# File Uploader
uploaded_file = st.sidebar.file_uploader(
    "Upload Client CSV Dataset", type=["csv"]
)

# Probability Sensitivity Threshold
threshold = st.sidebar.slider(
    "Prediction Sensitivity Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="Customers with predicted churn probability above this threshold are classified as At-Risk.",
)

# ==========================================
# 3. Main Dashboard Layout
# ==========================================
st.title("📊 Customer Churn Intelligence & Explainability Dashboard")

if uploaded_file is not None:
    # Read raw data
    raw_df = pd.read_csv(uploaded_file)

    # --- Dataset Insights Section ---
    st.subheader("📌 Dataset Insights & Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        total_sales = raw_df["Purchase_Amount"].sum()
        st.metric("Total Revenue", f"${total_sales:,.2f}")

    with col2:
        top_country = (
            raw_df.groupby("Country")["Purchase_Amount"].sum().idxmax()
        )
        st.metric("Top Revenue Country", top_country)

    with col3:
        avg_freq = raw_df.groupby("User_Name")["Transaction_ID"].count().mean()
        st.metric("Avg. Trans Frequency / User", f"{avg_freq:.1f}")

    # Country Filters in Sidebar
    all_countries = ["All"] + list(raw_df["Country"].unique())
    selected_country = st.sidebar.selectbox("Filter by Country", all_countries)

    filtered_raw_df = raw_df.copy()
    if selected_country != "All":
        filtered_raw_df = filtered_raw_df[
            filtered_raw_df["Country"] == selected_country
        ]

    # Process features using pipeline
    uploaded_file.seek(0)
    processed_df = load_and_clean_data(uploaded_file)
    featured_df = engineer_features(processed_df)

    if selected_country != "All":
        featured_df = featured_df[featured_df["Country"] == selected_country]

    # Preprocess & Predict Probabilities
    X_prep = preprocessor.transform(featured_df)
    X_prep_df = pd.DataFrame(
        X_prep, columns=feature_names, index=featured_df.index
    )

    # Get positive class probabilities
    probabilities = model.predict_proba(X_prep_df)[:, 1]
    featured_df["Churn_Probability"] = probabilities
    featured_df["Predicted_At_Risk"] = (probabilities >= threshold).astype(int)

    # --- Real-Time Summary Metric Cards ---
    st.subheader("⚡ Risk Metrics")

    at_risk_count = featured_df["Predicted_At_Risk"].sum()
    potential_revenue_loss = featured_df[featured_df["Predicted_At_Risk"] == 1][
        "avg_spending_per_trans"
    ].sum()

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Analyzed Customers", len(featured_df))
    m2.metric(
        "At-Risk Customers",
        f"{at_risk_count}",
        f"{(at_risk_count/len(featured_df))*100:.1f}%",
    )
    m3.metric("Potential Revenue Loss", f"${potential_revenue_loss:,.2f}")

    # --- Interactive SHAP Waterfall Section ---
    st.markdown("---")
    st.subheader("🔍 Local Explainability (Individual Customer Drivers)")

    customer_options = featured_df.index.tolist()
    selected_customer_idx = st.selectbox(
        "Select Customer Row Index to Explain:", customer_options
    )

    if selected_customer_idx is not None:
        row_pos = featured_df.index.get_loc(selected_customer_idx)

        st.write(
            f"**Predicted Churn Probability:** `{probabilities[row_pos]:.2%}`"
        )

        # Compute SHAP
        shap_values = explainer(X_prep_df)

        fig, ax = plt.subplots(figsize=(8, 4))
        shap.plots.waterfall(shap_values[row_pos], max_display=7, show=False)
        plt.tight_layout()

        st.pyplot(fig)

else:
    st.info("👈 Please upload a CSV file in the sidebar to begin analysis.")


# ==========================================
# Advanced Country & Category Analytics
# ==========================================
st.markdown("---")
st.subheader("📈 Country & Category Deep-Dive")

tab1, tab2, tab3 = st.tabs(
    [
        "🌍 Country Breakdown",
        "🏆 Top Customer per Category",
        "💡 Targeted Recommendations",
    ]
)

with tab1:
    col_a, col_b = st.columns(2)

    # 1. Age Group that buys the most per country
    bins = [17, 25, 35, 50, 65, 100]
    labels = ["18-25", "26-35", "36-50", "51-65", "65+"]
    raw_df["Age_Group"] = pd.cut(raw_df["Age"], bins=bins, labels=labels)

    age_country = (
        raw_df.groupby(["Country", "Age_Group"], observed=False)[
            "Purchase_Amount"
        ]
        .sum()
        .reset_index()
    )
    top_age_country = age_country.loc[
        age_country.groupby("Country")["Purchase_Amount"].idxmax()
    ]

    with col_a:
        st.write("**Top Spending Age Bracket per Country**")
        st.dataframe(
            top_age_country.rename(
                columns={
                    "Age_Group": "Top Age Bracket",
                    "Purchase_Amount": "Total Spend ($)",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    # 2. What each country bought the most (Product Category)
    cat_country = (
        raw_df.groupby(["Country", "Product_Category"])["Purchase_Amount"]
        .sum()
        .reset_index()
    )
    top_cat_country = cat_country.loc[
        cat_country.groupby("Country")["Purchase_Amount"].idxmax()
    ]

    with col_b:
        st.write("**Top Product Category per Country**")
        st.dataframe(
            top_cat_country.rename(
                columns={
                    "Product_Category": "Top Category",
                    "Purchase_Amount": "Total Spend ($)",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

with tab2:
    # 3. Account with highest transaction frequency for each category
    user_cat_freq = (
        raw_df.groupby(["Product_Category", "User_Name"])["Transaction_ID"]
        .count()
        .reset_index()
    )
    top_user_per_cat = user_cat_freq.loc[
        user_cat_freq.groupby("Product_Category")["Transaction_ID"].idxmax()
    ]

    st.write("**Most Frequent Buyer per Product Category**")
    st.dataframe(
        top_user_per_cat.rename(
            columns={
                "User_Name": "Top Buyer",
                "Transaction_ID": "Transaction Count",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

with tab3:
    # ==========================================
    # Automated Guidance & Recommendation Engine
    # ==========================================
    st.write("### 🤖 Customer Action & Ad Targeting Guidance")

    # Dropdown to evaluate individual accounts
    selected_user = st.selectbox(
        "Select Customer Account to Generate Recommendations:",
        raw_df["User_Name"].unique(),
    )

    user_data = raw_df[raw_df["User_Name"] == selected_user]

    # Calculate user category preference ratio
    category_counts = user_data["Product_Category"].value_counts(
        normalize=True
    )
    top_category = category_counts.index[0]
    top_category_pct = category_counts.iloc[0]

    # Guidance Rules
    st.info(f"Analyzing behavioral profile for **{selected_user}**...")

    # Rule 1: Category Dominance
    if top_category_pct >= 0.25:
        st.success(
            f"🎯 **Targeted Ad Alert:** Customer spends `{top_category_pct:.1%}` of their budget on **{top_category}**. "
            f"Action: Increase ad impressions for high-margin items in **{top_category}**."
        )

    # Rule 2: Promotional / Discount Sensitivity (Flexible Column Check)
    if "Is_Discounted" in user_data.columns:
        discount_ratio = user_data["Is_Discounted"].mean()
        if discount_ratio > 0.5:
            st.warning(
                f"🏷️ **Deal-Seeker Profile:** `{discount_ratio:.1%}` of purchases were made using promotions. "
                f"Action: Enroll customer in automated SMS/Email alerts whenever seasonal sales go live."
            )
        else:
            st.success(
                "💎 **Full-Price Buyer:** Customer rarely relies on discounts. Action: Target with early-access premium collections."
            )
    else:
        # Fallback message when offer data is not present in the dataset
        st.caption(
            "💡 *Note: Upload datasets containing an `Is_Discounted` or `Offer_Code` column to unlock automated sales/deal-sensitivity triggers.*"
        )