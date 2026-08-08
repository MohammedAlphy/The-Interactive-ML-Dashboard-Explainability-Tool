import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

# Import pipeline helpers
from pipeline import engineer_features, load_and_clean_data

# ==========================================
# Page Configuration
# ==========================================
st.set_page_config(
    page_title="Customer Churn & Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Customer Churn Intelligence & Explainability Dashboard")
st.markdown(
    "Upload a customer transaction dataset to analyze financial risk, explore country/category insights, and explain machine learning predictions using **SHAP**."
)


# ==========================================
# Universal Column Standardization
# ==========================================
def standardize_columns(df):
    """Maps common e-commerce column aliases across dataset formats to standard project names."""
    column_mapping = {
        # Revenue / Amount
        "totalamount": "Purchase_Amount",
        "revenue": "Purchase_Amount",
        "sales": "Purchase_Amount",
        "amount": "Purchase_Amount",
        "price": "Purchase_Amount",
        "unitprice": "Purchase_Amount",
        # Geography / Country
        "region": "Country",
        "countryname": "Country",
        "location": "Country",
        "country": "Country",
        # User Identifier
        "customerid": "User_Name",
        "customername": "User_Name",
        "userid": "User_Name",
        "clientid": "User_Name",
        "username": "User_Name",
        # Category
        "productcategory": "Product_Category",
        "category": "Product_Category",
        # Dates & IDs
        "orderdate": "Transaction_Date",
        "date": "Transaction_Date",
        "transactiondate": "Transaction_Date",
        "orderid": "Transaction_ID",
        "transactionid": "Transaction_ID",
        "invoiceno": "Transaction_ID",
        # Demographics, Payment & Offers
        "age": "Age",
        "customerage": "Age",
        "discount": "Is_Discounted",
        "isdiscounted": "Is_Discounted",
        "paymentmethod": "Payment_Method",
        "paymethod": "Payment_Method",
        "paymenttype": "Payment_Method",
    }

    # Normalize current column names (lowercase, no spaces, no underscores)
    current_cols = {
        str(col).lower().replace("_", "").replace(" ", "").strip(): col
        for col in df.columns
    }

    rename_dict = {}
    assigned_targets = set()

    for src_alias, target_col in column_mapping.items():
        if src_alias in current_cols and target_col not in assigned_targets:
            orig_col = current_cols[src_alias]
            if orig_col == target_col:
                assigned_targets.add(target_col)
                continue
            rename_dict[orig_col] = target_col
            assigned_targets.add(target_col)

    df = df.rename(columns=rename_dict)

    # Safely convert Transaction_Date to datetime
    if "Transaction_Date" in df.columns:
        df["Transaction_Date"] = pd.to_datetime(
            df["Transaction_Date"], errors="coerce"
        )

    return df


# ==========================================
# Load Model Artifacts & Alignment Helpers
# ==========================================
@st.cache_resource
def load_model_artifact():
    return joblib.load("churn_model_pipeline.pkl")


model_artifact = load_model_artifact()


def align_df_to_preprocessor(df, preprocessor):
    """Fills missing model features with appropriate default types and aligns column order."""
    if preprocessor is not None and hasattr(preprocessor, "feature_names_in_"):
        expected_cols = list(preprocessor.feature_names_in_)
        df_aligned = df.copy()

        # Identify categorical columns from the transformer pipeline
        cat_cols = set()
        if hasattr(preprocessor, "transformers_"):
            for name, trans, cols in preprocessor.transformers_:
                if name == "remainder" and trans == "drop":
                    continue
                cols_list = cols if isinstance(cols, list) else [cols]
                trans_name = f"{type(trans).__name__} {name}".lower()
                if any(k in trans_name for k in ["onehot", "cat", "encoder", "string"]):
                    cat_cols.update(cols_list)

        # Pad missing features with string fallback for encodings and np.nan for numeric scaling
        for col in expected_cols:
            if col not in df_aligned.columns:
                if col in cat_cols:
                    df_aligned[col] = "Missing"
                else:
                    df_aligned[col] = np.nan

        return df_aligned[expected_cols]

    return df


def get_predictions_and_shap_data(artifact, df):
    """Safely aligns features, transforms data, and generates model predictions."""
    df_features = df.drop(
        columns=["Churn_Probability", "Predicted_Churn"], errors="ignore"
    )

    # Case A: Artifact is a dictionary containing preprocessor and model
    if isinstance(artifact, dict):
        preprocessor = artifact.get("preprocessor")
        model = (
            artifact.get("model")
            or artifact.get("classifier")
            or artifact.get("pipeline")
        )

        if preprocessor is not None:
            df_aligned = align_df_to_preprocessor(df_features, preprocessor)
            X_trans = preprocessor.transform(df_aligned)
            probs = model.predict_proba(X_trans)[:, 1]
            return probs, preprocessor, model, X_trans
        else:
            df_cat = df_features.copy()
            for col in df_cat.select_dtypes(include=["object"]).columns:
                df_cat[col] = df_cat[col].astype("category")
            probs = model.predict_proba(df_cat)[:, 1]
            return probs, None, model, df_cat

    # Case B: Artifact is a Scikit-Learn Pipeline
    elif hasattr(artifact, "named_steps"):
        preprocessor = artifact.named_steps.get("preprocessor")
        model = artifact.named_steps.get("classifier", artifact)

        if preprocessor is not None:
            df_aligned = align_df_to_preprocessor(df_features, preprocessor)
            X_trans = preprocessor.transform(df_aligned)
            probs = model.predict_proba(X_trans)[:, 1]
        else:
            probs = artifact.predict_proba(df_features)[:, 1]
            X_trans = df_features

        return probs, preprocessor, model, X_trans

    # Case C: Standalone Model
    else:
        df_cat = df_features.copy()
        for col in df_cat.select_dtypes(include=["object"]).columns:
            df_cat[col] = df_cat[col].astype("category")
        probs = artifact.predict_proba(df_cat)[:, 1]
        return probs, None, artifact, df_cat


# ==========================================
# Sidebar Controls
# ==========================================
st.sidebar.header("⚙️ Configuration")
uploaded_file = st.sidebar.file_uploader(
    "Upload Customer CSV Dataset", type=["csv"]
)

threshold = st.sidebar.slider(
    "Churn Probability Risk Threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05,
    help="Customers with a predicted churn probability above this threshold will be flagged as high risk.",
)

# ==========================================
# Main Dashboard Execution
# ==========================================
if uploaded_file is not None:
    # 1. Read Raw CSV Stream Once & Standardize
    raw_df = pd.read_csv(uploaded_file)
    raw_df = standardize_columns(raw_df)

    # 2. Defensive Age Binning (Only if 'Age' column exists)
    if "Age" in raw_df.columns:
        bins = [17, 25, 35, 50, 65, 100]
        labels = ["18-25", "26-35", "36-50", "51-65", "65+"]
        raw_df["Age_Group"] = pd.cut(raw_df["Age"], bins=bins, labels=labels)

    # --- Dataset Overview Section ---
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

    # Sidebar Country Filter
    all_countries = ["All"] + list(raw_df["Country"].unique())
    selected_country = st.sidebar.selectbox("Filter by Country", all_countries)

    if selected_country != "All":
        raw_df = raw_df[raw_df["Country"] == selected_country].reset_index(
            drop=True
        )

    # --- Advanced Analytics Section ---
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

        with col_a:
            st.write("**Top Spending Age Bracket per Country**")
            if "Age_Group" in raw_df.columns:
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
            else:
                st.info(
                    "ℹ️ Uploaded dataset does not contain customer age data."
                )

        with col_b:
            st.write("**Top Product Category per Country**")
            cat_country = (
                raw_df.groupby(["Country", "Product_Category"])[
                    "Purchase_Amount"
                ]
                .sum()
                .reset_index()
            )
            top_cat_country = cat_country.loc[
                cat_country.groupby("Country")["Purchase_Amount"].idxmax()
            ]
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
        st.write("### 🤖 Customer Action & Ad Targeting Guidance")
        selected_user_guide = st.selectbox(
            "Select Customer Account to Generate Recommendations:",
            raw_df["User_Name"].unique(),
            key="user_guide_select",
        )

        user_data = raw_df[raw_df["User_Name"] == selected_user_guide]

        category_counts = user_data["Product_Category"].value_counts(
            normalize=True
        )
        top_category = category_counts.index[0]
        top_category_pct = category_counts.iloc[0]

        st.info(
            f"Analyzing behavioral profile for **{selected_user_guide}**..."
        )

        if top_category_pct >= 0.25:
            st.success(
                f"🎯 **Targeted Ad Alert:** Customer spends `{top_category_pct:.1%}` of their transactions on **{top_category}**. "
                f"Action: Serve targeted ads and promotions for high-margin items in **{top_category}**."
            )

        if "Is_Discounted" in user_data.columns:
            discount_ratio = user_data["Is_Discounted"].mean()
            if discount_ratio > 0.5:
                st.warning(
                    f"🏷️ **Deal-Seeker Profile:** `{discount_ratio:.1%}` of purchases were promotional. "
                    f"Action: Enroll customer in automated SMS/Email sale notifications."
                )
            else:
                st.success(
                    "💎 **Full-Price Buyer:** Customer rarely uses discounts. Action: Direct target with early-access premium collections."
                )
        else:
            st.caption(
                "💡 *Note: Include an `Is_Discounted` or `Offer_Code` column to unlock discount sensitivity rules.*"
            )

    # --- ML Churn Prediction & Risk Evaluation ---
    st.markdown("---")
    st.subheader("🤖 Churn Predictions & Risk Evaluation")

    # Reset stream pointer for pipeline consumption
    uploaded_file.seek(0)
    processed_df = load_and_clean_data(uploaded_file)
    processed_df = standardize_columns(processed_df)

    featured_df = engineer_features(processed_df)

    # Generate Model Predictions safely via helper
    churn_probs, preprocessor, xgb_model, transformed_features = (
        get_predictions_and_shap_data(model_artifact, featured_df)
    )

    featured_df["Churn_Probability"] = churn_probs
    featured_df["Predicted_Churn"] = (churn_probs >= threshold).astype(int)

    # Risk Metrics
    high_risk_count = featured_df["Predicted_Churn"].sum()
    at_risk_revenue = raw_df[
        raw_df["User_Name"].isin(
            featured_df[featured_df["Predicted_Churn"] == 1].index
        )
    ]["Purchase_Amount"].sum()

    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Analyzed Customers", len(featured_df))
    m_col2.metric("Flagged High-Risk Customers", high_risk_count)
    m_col3.metric("Estimated Revenue at Risk", f"${at_risk_revenue:,.2f}")

    # Risk Directory Table
    st.write("### 📋 Customer Risk Directory")
    st.dataframe(
        featured_df[["Churn_Probability", "Predicted_Churn"]].sort_values(
            by="Churn_Probability", ascending=False
        ),
        use_container_width=True,
    )

    # --- SHAP Explainability Section ---
    st.markdown("---")
    st.subheader("🔍 Local Model Interpretability (SHAP Waterfall)")

    selected_account = st.selectbox(
        "Select Customer Account to Explain Churn Decision:",
        featured_df.index.unique(),
    )

    if selected_account is not None:
        account_idx = featured_df.index.get_loc(selected_account)

        # Compute SHAP values using transformed features
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer(transformed_features)

        fig, ax = plt.subplots(figsize=(8, 5))
        shap.plots.waterfall(shap_values[account_idx], show=False)
        st.pyplot(fig)

else:
    # Initial startup screen when no CSV is provided
    st.info(
        "👋 **Welcome!** Please upload a customer CSV dataset in the sidebar to generate metrics, deep-dive insights, and SHAP model explanations."
    )
