import numpy as np
import pandas as pd


def standardize_columns(df):
    """Maps common e-commerce column aliases to standard project names and strips duplicates."""
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

    # Normalize current column names (lowercase, no spaces/underscores)
    current_cols = {
        str(col).lower().replace("_", "").replace(" ", "").strip(): col
        for col in df.columns
    }

    rename_dict = {}
    assigned_targets = set()

    for src_alias, target_col in column_mapping.items():
        if src_alias in current_cols and target_col not in assigned_targets:
            orig_col = current_cols[src_alias]
            # Avoid renaming if the target column name is already in the original dataframe
            if target_col in df.columns:
                assigned_targets.add(target_col)
                continue
            rename_dict[orig_col] = target_col
            assigned_targets.add(target_col)

    df = df.rename(columns=rename_dict)

    # Remove duplicate columns if two raw columns matched the same standard name
    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    if "Transaction_Date" in df.columns:
        df["Transaction_Date"] = pd.to_datetime(
            df["Transaction_Date"], errors="coerce"
        )

    return df


def load_and_clean_data(file_input):
    """Loads CSV, standardizes column names, and applies initial dataset cleaning."""
    df = pd.read_csv(file_input)
    df = standardize_columns(df)

    # Drop duplicate transaction records if ID is present
    if "Transaction_ID" in df.columns:
        df = df.drop_duplicates(subset=["Transaction_ID"])

    # Drop rows missing critical user or date identifiers
    if "User_Name" in df.columns:
        df = df.dropna(subset=["User_Name"])
    if "Transaction_Date" in df.columns:
        df = df.dropna(subset=["Transaction_Date"])

    # Convert Purchase_Amount to numeric
    if "Purchase_Amount" in df.columns:
        df["Purchase_Amount"] = pd.to_numeric(
            df["Purchase_Amount"], errors="coerce"
        ).fillna(0.0)

    return df


def engineer_features(df):
    """Aggregates transaction-level data into user-level features for ML churn model inference."""
    df = standardize_columns(df)

    if "User_Name" not in df.columns or "Transaction_Date" not in df.columns:
        raise ValueError(
            "Dataset missing required 'User_Name' or 'Transaction_Date' columns."
        )

    max_date = df["Transaction_Date"].max()

    # Define dynamic aggregations based on column availability
    agg_dict = {
        "first_purchase": ("Transaction_Date", "min"),
        "last_purchase": ("Transaction_Date", "max"),
        "transaction_count": ("Transaction_Date", "count"),
    }

    if "Purchase_Amount" in df.columns:
        agg_dict["total_spend"] = ("Purchase_Amount", "sum")
        agg_dict["avg_spend"] = ("Purchase_Amount", "mean")

    if "Is_Discounted" in df.columns:
        agg_dict["discount_ratio"] = ("Is_Discounted", "mean")

    user_metrics = df.groupby("User_Name").agg(**agg_dict)

    # Compute tenure and recency metrics
    user_metrics["tenure_months"] = (
        (max_date - user_metrics["first_purchase"]).dt.days / 30.44
    )
    user_metrics["recency_days"] = (
        (max_date - user_metrics["last_purchase"]).dt.days
    )

    # Remove temporary datetime objects before passing to model transformer
    featured_df = user_metrics.drop(
        columns=["first_purchase", "last_purchase"], errors="ignore"
    )

    return featured_df
