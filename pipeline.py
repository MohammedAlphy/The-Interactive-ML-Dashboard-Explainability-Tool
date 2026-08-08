import pandas as pd
import numpy as np
import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def load_and_clean_data(file_input) -> pd.DataFrame:
    """Reads raw CSV data from a file path or a Streamlit uploaded file object."""
    if isinstance(file_input, str):
        df = pd.read_csv(file_input)
    else:
        # Streamlit uploaded file handler
        df = pd.read_csv(file_input)

    # Clean logic
    if "Transaction_ID" in df.columns:
        df.drop(columns=["Transaction_ID"], inplace=True)

    if "Transaction_Date" in df.columns:
        df["Transaction_Date"] = pd.to_datetime(df["Transaction_Date"])

    df.ffill(inplace=True)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates aggregate customer-level features such as tenure and frequency."""
    df = df.copy()
    
    # Calculate customer tenure (in months) relative to the latest date in the dataset
    max_date = df['Transaction_Date'].max()
    
    # Calculate account-level metrics
    user_metrics = df.groupby('User_Name').agg(
        first_purchase=('Transaction_Date', 'min'),
        transaction_count=('Purchase_Amount', 'count'),
        total_spent=('Purchase_Amount', 'sum')
    ).reset_index()
    
    # Tenure in months
    user_metrics['tenure_months'] = (max_date - user_metrics['first_purchase']).dt.days / 30.44
    user_metrics['avg_spending_per_trans'] = user_metrics['total_spent'] / user_metrics['transaction_count']
    
    # Merge user aggregate features back to main DataFrame
    df = df.merge(
        user_metrics[['User_Name', 'tenure_months', 'transaction_count', 'avg_spending_per_trans']], 
        on='User_Name', 
        how='left'
    )
    
    # Drop high-cardinality metadata columns not used directly in modeling
    df.drop(columns=['User_Name', 'Transaction_Date'], inplace=True)
    
    return df


def build_preprocessor(numeric_features: list, categorical_features: list) -> ColumnTransformer:
   
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    return preprocessor


def prepare_data_splits(df: pd.DataFrame, target_column: str, test_size: float = 0.2, random_state: int = 42):
   
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

import pandas as pd


def print_dataset_insights(filepath: str):
    df = pd.read_csv(filepath)
    df["Transaction_Date"] = pd.to_datetime(df["Transaction_Date"])

    print("=" * 50)
    print("DATASET INSIGHTS SUMMARY")
    print("=" * 50)

    # 1. Age Region/Group that buys the most from each country
    bins = [17, 25, 35, 50, 65, 100]
    labels = ["18-25", "26-35", "36-50", "51-65", "65+"]
    df["Age_Group"] = pd.cut(df["Age"], bins=bins, labels=labels)

    age_by_country = (
        df.groupby(["Country", "Age_Group"], observed=False)["Purchase_Amount"]
        .sum()
        .reset_index()
    )
    top_age_per_country = age_by_country.loc[
        age_by_country.groupby("Country")["Purchase_Amount"].idxmax()
    ]

    print("\n1. Top Purchasing Age Group per Country:")
    for _, row in top_age_per_country.iterrows():
        print(
            f"   - {row['Country']}: Age Group {row['Age_Group']} (${row['Purchase_Amount']:,.2f})"
        )

    # 2. What each country bought the most (Product Category)
    cat_by_country = (
        df.groupby(["Country", "Product_Category"])["Purchase_Amount"]
        .sum()
        .reset_index()
    )
    top_cat_per_country = cat_by_country.loc[
        cat_by_country.groupby("Country")["Purchase_Amount"].idxmax()
    ]

    print("\n2. Top Product Category per Country:")
    for _, row in top_cat_per_country.iterrows():
        print(
            f"   - {row['Country']}: {row['Product_Category']} (${row['Purchase_Amount']:,.2f})"
        )

    # 3. Country that buys the most in general
    total_by_country = (
        df.groupby("Country")["Purchase_Amount"]
        .sum()
        .reset_index()
        .sort_values(by="Purchase_Amount", ascending=False)
    )
    top_country = total_by_country.iloc[0]

    print("\n3. Highest Spending Country Overall:")
    print(
        f"   - {top_country['Country']} with a total spend of ${top_country['Purchase_Amount']:,.2f}"
    )

    # 4. Average transaction frequency per account
    avg_freq = df.groupby("User_Name")["Transaction_ID"].count().mean()
    print(f"\n4. Average Transaction Frequency per Account: {avg_freq:.2f}")

    # 5. Customer tenure in months
    latest_date = df["Transaction_Date"].max()
    tenure_per_user = df.groupby("User_Name")["Transaction_Date"].agg(
        lambda x: (latest_date - x.min()).days / 30.44
    )
    avg_tenure = tenure_per_user.mean()
    print(f"5. Average Customer Tenure: {avg_tenure:.2f} months")
    print("=" * 50)


# Call the function
print_dataset_insights("ecommerce_transactions.csv")