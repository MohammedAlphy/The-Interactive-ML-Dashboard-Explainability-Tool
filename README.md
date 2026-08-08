# 📊 Interactive ML Dashboard & Explainability Tool

An end-to-end Streamlit web application that predicts customer churn risk, delivers regional sales analytics, and explains model decisions using **SHAP (SHapley Additive exPlanations)**.


## 🚀 Key Features
* **Universal Column Mapping:** Built-in standardization that dynamically normalizes mismatched
  e-commerce CSV headers (`revenue`, `TotalAmount`, `order_date`, `CustomerID`, etc.).
* **Dynamic Feature Alignment:** Automatically handles missing training features and aligns schema order to ensure crash-free model execution across varied inputs.
* **Churn Prediction & Financial Risk:** Flags high-risk accounts based on a customizable probability threshold slider and estimates total revenue at risk.
* **Local Model Interpretability:** Explains individual customer predictions using SHAP waterfall plots.
* **Country & Category Insights:** Deep-dive analytics tabs providing buyer breakdowns, spend distribution, and automated marketing rule recommendations.

---

## 🧪 Tested Datasets
The application and data pipeline have been validated against the following benchmark dataset schemas:
1. **E-Commerce Customer Transactions Dataset** (Standard 5,000-row baseline)
2. **Amazon Sales Dataset** (~100,000 transaction records)
3. **Retail & Order History Dataset** (Alternative column naming format)

---

## ⚠️ Schema Compatibility & Bug Reporting

While the data pipeline includes dynamic column mapping and default feature padding, unexpected dataset structures (e.g., non-standard datetime formats, missing critical user identifiers, or unique header names) may occasionally trigger a `KeyError` or preprocessing exception.

### 📬 Found a bug or running a new dataset?
If you encounter an error while uploading a custom dataset that's completely normal:
1. **Open an Issue:** Submit a bug report on the repository's **Issues** tab. Or reach out directly at my [Gmail](mohammedlotfyismail@gmail.com).
3. **What to include:**
   * A snippet or sample row of your dataset's header columns.
   * The exact error message / traceback displayed on Streamlit.

---

### 🛠️ Project Structure
```
├── app.py                   # Streamlit dashboard interface & execution flow
├── pipeline.py              # Data cleaning, column standardization & feature engineering
├── churn_model_pipeline.pkl # Pre-trained ML model artifact & preprocessor
├── requirements.txt         # Required Python dependencies
└── README.md                # Project documentation
```
---

## ⚙️ Quickstart Setup

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/Project-1-Interactive-ML-Dashboard.git](https://github.com/YOUR_USERNAME/Project-1-Interactive-ML-Dashboard.git)
   cd Project-1-Interactive-ML-Dashboard



2. **Install Dependencies:**
```bash
pip install -r requirements.txt

```


3. **Run the Dashboard:**
```bash
streamlit run app.py

```
