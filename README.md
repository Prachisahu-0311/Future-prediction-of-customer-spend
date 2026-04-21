# **Short-Term Customer Spend Prediction (30-Day CLV)**

---

## 🚀 Project Overview

In the highly competitive retail industry, businesses need actionable insights to improve customer retention and personalize promotions. This project addresses this challenge by providing **short-term customer lifetime value (CLV)** predictions.

### **What We Solve**

The solution leverages **historical transaction data** to estimate how much a customer is expected to spend in the next **30 days**, ensuring businesses can:

- Design **personalized promotions**
- Enhance **retention and loyalty** strategies
- Create **revenue forecasts**

This is an **end-to-end production-ready pipeline**, focusing not only on **model accuracy** but also on:
- Robust preprocessing
- Handling time-based leakage
- Interpretable results
- Usable interfaces via a **UI tool** (Streamlit)

---

## 📝 Problem Definition

### **Business Objective**
Predict a customer’s spend for the upcoming 30 days based on their past purchase behavior, empowering decision-makers with practical insights.

### **Machine Learning Objective**
- **Problem Type**: **Supervised Regression**
- **Input**: Historical transactional features for each customer
- **Output**: Predicted spend for the next 30 days

---

## 📊 Dataset

### Dataset Used
- **Source**: [Ecommerce Dataset on Kaggle](https://www.kaggle.com/datasets/carrie1/ecommerce-data)
- **Description**: Contains real-world records of online retail transactions.

### **Key Columns Explained**
| Column        | Description                                                                 |
|---------------|-----------------------------------------------------------------------------|
| `CustomerID`  | Unique ID of the customer                                                  |
| `InvoiceNo`   | Invoice identifier                                                         |
| `InvoiceDate` | Date of transaction                                                        |
| `Quantity`    | Number of units purchased                                                  |
| `UnitPrice`   | Price per unit                                                             |
| `StockCode`   | Identifier for product type                                                |
| `Country`     | Customer’s country                                                        |

### **Derived Field**: 
- `TransactionAmount = Quantity × UnitPrice`

---

## 📚 Data Handling Strategy

### **Time-Based Cutoff to Mimic Real Inference**
A **time-aware cutoff** minimizes data leakage and reflects a real production scenario:
1. Transactions before the cutoff date are used for **feature engineering**.
2. Transactions in the 30 days after the cutoff date define the **target variable**.

---

## 🔧 Feature Engineering

The raw invoice-level transactions are aggregated into a **customer-level dataset**, with one row per customer. 

### **Implemented Features**
| Feature            | Description                                                                   |
|--------------------|-------------------------------------------------------------------------------|
| `frequency`        | Number of unique invoices                                                    |
| `total_spend`      | Total spend before the cutoff                                                |
| `avg_order_value`  | Mean transaction value                                                       |
| `total_quantity`   | Total items purchased                                                        |
| `unique_products`  | Number of distinct products purchased                                         |
| `recency_days`     | Days since the last purchase                                                 |
| `tenure_days`      | Customer lifespan (**first purchase → last purchase** before cutoff)         |

### **Target Variable**: `target_30d_spend`
- **Definition**: Total transaction amount spent in the **30 days after the cutoff**.
- **Special Case**: Customers without transactions post-cutoff are given a `target_30d_spend = 0`.

---

## 🤖 Modeling Approach

### Models Used:
#### 1. **Baseline: Linear Regression**
- Establishes a simple benchmark.
- Captures **linear relationships** in customer behavior.

#### 2. **Advanced: Random Forest Regressor**
- Handles **non-linear interactions** in data.
- Robust against outliers and skewed distributions.

### **Evaluation Metrics**
- **MAE (Mean Absolute Error)**: Measures average absolute prediction error.
- **RMSE (Root Mean Squared Error)**: Penalizes large prediction errors.
- **R² Score**: Explains variance captured by the model.

---

## 📈 **Validation Results**

| Model             | **MAE ↓** | **RMSE ↓** | **R² ↑** |
|-------------------|-----------|------------|----------|
| **Linear Regression** | 535.26    | 1237.06    | 0.52     |
| **Random Forest**     | 528.48    | 1542.18    | 0.25     |

### **Model Selection Rationale**
The **Random Forest Regressor** was chosen as the final model because:
1. It provided a **lower MAE**, reducing monetary prediction error (aligned with business goals).
2. Captured **non-linear behaviors** of customers' spending patterns.
3. Offers intrinsic **feature importance** for transparency.

---

## 🔬 **Feature Importance Analysis**

Random Forest highlighted the **most influential customer behaviors**:
1. **Total Spend**: Customer's historical spend is the most critical indicator.
2. **Average Order Value**
3. **Purchase Frequency**
4. **Recency**: How recently the customer made a purchase.
5. **Customer Tenure**

Feature importances are saved in: `models/feature_importance.csv`.

---

## ⚡ Inference Pipeline

During inference, this pipeline:
- Ingests a **customer's recent transactions**.
- Recomputes features dynamically up to the most recent purchase.
- Predicts 30-day spending using the trained Random Forest model.

The results can be queried programmatically or accessed via a **Streamlit-based UI**.

---

## 🖥 System Architecture

The **end-to-end architecture** is as follows:
```plaintext
Raw Transactions (CSV)
      ↓
 Data Preprocessing (remove noise, clean invalid rows)
      ↓
 Feature Engineering (aggregate meaningful customer-level features)
      ↓
 Model Training (Random Forest + Linear Regression)
      ↓
 Model Evaluation & Saving (Comparisons + feature importances)
      ↓
 Streamlit UI for predictions
```

---

## 🔧 Technology Stack

| Component                | Library                          |
|--------------------------|----------------------------------|
| **Language**             | Python                          |
| **Data Processing**      | Pandas, NumPy                   |
| **Modeling**             | Scikit-Learn                    |
| **Model Storage**        | Joblib                          |
| **UI Tool**              | Streamlit                       |
| **Version Control**      | Git, GitHub                     |

---

## 📂 Repository Structure

```plaintext
HCL-Hackathon-Data-Forge/
├── data/
│   ├── raw/                          # Raw transaction CSVs
│   └── processed/                    # Processed customer-level files
├── notebooks/                        # Jupyter Notebooks for EDA, experiments
├── src/
│   ├── data_cleaning.py              # Data cleaning scripts
│   ├── feature_engineering.py        # Feature engineering logic
│   ├── train_model.py                # Model training/evaluation scripts
│   ├── inference.py                  # Inference logic
├── models/
│   ├── rf_customer_spend_model.pkl   # Trained Random Forest model
│   ├── feature_importance.csv        # Feature importance analysis
│   ├── model_comparision.csv         # Validation metrics comparison
├── app.py                            # Streamlit app for predictions
├── requirements.txt                  # Python dependency file
└── README.md                         # Project documentation (this file)
```

---

## 🛠 Execution Steps

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Train the Model
```bash
python src/train_model.py
```

### Step 3: Run the Streamlit UI App
```bash
streamlit run app.py
```

---

## 🌐 Dashboard Link

Check out the deployed Streamlit CLV Analytics Dashboard:
👉 [CLV Dashboard](https://hcl-hackathon-data-forge-6sqn2hvvj8yfzycjp5ktgf.streamlit.app/)

This README provides a high-level yet detailed understanding of the **Short-Term CLV Prediction Project** and its end-to-end implementation.
