# 💳 Credit Delinquency Early Warning System (EWS)

Welcome! 👋 This repository contains a complete, student-friendly Machine Learning project that predicts whether a bank customer will miss their credit card or loan payments in the near future.

By catching risk **early**, banks can reach out to customers to help them structure payments before they default.

---

## 🌟 What is this project about?

Imagine you run a bank. Some customers take credit cards or loans. Most pay on time, but a few face financial trouble and stop paying. 

- **The Problem:** If a bank waits until a customer has already stopped paying for 90 days, it is often too late to recover the money.
- **The Solution (EWS):** An **Early Warning System (EWS)** acts like a financial smoke detector. It analyzes payment patterns, credit usage, and income stress to spot warning signs **before** a default happens.

---

## 🚀 Key Features at a Glance

1. **🛡️ Data Leakage Prevention:** Drops future/hidden signals that real bank systems wouldn't have at prediction time.
2. **🧮 Safe Feature Engineering:** Calculates repayment stress ratios safely without division-by-zero crashes.
3. **⚡ LightGBM Champion Model:** Fast gradient boosting algorithm optimized for handling imbalanced financial data.
4. **🎯 Probability Calibration (Isotonic Regression):** Converts raw machine learning scores into accurate real-world percentages (e.g., "75% chance of delinquency").
5. **⚙️ Threshold Sweep Tuning:** Finds the optimal cutoff point to catch maximum defaults while minimizing false alarms.
6. **🚦 Business Rule Overlay:** Combines AI model predictions with real-world banking rules for safety.
7. **💡 Reason Codes (SHAP Explainability):** Tells bank officers *why* a customer was flagged (e.g., "High EMI burden + auto-debit bounce").

---

## 📂 Project Structure

```text
├── Credit_card_delinquency_ews_system.ipynb  # Primary Python notebook with end-to-end ML pipeline
├── data analysis.ipynb                       # Exploratory Data Analysis (EDA) & distribution checks
├── File01_Delinquency_ews_Model.csv          # Training dataset (80,000 records)
├── File02_Delinquency_ews_20k_1_Test_Model.csv # Test dataset for threshold tuning (20,000 records)
├── File03_Delinquency_ews_20k_2_Bus_Validate.csv # Business validation dataset (20,000 records)
├── File04_Delinquency_Results_Submit_Final.csv # Final predictions output file
├── DOCUMENTATION.md                          # Detailed technical documentation & system architecture
└── INTERVIEW_NOTES.md                       # Comprehensive guide for job interview prep
```

---

## 🛠️ Tech Stack & Requirements

- **Language:** Python 3.9+
- **Data Wrangling:** `pandas`, `numpy`
- **Machine Learning:** `lightgbm`, `scikit-learn`
- **Model Explainability:** `shap`
- **Data Visualization:** `matplotlib`, `seaborn`

---

## ⚡ How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/-Credit-Delinquency-Early-Warning-System.git
   cd -Credit-Delinquency-Early-Warning-System
   ```

2. **Install required packages:**
   ```bash
   pip install pandas numpy lightgbm scikit-learn shap matplotlib seaborn
   ```

3. **Run the main pipeline notebook:**
   Open and run `Credit_card_delinquency_ews_system.ipynb` in Jupyter Notebook or VS Code. It will train the model, calibrate probabilities, perform threshold tuning, generate SHAP reason codes, and save the final predictions to `File04_Delinquency_Results_Submit_Final.csv`.

---

## 📘 Further Reading & Documentation

- For in-depth technical architecture, mathematical formulas, and internal workflow diagrams, read [DOCUMENTATION.md](file:///c:/Users/ishan/OneDrive/Documents/GITHUB/-Credit-Delinquency-Early-Warning-System/DOCUMENTATION.md).
- Preparing for Data Science, Machine Learning, or Financial Analytics interviews? Check out [INTERVIEW_NOTES.md](file:///c:/Users/ishan/OneDrive/Documents/GITHUB/-Credit-Delinquency-Early-Warning-System/INTERVIEW_NOTES.md).
