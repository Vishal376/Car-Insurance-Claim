# 🚗 Car Insurance Claim Prediction

End-to-End Machine Learning & Business Intelligence project to predict car insurance claim probability and generate actionable business insights.

---

## 📌 Problem Statement

Insurance companies face significant losses due to unexpected claims.  
The goal of this project is to **predict the probability of an insurance claim** for a policy using historical data and help businesses:

- Identify high-risk policies
- Optimize premium pricing
- Improve operational decision-making

---

## 📂 Dataset

The project uses two datasets:

- **train.csv**
  - Historical insurance data
  - Target column: `is_claim` (0 = No Claim, 1 = Claim)

- **test.csv**
  - Unseen policy data
  - Used for predicting claim probability

---

## 🛠 Tech Stack

- **Python**
- **Pandas, NumPy**
- **Scikit-learn**
- **XGBoost**
- **LightGBM**
- **Power BI**
- **Joblib**

---

## 🔄 Project Workflow

1. Data Loading & Exploration  
2. Feature Categorization  
   - Numerical  
   - Categorical  
   - Boolean  
3. Data Preprocessing  
   - Missing value imputation  
   - Scaling  
   - One-hot encoding  
4. Model Training  
5. Model Evaluation  
6. Final Model Selection  
7. Test Data Prediction  
8. Power BI Dashboard for Business Insights  

---

## 🤖 Machine Learning Models Used

- Logistic Regression (Baseline)
- Decision Tree
- Random Forest
- XGBoost
- LightGBM

👉 Final model selected based on **ROC-AUC and F1-score**, not just accuracy.

---

## 📊 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

---

## 📈 Power BI Business Insights

The dashboard provides:

- **Top risky car segments**
- **Area-wise claim frequency**
- **Impact of safety features on claims**
- **Risk-based policy segmentation for operations**

These insights help underwriting, pricing, and operations teams take data-driven decisions.

---

## 📤 Output

- Trained ML pipeline (`final_model.joblib`)
- Claim probability predictions for test data
- Interactive Power BI dashboard

---

## 🚀 Future Scope

- Real-time API deployment
- Premium recommendation engine
- Fraud detection module
- CRM integration

---

## 👤 Author

**Vishal Singla**  
B.Tech (Computer Science & Engineering)  
Data Analytics & Machine Learning

---

⭐ If you like this project, feel free to star the repository!
