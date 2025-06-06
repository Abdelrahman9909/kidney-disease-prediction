# kidney-disease-prediction
kidney disease prediction use machine learning models and application
# 🧠 Kidney Disease Prediction App

This is an interactive Streamlit application that uses machine learning models to predict kidney disease conditions based on uploaded medical datasets. It allows users to visualize predictions, select different models, and receive personalized predictions based on their input.

---

## 🚀 Features

- 📤 **Upload CSV or Excel Files** for prediction
- 👤 **User Info Input** (Name, Age, Email, Motivation Message)
- 🧠 **Model Selection**: Choose between **Naive Bayes** and **Random Forest**
- ⚖️ **SMOTE** applied to balance imbalanced datasets
- 📊 **Feature Selection** using `SelectKBest`
- 🧼 **Data Preprocessing**: Missing values handled, outliers removed
- 🧾 **Evaluation Metrics**: Accuracy, Classification Report, Confusion Matrix
- 📷 **Visuals**: Includes kidney anatomy and disease images
- 🔍 **Label Mapping Displayed** before Confusion Matrix for clarity:
  
  ```python
  {'High_Risk': 0, 'Low_Risk': 1, 'Moderate_Risk': 2, 'No_Disease': 3, 'Severe_Disease': 4}
