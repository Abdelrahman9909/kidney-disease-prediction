# kidney-disease-prediction
kidney disease prediction use machine learning models and application
# 🧠 Kidney Disease Prediction App

This Streamlit-based web application predicts kidney disease conditions using machine learning models. It is designed for healthcare practitioners, patients, and data scientists who want to upload medical data (CSV/Excel), select prediction models, and receive real-time classification results with helpful visualizations.

---

## 📌 Overview

The app classifies kidney disease conditions into five categories:

```python
{
  'High_Risk': 0,
  'Low_Risk': 1,
  'Moderate_Risk': 2,
  'No_Disease': 3,
  'Severe_Disease': 4
}
```

Even if the uploaded file **doesn't contain a 'Target' column**, the app still predicts the condition based on input features by simulating model training.

---

## 🚀 Features

### 📅 Input Options
- Upload medical data files in **CSV or Excel format**
- Enter personal details: **Name**, **Age**, **Email**
- Add a **motivation sentence** for personalized experience

### 🔍 Model Selection
- Choose between:
  - 🧠 **Naive Bayes**
  - 🌲 **Random Forest**
- Models are trained and evaluated using `train_test_split`

### 🔧 Data Preprocessing
- Handles **missing values** via column-wise mean imputation
- Detects and removes **outliers** using **Z-score**
- Applies **Label Encoding** to categorical variables
- Selects top features using **SelectKBest (ANOVA F-test)**
- Applies **Standard Scaling** to numeric features
- Uses **SMOTE** to handle imbalanced datasets

### 📊 Evaluation Metrics
- **Accuracy Score**
- **Classification Report** (Precision, Recall, F1-score)
- **Confusion Matrix** with visual heatmap
- **Label Explanation** appears above the confusion matrix

### 🤖 Model Output
- Final prediction based on selected model
- Personalized user message with predicted **kidney condition**
- Prediction still works if `'Target'` column is missing using simulated training

### 🖼️ Visuals
- Images of kidney anatomy and polycystic kidney disease
- Stylish layout with centered titles and clean user interface

---

## 📁 File Structure

```
kidney-disease-predictor/
│
├── app.py              # Streamlit app code
├── requirements.txt    # Python dependencies
└── README.md           # You're here!
```

---

## 📦 Setup Instructions

### 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/kidney-disease-predictor.git
cd kidney-disease-predictor
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run app.py
```

---

## 💪 Sample Data Format

Your uploaded file should contain numerical and categorical features relevant to kidney disease. Example columns:

| Age | Blood_Pressure | Serum_Creatinine | Albumin | Sugar | Target |
|-----|----------------|------------------|---------|-------|--------|
| 45  | 70             | 1.2              |
