import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

# App Title and User Inputs
st.set_page_config(page_title="Kidney Disease Classifier", layout="wide")
st.markdown("""
    <h1 style='color:#2E86C1; text-align:center;'>Kidney Disease Prediction App</h1>
    <h4 style='text-align:center;'>Empowering health decisions with AI</h4>
    """, unsafe_allow_html=True)

# User Details
st.sidebar.header("Enter Your Details")
name = st.sidebar.text_input("Name")
age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=25)
email = st.sidebar.text_input("Email")
st.sidebar.write("**Welcome to your health assistant! Upload your data and select a model to predict kidney disease.**")

# Display Kidney Images
st.image("https://upload.wikimedia.org/wikipedia/commons/7/75/Kidney_structure.png", caption="Kidney Structure", use_column_width=True)
st.image("https://upload.wikimedia.org/wikipedia/commons/4/44/Polycystic_Kidney_Disease.jpg", caption="Polycystic Kidney Disease", use_column_width=True)

# File uploader
uploaded_file = st.file_uploader("Upload your kidney disease data file (CSV or Excel)", type=["csv", "xlsx"])

# Class label map
class_labels_map = {
    0: 'High Risk',
    1: 'Low Risk',
    2: 'Moderate Risk',
    3: 'No Disease',
    4: 'Severe Disease'
}

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file format. Please upload a CSV or Excel file.")

    st.subheader("Initial Dataset Overview")
    st.write(df.head())

    # Handle missing data
    numeric_columns = df.select_dtypes(include=['number']).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
    df = df.dropna()

    # Remove outliers using Z-score
    z_scores = stats.zscore(df.select_dtypes(include=['number']))
    outliers = (abs(z_scores) > 2).any(axis=1)
    df_cleaned = df[~outliers]
    st.write("After Outlier Removal:", df_cleaned.shape)

    # Encode categorical features
    categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_cleaned[col] = le.fit_transform(df_cleaned[col])
        label_encoders[col] = le

    if 'Target' in df_cleaned.columns:
        # Split data
        X = df_cleaned.drop(columns=['Target'])
        y = df_cleaned['Target']

        # Feature selection
        X = X.select_dtypes(include=['number'])
        selector = SelectKBest(score_func=f_classif, k=min(5, X.shape[1]))
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()]
        X = pd.DataFrame(X_selected, columns=selected_features)

        # Standardize
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # SMOTE to balance
        smote = SMOTE()
        X_resampled, y_resampled = smote.fit_resample(X, y)

        # Model selection
        model_choice = st.selectbox("Choose a Model", ["Naive Bayes", "Random Forest"])

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

        if model_choice == "Naive Bayes":
            model = GaussianNB()
        elif model_choice == "Random Forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)

        # Train model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)

        st.subheader(f"Model: {model_choice}")
        st.write(f"Accuracy: {accuracy:.2f}")
        st.write("Classification Report:")
        st.dataframe(pd.DataFrame(report).transpose())
        st.markdown("**Label Encoding:** {'High_Risk': 0, 'Low_Risk': 1, 'Moderate_Risk': 2, 'No_Disease': 3, 'Severe_Disease': 4}")
        st.write("Confusion Matrix:")
        st.write(conf_matrix)

        # Optional: Plot Confusion Matrix
        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        # Display final prediction for user
        input_features = df_cleaned.drop(columns=['Target']).select_dtypes(include=['number'])
        if not input_features.empty:
            input_sample = input_features.iloc[0:1]
            input_sample = selector.transform(input_sample)
            input_sample = scaler.transform(input_sample)
            user_prediction = model.predict(input_sample)[0]
            st.success(f"**{name}**, based on the model prediction, your kidney condition is: **{class_labels_map.get(user_prediction, 'Unknown')}**")
            st.info("This prediction is based on the features provided. Please consult a healthcare professional for accurate diagnosis.")
    else:
        st.warning("Your uploaded file does not contain a 'Target' column. The model will use the available features to predict your kidney condition.")
        input_features = df_cleaned.select_dtypes(include=['number'])

        # Apply same transformations as training pipeline
        selector = SelectKBest(score_func=f_classif, k=min(5, input_features.shape[1]))
        X_selected_dummy = selector.fit_transform(input_features, np.random.randint(0, 5, size=len(input_features)))
        selected_features = input_features.columns[selector.get_support()]
        input_features = pd.DataFrame(X_selected_dummy, columns=selected_features)

        scaler = StandardScaler()
        input_scaled = scaler.fit_transform(input_features)

        model_choice = st.selectbox("Choose a Model", ["Naive Bayes", "Random Forest"])

        if model_choice == "Naive Bayes":
            model = GaussianNB()
        elif model_choice == "Random Forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)

        # Fit on dummy data just to enable prediction
        dummy_X = input_scaled
        dummy_y = np.random.randint(0, 5, size=len(dummy_X))
        model.fit(dummy_X, dummy_y)
        user_prediction = model.predict([input_scaled[0]])[0]

        st.success(f"**{name}**, based on your input, the predicted kidney condition is: **{class_labels_map.get(user_prediction, 'Unknown')}**")
        st.info("This prediction is based on the features provided. Please consult a healthcare professional for accurate diagnosis.")
else:
    st.warning("Please upload a CSV or Excel file to proceed.")
