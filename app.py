import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Function to load the dataset
def load_data(file_path):
    return pd.read_csv(file_path)

# Function to clean and preprocess the data
def preprocess_data(df):
    print(df.columns)  # Print the columns to debug

    # Only keep relevant columns
    df = df[["Country", "EdLevel", "YearsCodePro", "DevType", "Salary"]]
    
    # Drop rows with missing Salary
    df = df.dropna(subset=['Salary'])

    # Clean YearsCodePro
    def clean_experience(x):
        if x == 'More than 50 years':
            return 50
        if x == 'Less than 1 year':
            return 0.5
        return float(x)

    df['YearsCodePro'] = df['YearsCodePro'].apply(clean_experience)

    # Clean EdLevel
    def clean_education(x):
        if isinstance(x, str):
            if 'Bachelor’s degree' in x:
                return 'Bachelor’s degree'
            if 'Master’s degree' in x:
                return 'Master’s degree'
            if 'Professional degree' in x or 'Other doctoral' in x:
                return 'Post grad'
            return 'Less than a Bachelors'
        return 'Unknown'

    df['EdLevel'] = df['EdLevel'].apply(clean_education)

    # Clean DevType
    df['DevType'] = df['DevType'].apply(lambda x: x.split(';') if isinstance(x, str) else [])
    df = df.explode('DevType').dropna(subset=['DevType'])

    return df

# Load the dataset
file_path = "cleaned_survey_results_public.csv"  # Ensure this matches your uploaded file
df = load_data(file_path)
df = preprocess_data(df)

# Filter the data
df = df[df["Salary"] <= 300000]
df = df[df["Salary"] >= 10000]

# Prepare the data for modeling
X = df.drop("Salary", axis=1)
y = df["Salary"]

# Encode categorical features and scale numerical features
categorical_features = ['Country', 'EdLevel', 'DevType']
numeric_features = ['YearsCodePro']

categorical_pipeline = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numeric_pipeline = Pipeline([
    ('scaler', StandardScaler())
])

transformer = ColumnTransformer(
    transformers=[
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features)
    ]
)

# Train a model
X_transformed = transformer.fit_transform(X)
model = XGBRegressor()
model.fit(X_transformed, y)

# Streamlit application interface
st.title("Advanced Salary Prediction App")

# User input for prediction
st.header("Enter your details for salary prediction")
country = st.selectbox("Country", df["Country"].unique())
ed_level = st.selectbox("Education Level", df["EdLevel"].unique())
years_experience = st.number_input("Years of Professional Coding Experience", min_value=0.0, max_value=50.0, step=0.1)

# Get DevType input
dev_type = st.multiselect("Development Type", df["DevType"].unique())
dev_type_str = '; '.join(dev_type) if dev_type else 'None'

if st.button("Predict Salary"):
    input_data = pd.DataFrame({
        "Country": [country],
        "EdLevel": [ed_level],
        "YearsCodePro": [years_experience],
        "DevType": [dev_type_str]
    })

    # Transform the input data
    input_transformed = transformer.transform(input_data)

    # Make the prediction
    predicted_salary = model.predict(input_transformed)
    st.success(f"Predicted Salary: ${predicted_salary[0]:,.2f}")
