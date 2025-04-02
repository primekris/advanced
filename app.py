import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load pre-trained models and scalers
heart_classifier = joblib.load('knn_model_heart.joblib')

liver_classifier = joblib.load('knn_model_liver.joblib')

scaler_heart = joblib.load('scaler_heart.joblib')
scaler_liver = joblib.load('scaler_liver.joblib')

# Function for heart disease prediction
def predict_heart_disease(data):
    sample_data = pd.DataFrame([data])
    scaled_data = scaler_heart.transform(sample_data)
    pred = heart_classifier.predict(scaled_data)[0]
    prob = np.max(heart_classifier.predict_proba(scaled_data)[0])
    return pred, prob



# Function for liver disease prediction
def predict_liver_disease(data):
    sample_data = pd.DataFrame([data])
    scaled_data = scaler_liver.transform(sample_data)
    pred = liver_classifier.predict(scaled_data)[0]
    prob = np.max(liver_classifier.predict_proba(scaled_data)[0])
    return pred, prob

# Streamlit UI
st.title("Health Disease Prediction App")

# Select Prediction Model
model_choice = st.selectbox("Select Prediction Model", ["Heart Disease Prediction","Liver Disease Prediction"])

# Input fields for Heart Disease Prediction
if model_choice == "Heart Disease Prediction":
    st.header("Heart Disease Prediction")
    age = st.number_input("Age", min_value=29, max_value=80, value=29)
    sex = st.selectbox("Sex", [0, 1])
    cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=94, max_value=200, value=94)
    chol = st.number_input("Cholesterol (chol)", min_value=127, max_value=600, value=127)
    fbs = st.selectbox("Fasting Blood Sugar (fbs)", [0, 1])
    restecg = st.selectbox("Resting Electrocardiographic Results (restecg)", [0, 1, 2])
    thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", min_value=71, max_value=250, value=71)
    exang = st.selectbox("Exercise Induced Angina (exang)", [0, 1])
    oldpeak = st.number_input("Depression Induced by Exercise (oldpeak)", min_value=0.0, max_value=10.0, value=0.0)
    slope = st.selectbox("Slope of Peak Exercise ST Segment (slope)", [0, 1, 2])
    ca = st.number_input("Number of Major Vessels Colored by Fluoroscopy (ca)", min_value=0, max_value=4, value=0)
    thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])

    heart_input_data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }

    if st.button("Predict Heart Disease"):
        pred, prob = predict_heart_disease(heart_input_data)
        if pred == 1:
            st.error(f"Prediction: Heart Disease detected with probability {prob:.2f}")
        else:
            st.success(f"Prediction: No Heart Disease detected with probability {prob:.2f}")



# Input fields for Liver Disease Prediction
elif model_choice == "Liver Disease Prediction":
    st.header("Liver Disease Prediction")
    # Input fields for each parameter
    Age = st.number_input("Age", min_value=1, max_value=100, value=50, step=1)
    Gender = st.selectbox("Gender", ["Male", "Female"], index=0)
    Total_Bilirubin = st.number_input("Total_Bilirubin", min_value=0, max_value=75, value=0, step=1)
    Direct_Bilirubin = st.number_input("Direct_Bilirubin", min_value=0, max_value=20, value=0, step=1)
    Alkaline_Phosphotase = st.number_input("Alkaline_Phosphotase", min_value=60, max_value=2150, value=1000, step=10)
    Alamine_Aminotransferase = st.number_input("Alamine_Aminotransferase", min_value=10, max_value=2000, value=100, step=10)
    Aspartate_Aminotransferase = st.number_input("Aspartate_Aminotransferase", min_value=10.0, max_value=5000.0, value=1000.0, step=10.1)
    Total_Protiens = st.number_input("Total_Protiens", min_value=2.0, max_value=10.00, value=5.0, step=0.1)
    Albumin = st.number_input("Albumin", min_value=0, max_value=6, value=0, step=1)
    Albumin_and_Globulin_Ratio = st.number_input("Albumin_and_Globulin_Ratio", min_value=1, max_value=2, value=1, step=1)

    Gender_map = {'Male': 0, 'Female': 1} 

    # Create the input dictionary for prediction
    liver_input_data = {
    'Age': Age,
    'Gender': Gender_map[Gender],
    'Total_Bilirubin': Total_Bilirubin,
    'Direct_Bilirubin': Direct_Bilirubin,
    'Alkaline_Phosphotase': Alkaline_Phosphotase,
    'Alamine_Aminotransferase': Alamine_Aminotransferase,
    'Aspartate_Aminotransferase': Aspartate_Aminotransferase,
    'Total_Protiens': Total_Protiens,
    'Albumin': Albumin,
    'Albumin_and_Globulin_Ratio': Albumin_and_Globulin_Ratio
    }

    # When the user clicks the "Predict" button
    if st.button("Predict"):
        with st.spinner('Making prediction...'):
            pred, prob = predict_liver_disease(liver_input_data)
            if pred == 1:
                st.error(f"Prediction: Liver Disease detected with probability {prob:.2f}")
            else:
                st.success(f"Prediction: No Liver Disease detected with probability {prob:.2f}")




# KRISHNA 
