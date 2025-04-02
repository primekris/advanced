import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load pre-trained models and scalers
heart_classifier = joblib.load('knn_model_heart.joblib')
kidney_classifier = joblib.load('knn_model_kidney.joblib')
hypertension_classifier = joblib.load('knn_model_hypertension.joblib')
diabetes_classifier = joblib.load('knn_model_diabetes.joblib')
liver_classifier = joblib.load('knn_model_liver.joblib')

scaler_heart = joblib.load('scaler_heart.joblib')
scaler_kidney = joblib.load('scaler_kidney.joblib')
scaler_hypertension = joblib.load('scaler_hypertension.joblib')
scaler_diabetes = joblib.load('scaler_diabetes.joblib')
scaler_liver = joblib.load('scaler_liver.joblib')

# Function for heart disease prediction
def predict_heart_disease(data):
    sample_data = pd.DataFrame([data])
    scaled_data = scaler_heart.transform(sample_data)
    pred = heart_classifier.predict(scaled_data)[0]
    prob = np.max(heart_classifier.predict_proba(scaled_data)[0])
    return pred, prob

# Function for kidney disease prediction
def predict_kidney_disease(data):
    sample_data = pd.DataFrame([data])
    scaled_data = scaler_kidney.transform(sample_data)
    pred = kidney_classifier.predict(scaled_data)[0]
    prob = np.max(kidney_classifier.predict_proba(scaled_data)[0])
    return pred, prob

# Function for hypertension risk prediction
def predict_hypertension(data):
    sample_data = pd.DataFrame([data])
    scaled_data = scaler_hypertension.transform(sample_data)
    pred = hypertension_classifier.predict(scaled_data)[0]
    prob = np.max(hypertension_classifier.predict_proba(scaled_data)[0])
    return pred, prob

# Function for diabetes prediction
def predict_diabetes(data):
    sample_data = pd.DataFrame([data])
    scaled_data = scaler_diabetes.transform(sample_data)
    pred = diabetes_classifier.predict(scaled_data)[0]
    prob = diabetes_classifier.predict_proba(scaled_data)[0][pred]
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
model_choice = st.selectbox("Select Prediction Model", ["Heart Disease Prediction","Kidney Disease Prediction","Hypertension Risk Prediction","Diabetes Outcome Prediction","Liver Disease Prediction"])

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

# Input fields for Kidney Disease Prediction
elif model_choice == "Kidney Disease Prediction":
    st.header("Kidney Disease Prediction")
    age = st.number_input("age",min_value=0, max_value=100, value=50, step=1)
    bp = st.number_input("bp", min_value=50.0, max_value=180.0, value=70.0, step=0.1)
    sg = st.number_input("sg", min_value=1.0, max_value=2.0, value=1.5, step=0.1)
    al = st.number_input("al", min_value=0, max_value=5, value=1)
    su = st.number_input("su", min_value=0, max_value=5, value=0)
    rbc = st.selectbox("rbc",["normal", "abnormal"], index=0)
    pc = st.selectbox("pc", ["normal", "abnormal"], index=0)
    pcc = st.selectbox("pcc",["notpresent", "present"], index=0)
    ba = st.selectbox("ba", ["notpresent", "present"], index=0)
    bgr = st.number_input("bgr", min_value=20.0, max_value=490.0, value=50.0, step=0.1)
    bu = st.number_input("bu", min_value=1, max_value=400, value=1)
    sc = st.number_input("sc", min_value=0, max_value=80, value=0)
    sod = st.number_input("sod", min_value=4.0, max_value=170.0, value=5.0, step=0.1)
    pot = st.number_input("pot",min_value=1.0, max_value=50.0, value=10.5, step=1.0)
    hemo = st.number_input("hemo", min_value=3.0, max_value=20.0, value=10.5, step=1.0)
    pcv = st.number_input("pcv", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    wc = st.number_input("wc", min_value=0.0, max_value=50000.0, value=5000.0, step=0.1)
    rc = st.number_input("rc", min_value=0, max_value=10, value=1)
    htn = st.selectbox("htn", ["no", "yes"], index=0)
    cad = st.selectbox("cad", ["no", "yes"], index=0)
    appet = st.selectbox("appet", ["good", "poor"], index=0)
    pe = st.selectbox("pe",["no", "yes"], index=0)
    ane = st.selectbox("ane", ["no", "yes"], index=0)

    rbc_map={'normal':0,'abnormal':1}
    pc_map={'normal':0,'abnormal':1}
    pcc_map={'notpresent':0,'present':1}
    ba_map={'notpresent':0,'present':1}
    htn_map={'no':0,'yes':1}
    cad_map={'no':0,'yes':1}
    appet_map={'good':0,'poor':1}
    pe_map={'no':0,'yes':1}
    ane_map={'no':0,'yes':1}

    kidney_input_data= {'age':age,
    'bp':bp,
    'sg':sg,
    'al':al,
    'su':su,
    'rbc':rbc_map[rbc],
    'pc':pc_map[pc],
    'pcc':pcc_map[pcc],
    'ba':ba_map[ba],
    'bgr':bgr,
    'bu':bu,
    'sc':sc,
    'sod':sod,
    'pot':pot,
    'hemo':hemo,
    'pcv':pcv,
    'wc':wc,
    'rc':rc,
    'htn':htn_map[htn],
    'cad':cad_map[cad],
    'appet':appet_map[appet],
    'pe':pe_map[pe],
    'ane':ane_map[ane]
    }

    if st.button("Predict"):
        with st.spinner('Making prediction...'):
            pred, prob = predict_kidney_disease(kidney_input_data)
            if pred == 1:
                st.error(f"Prediction:  Kidney Disease detected with probability {prob:.2f}")
            else:
                st.success(f"Prediction: No Kidney Disease detected with probability {prob:.2f}")

# Input fields for Hypertension Risk Prediction
elif model_choice == "Hypertension Risk Prediction":
    st.header("Hypertension Risk Prediction")
    male = st.number_input("male", min_value=0, max_value=1, value=0, step=1)
    age = st.number_input("age", min_value=32, max_value=70, value=40, step=1)
    currentSmoker = st.number_input("currentSmoker", min_value=0, max_value=1, value=0, step=1)
    cigsPerDay = st.number_input("cigsPerDay", min_value=0, max_value=70, value=0, step=1)
    BPMeds = st.number_input("BPMeds", min_value=0, max_value=1, value=0, step=1)
    diabetes = st.number_input("diabetes", min_value=0, max_value=1, value=0, step=1)
    totChol = st.number_input("totChol", min_value=100.0, max_value=700.0, value=200.0, step=1.0)
    sysBP = st.number_input("sysBP", min_value=80.0, max_value=300.00, value=100.0, step=1.1)
    diaBP = st.number_input("diaBP", min_value=40.0, max_value=150.00, value=100.0, step=1.1)
    BMI = st.number_input("BMI", min_value=10, max_value=60, value=20, step=1)
    heartRate = st.number_input("heartRate", min_value=40, max_value=150, value=60, step=1)
    glucose = st.number_input("glucose", min_value=40, max_value=400, value=70, step=1)

    # Create the input dictionary for prediction
    hypertension_input_data = {
    'male': male,
    'age': age,
    'currentSmoker': currentSmoker,
    'cigsPerDay': cigsPerDay,
    'BPMeds': BPMeds,
    'diabetes': diabetes,
    'totChol': totChol,
    'sysBP':sysBP,
    'diaBP': diaBP,
    'BMI': BMI,
    'heartRate': heartRate,
    'glucose': glucose
    }

    if st.button("Predict"):
        with st.spinner('Making prediction...'):
            pred, prob = predict_hypertension(hypertension_input_data)
            if pred == 1:
                st.error(f"Prediction: Hypertension Risk detected with probability {prob:.2f}")
            else:
                st.success(f"Prediction: No Hypertension Risk detected with probability {prob:.2f}")

# Input fields for Diabetes Outcome Prediction
elif model_choice == "Diabetes Outcome Prediction":
    st.header("Diabetes Outcome Prediction")
    Pregnancies = st.number_input("Pregnancies", min_value=0.0, max_value=20.0, value=10.0)
    Glucose = st.number_input("Glucose", min_value=0.0, max_value=200.0, value=100.0)
    BloodPressure = st.number_input("Blood Pressure", min_value=0.0, max_value=150.0, value=100.0)
    SkinThickness = st.number_input("Skin Thickness", min_value=0.0, max_value=100.0, value=15.0)
    Insulin = st.number_input("Insulin", min_value=0, max_value=1000, value=79)
    BMI = st.number_input("BMI", min_value=0.0, max_value=100.0, value=31.25)
    DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=5.0, value=1.0)
    Age = st.number_input("Age", min_value=1, max_value=100, value=50, step=1)

    diabetes_input_data = {
        'Pregnancies': Pregnancies,
        'Glucose': Glucose,
        'BloodPressure': BloodPressure,
        'SkinThickness': SkinThickness,
        'Insulin': Insulin,
        'BMI': BMI,
        'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
        'Age': Age
    }

    if st.button("Predict Diabetes Outcome"):
        pred, prob = predict_diabetes(diabetes_input_data)
        if pred == 1:
            st.error(f"Prediction: Diabetes detected with probability {prob:.2f}")
        else:
            st.success(f"Prediction: No Diabetes detected with probability {prob:.2f}")

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
