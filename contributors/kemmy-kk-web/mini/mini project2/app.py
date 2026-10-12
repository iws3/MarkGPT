import pickle

import numpy as np
import streamlit as st
from tensorflow import keras
import shap
import matplotlib.pyplot as plt


# Page config

st.set_page_config(page_title="Heart Disease Risk Predictor", page_icon="❤️", layout="centered")

Model_path = "Heart_Disease.keras"
Kit_path = "Heart_Disease_kit.pkl"



# Load model + preprocessing kit (cached so it only loads once)

@st.cache_resource
def load_model_and_kit():
    model = keras.models.load_model(Model_path)
    with open(Kit_path, "rb") as f:
        kit = pickle.load(f)
    return model, kit


model, kit = load_model_and_kit()

scaler_mean = np.array(kit["scaler_mean"], dtype=np.float32)
scaler_scale = np.array(kit["scaler_scale"], dtype=np.float32)
feature_cols = kit["feature_cols"]
train_medians = kit["train_medians"]


@st.cache_resource
def get_explainer(_model, _scaler_mean, _scaler_scale, _feature_cols, _train_medians):
    background_raw = np.array([[_train_medians[col] for col in _feature_cols]], dtype=np.float32)
    background_scaled = (background_raw - _scaler_mean) / _scaler_scale
    background = np.repeat(background_scaled, 50, axis=0)
    explainer = shap.DeepExplainer(_model, background)
    return explainer


explainer = get_explainer(model, scaler_mean, scaler_scale, feature_cols, train_medians)



LEVEL_LABELS = {0: "Low", 1: "Medium", 2: "High"}
LEVEL_VALUES = {v: k for k, v in LEVEL_LABELS.items()}
ORDINAL_FEATURES = {"Smoking", "Alcohol_Intake", "Physical_Activity", "Diet", "Stress_Level"}
BINARY_FEATURES = {
    "Gender",
    "Hypertension",
    "Diabetes",
    "Hyperlipidemia",
    "Family_History",
    "Previous_Heart_Attack",
}



# UI

st.title("❤️ Heart Disease Risk Predictor")
st.caption(
    "Enter patient details below. This tool uses a trained neural network "
    "and is for educational/demo purposes only_ not a medical diagnosis."
)
#patient information-predict-trained model-prediction

# Age is asked outside the form so the page can react to it immediately
# (widgets inside st.form only update once the form is submitted).
age = st.number_input("Age", min_value=1, max_value=120, value=int(train_medians["Age"]))

if age <= 13:
    st.info(
        "This tool is designed to assess adult cardiovascular risk factors "
        "(smoking, blood pressure, cholesterol, etc.) and is not applicable "
        "to children. Please enter an age of 14 or above to continue."
    )
    st.stop()

with st.form("patient_form"):
    st.subheader("Demographics")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", options=["Female", "Male"], index=0)
    with col2:
        weight = st.number_input("Weight (kg)", min_value=20.0, max_value=300.0, value=float(train_medians["Weight"]))
        height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=float(train_medians["Height"]))

    bmi_default = float(train_medians["BMI"])
    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=70.0,
        value=bmi_default,
        help="Auto-calculated suggestion updates only on next run; adjust manually if needed.",
    )

    st.subheader("Lifestyle")
    col3, col4 = st.columns(2)
    with col3:
        smoking = st.selectbox("Smoking level", options=["Low", "Medium", "High"], index=int(train_medians["Smoking"]))
        alcohol = st.selectbox("Alcohol intake", options=["Low", "Medium", "High"], index=int(train_medians["Alcohol_Intake"]))
    with col4:
        activity = st.selectbox("Physical activity", options=["Low", "Medium", "High"], index=int(train_medians["Physical_Activity"]))
        diet = st.selectbox("Diet quality", options=["Low", "Medium", "High"], index=int(train_medians["Diet"]))

    stress = st.selectbox("Stress level", options=["Low", "Medium", "High"], index=int(train_medians["Stress_Level"]))

    st.subheader("Medical History")
    col5, col6 = st.columns(2)
    with col5:
        hypertension = st.checkbox("Hypertension", value=bool(train_medians["Hypertension"]))
        diabetes = st.checkbox("Diabetes", value=bool(train_medians["Diabetes"]))
        hyperlipidemia = st.checkbox("Hyperlipidemia", value=bool(train_medians["Hyperlipidemia"]))
    with col6:
        family_history = st.checkbox("Family history of heart disease", value=bool(train_medians["Family_History"]))
        previous_attack = st.checkbox("Previous heart attack", value=bool(train_medians["Previous_Heart_Attack"]))

    st.subheader("Vitals & Labs")
    col7, col8 = st.columns(2)
    with col7:
        systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=60, max_value=260, value=int(train_medians["Systolic_BP"]))
        diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=30, max_value=160, value=int(train_medians["Diastolic_BP"]))
    with col8:
        heart_rate = st.number_input("Heart rate (bpm)", min_value=30, max_value=220, value=int(train_medians["Heart_Rate"]))
        blood_sugar = st.number_input("Fasting blood sugar (mg/dL)", min_value=40, max_value=500, value=int(train_medians["Blood_Sugar_Fasting"]))

    cholesterol = st.number_input(
        "Total cholesterol (mg/dL)", min_value=80, max_value=500, value=int(train_medians["Cholesterol_Total"])
    )

    submitted = st.form_submit_button("Predict", use_container_width=True)



# Prediction

if submitted:
    raw = {
        "Age": age,
        "Gender": 1 if gender == "Male" else 0,
        "Weight": weight,
        "Height": height,
        "BMI": bmi,
        "Smoking": LEVEL_VALUES[smoking],
        "Alcohol_Intake": LEVEL_VALUES[alcohol],
        "Physical_Activity": LEVEL_VALUES[activity],
        "Diet": LEVEL_VALUES[diet],
        "Stress_Level": LEVEL_VALUES[stress],
        "Hypertension": int(hypertension),
        "Diabetes": int(diabetes),
        "Hyperlipidemia": int(hyperlipidemia),
        "Family_History": int(family_history),
        "Previous_Heart_Attack": int(previous_attack),
        "Systolic_BP": systolic_bp,
        "Diastolic_BP": diastolic_bp,
        "Heart_Rate": heart_rate,
        "Blood_Sugar_Fasting": blood_sugar,
        "Cholesterol_Total": cholesterol,
    }

    # Build the feature vector in the exact order the model was trained on
    x = np.array([[raw[col] for col in feature_cols]], dtype=np.float32)

    # Apply the SAME scaling used during training
    x_scaled = (x - scaler_mean) / scaler_scale

    prob = float(model.predict(x_scaled, verbose=0).flatten()[0])
    pred_label = "Heart Disease" if prob >= 0.5 else "No Heart Disease"

    st.divider()
    st.subheader("Result")

    if prob >= 0.5:
        st.error(f"⚠️ Prediction: **{pred_label}**")
    else:
        st.success(f"✅ Prediction: **{pred_label}**")

    st.metric("Predicted probability of heart disease", f"{prob * 100:.1f}%")
    st.progress(min(max(prob, 0.0), 1.0))

    # SHAP explanation 
    st.subheader("Why this prediction?")
    st.caption(
        "This chart shows how much each factor pushed the predicted risk "
        "up (red) or down (green) for this specific patient, relative to a "
        "typical baseline patient."
    )

    with st.spinner("Computing explanation..."):
        shap_values = explainer.shap_values(x_scaled)
        # For a single-output sigmoid model, shap_values is often a list
        # with one array in it; handle both cases defensively.
        sv = shap_values[0] if isinstance(shap_values, list) else shap_values
        sv = np.array(sv).flatten()

        def to_scalar(v):
            # Handles plain floats, numpy arrays/scalars, lists, and
            # TensorFlow tensors, converting any of them to a single float.
            if isinstance(v, (list, tuple)):
                v = v[0]
            if hasattr(v, "numpy"):
                v = v.numpy()
            v = np.array(v).flatten()
            return float(v[0])

        base_value = to_scalar(explainer.expected_value)

        explanation = shap.Explanation(
            values=sv,
            base_values=base_value,
            data=x[0],
            feature_names=feature_cols,
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        shap.plots.bar(explanation, show=False)
        st.pyplot(fig)
        plt.close(fig)

    st.warning(
        "⚠️ This prediction is generated by a machine learning model for demonstration "
        "purposes only. It is not a substitute for professional medical advice, diagnosis, "
        "or treatment. Always consult a qualified healthcare provider."
    )