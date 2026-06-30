import gradio as gr
import joblib
import pandas as pd

# ============================================================
# Load trained model
# ============================================================
# Change this filename if yours is different
model = joblib.load("rf_model.joblib")


# ============================================================
# Prediction Function
# ============================================================
def predict_and_advise(age, systolic_bp, diastolic_bp,blood_sugar, body_temp, heart_rate):

    # Create dataframe
    input_data = pd.DataFrame([{
        "Age": age,
        "SystolicBP": systolic_bp,
        "DiastolicBP": diastolic_bp,
        "BS": blood_sugar,
        "BodyTemp": body_temp,
        "HeartRate": heart_rate
    }])

    # ============================================================
    # Feature Engineering (Must Match Training)
    # ============================================================
    input_data["PulsePressure"] = (
        input_data["SystolicBP"] - input_data["DiastolicBP"]
    )

    input_data["MAP"] = (
        input_data["DiastolicBP"]
        + (input_data["SystolicBP"] - input_data["DiastolicBP"]) / 3
    )

    input_data["HighBS"] = (
        input_data["BS"] >= 8.0
    ).astype(int)

    # Predict
    pred_numeric = model.predict(input_data)[0]

    # ============================================================
    # Advice
    # ============================================================
    if pred_numeric == 0:

        status = "🟢 LOW RISK"

        summary = f"""
Vitals appear to fall within generally acceptable ranges for a **{age}-year-old**
pregnant patient.
"""

        suggestions = """
### Recommendations

- Continue attending routine antenatal visits.
- Eat a balanced diet.
- Stay hydrated.
- Exercise safely as recommended by your healthcare provider.
- Continue monitoring blood pressure and blood sugar regularly.
"""

    elif pred_numeric == 1:

        status = "🟡 MID RISK"

        summary = f"""
Some vital signs require closer monitoring.

**Current readings**

- Blood Pressure: **{systolic_bp}/{diastolic_bp} mmHg**
- Blood Sugar: **{blood_sugar} mmol/L**
"""

        suggestions = """
### Recommendations

- Inform your obstetrician or midwife.
- Reduce excess salt and sugar intake.
- Monitor your blood pressure daily.
- Follow all prenatal appointments.
"""

    else:

        status = "🔴 HIGH RISK"

        summary = """
The patient's vital signs indicate a significantly elevated maternal health risk
that requires prompt medical evaluation.
"""

        suggestions = """
### Recommendations

- Seek immediate medical attention.
- Contact your obstetrician immediately.
- Avoid strenuous physical activity.
- Rest while waiting for professional medical care.
- Bring previous medical records if available.
"""

    disclaimer = """
---
### ⚠️ Medical Disclaimer

This application is intended **only for educational and screening purposes**.

It **does not** replace professional medical advice, diagnosis, or treatment.
Always consult a qualified healthcare professional regarding medical concerns.
"""

    return (
        status,
        summary,
        suggestions,
        disclaimer,
        gr.update(visible=True),
    )


# ============================================================
# Interface
# ============================================================

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤰 Maternal Health Risk Assessment")

    gr.Markdown(
        """
Enter the patient's vital signs below to estimate the maternal health risk
using a trained Random Forest model.
"""
    )

    with gr.Row():

        # Left Column
        with gr.Column(scale=1):

            gr.Markdown("### 🩺 Vital Signs Input Form")

            age = gr.Slider(
                minimum=10,
                maximum=70,
                value=25,
                step=1,
                label="Age (years)",
            )

            systolic_bp = gr.Number(
                value=120,
                label="Systolic Blood Pressure (mmHg)",
            )

            diastolic_bp = gr.Number(
                value=80,
                label="Diastolic Blood Pressure (mmHg)",
            )

            blood_sugar = gr.Number(
                value=7.5,
                label="Blood Sugar (mmol/L)",
            )

            body_temp = gr.Number(
                value=98.0,
                label="Body Temperature (°F)",
            )

            heart_rate = gr.Number(
                value=75,
                label="Heart Rate (bpm)",
            )

            submit_btn = gr.Button(
                "Analyze Patient Risk",
                variant="primary",
            )

        # Right Column
        with gr.Column(scale=1):

            with gr.Group(visible=False) as results_container:

                gr.Markdown("### 📊 Assessment Output")

                output_tier = gr.Textbox(label="Predicted Risk", interactive=False,)

                gr.Markdown("---")
                gr.Markdown("### 📋 General Case Information")
                output_summary = gr.Markdown()

                gr.Markdown("### 💡 Recommended Next Steps & Suggestions")
                output_suggestions = gr.Markdown()

                gr.Markdown("---")
                output_disclaimer = gr.Markdown()

    submit_btn.click(
        fn=predict_and_advise,
        inputs=[
            age,
            systolic_bp,
            diastolic_bp,
            blood_sugar,
            body_temp,
            heart_rate,
        ],
        outputs=[
            output_tier,
            output_summary,
            output_suggestions,
            output_disclaimer,
            results_container,
        ],
    )


demo.launch()