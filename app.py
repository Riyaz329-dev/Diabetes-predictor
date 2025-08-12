import os
os.environ["GRADIO_USE_SSR"] = "0"

import json
import pandas as pd
import xgboost as xgb
import gradio as gr

MODEL_PATH = "diabetes_xgb_model.json"
GENDER_PATH = "gender_encoder.json"
SMOKE_PATH  = "smoke_encoder.json"

FEATURES = [
    "gender", "age", "hypertension", "heart_disease", "smoking_history",
    "bmi", "HbA1c_level", "blood_glucose_level"
]

def _load_json(path, fallback):
    try:
        with open(path, "r") as f:
            data = json.load(f)
        print(f"[LOAD OK] {path}: type={type(data).__name__}")
        return data
    except Exception as e:
        print(f"[LOAD FAIL] {path}: {e}")
        return fallback

def _to_mapping_and_choices(obj, default_list):
    if isinstance(obj, dict):
        return obj, list(obj.keys())
    if isinstance(obj, list):
        return {cls: i for i, cls in enumerate(obj)}, obj
    return {cls: i for i, cls in enumerate(default_list)}, default_list

# ---- model ----
booster = xgb.Booster()
booster.load_model(MODEL_PATH)
print("[MODEL] Booster loaded")

# ---- encoders ----
gender_raw = _load_json(GENDER_PATH, ["Female", "Male", "Other"])
smoke_raw  = _load_json(SMOKE_PATH,  ["No Info", "current", "ever", "never", "past"])

gender_map, gender_choices = _to_mapping_and_choices(gender_raw, ["Female", "Male", "Other"])
smoke_map,  smoke_choices  = _to_mapping_and_choices(smoke_raw,  ["No Info", "current", "ever", "never", "past"])

def predict_diabetes(gender, age, htn, hd, smoking, bmi, hba1c, glucose):
    row = pd.DataFrame([{
        "gender": gender_map.get(gender, 0),
        "age": age,
        "hypertension": int(htn),
        "heart_disease": int(hd),
        "smoking_history": smoke_map.get(smoking, 0),
        "bmi": bmi,
        "HbA1c_level": hba1c,
        "blood_glucose_level": glucose
    }], columns=FEATURES)

    dmat = xgb.DMatrix(row, feature_names=FEATURES)
    prob = float(booster.predict(dmat)[0])
    label = "Diabetic" if prob >= 0.5 else "Non-Diabetic"
    return label, f"{prob:.3f}"

with gr.Blocks(title="Diabetes Predictor (XGBoost)") as demo:
    gr.Markdown("## Diabetes Prediction (XGBoost)")
    with gr.Row():
        with gr.Column():
            g = gr.Dropdown(gender_choices, value=gender_choices[0], label="Gender")
            age = gr.Slider(1, 100, value=40, step=1, label="Age")
            htn = gr.Dropdown([0, 1], value=0, label="Hypertension (0/1)")
            hd  = gr.Dropdown([0, 1], value=0, label="Heart disease (0/1)")
            smk = gr.Dropdown(smoke_choices, value=smoke_choices[0], label="Smoking history")
            bmi = gr.Slider(10.0, 60.0, value=27.0, step=0.1, label="BMI")
            hba1c = gr.Slider(3.5, 12.0, value=5.8, step=0.1, label="HbA1c (%)")
            glu = gr.Slider(60, 300, value=140, step=1, label="Blood glucose (mg/dL)")
            btn = gr.Button("Predict")
        with gr.Column():
            out_label = gr.Textbox(label="Prediction")
            out_prob  = gr.Textbox(label="Probability of Diabetes")

    btn.click(predict_diabetes, [g, age, htn, hd, smk, bmi, hba1c, glu], [out_label, out_prob])

demo.launch()
