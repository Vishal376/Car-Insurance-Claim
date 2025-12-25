import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import requests

MODEL_URL=r'https://huggingface.co/jgvghf/car-insurance-model/blob/main/final_model.joblib'
MODEL_PATH = r"models/final_model.joblib"



os.makedirs("models", exist_ok=True) #if exists then not make again

#Download model if not exists
if not os.path.exists(MODEL_PATH):
    st.info("Downloading from HuggingFace")
    with open(MODEL_PATH,"wb") as f:
        f.write(requests.get(MODEL_URL).content)


# Load trained model
model = joblib.load(MODEL_PATH)

st.set_page_config(page_title="Car Insurance Claim Predictor", layout="wide")
st.title("Car Insurance Claim Predictor 🚗")

st.markdown("""
This app predicts the probability that a customer will make a car insurance claim based on car, policy, and demographic features.
""")

# Sidebar inputs
st.sidebar.header("Policy & Vehicle Features")

policy_tenure = st.sidebar.slider("Policy Tenure (years)", 0, 10, 1)
age_of_car = st.sidebar.slider("Age of Car (years)", 0, 20, 3)
age_of_policyholder = st.sidebar.slider("Age of Policyholder (years)", 18, 80, 35)
population_density = st.sidebar.number_input("Population Density (per km²)", 0, 10000, 1000)
airbags = st.sidebar.slider("Number of Airbags", 0, 10, 2)
ncap_rating = st.sidebar.slider("NCAP Rating (0-5)", 0, 5, 4)

# Categorical / binary
area_cluster = st.sidebar.selectbox("Area Cluster", ["A", "B", "C", "D"])
segment = st.sidebar.selectbox("Car Segment", ["A", "B1", "B2", "C1", "C2"])
fuel_type = st.sidebar.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "Electric"])
engine_type = st.sidebar.selectbox("Engine Type", ["Petrol", "Diesel", "Electric"])
transmission_type = st.sidebar.selectbox("Transmission Type", ["Manual", "Automatic"])
is_esc = st.sidebar.checkbox("ESC Present")
is_tpms = st.sidebar.checkbox("TPMS Present")

# Input DataFrame
input_dict = {
    "policy_tenure": [policy_tenure],
    "age_of_car": [age_of_car],
    "age_of_policyholder": [age_of_policyholder],
    "population_density": [population_density],
    "airbags": [airbags],
    "ncap_rating": [ncap_rating],
    "area_cluster": [area_cluster],
    "segment": [segment],
    "fuel_type": [fuel_type],
    "engine_type": [engine_type],
    "transmission_type": [transmission_type],
    "is_esc": [int(is_esc)],
    "is_tpms": [int(is_tpms)]
}

input_df = pd.DataFrame(input_dict)

# --- ADD MISSING COLUMNS ---
required_cols = ['width','is_rear_window_washer','is_power_door_locks','is_parking_sensors',
                 'is_driver_seat_height_adjustable','max_torque','displacement','is_front_fog_lights',
                 'is_day_night_rear_view_mirror','height','make','is_brake_assist','is_speed_alert',
                 'is_central_locking','is_adjustable_steering','length','is_parking_camera','steering_type',
                 'turning_radius','gross_weight','model','rear_brakes_type','max_power','is_rear_window_wiper',
                 'policy_id','is_ecw','cylinder','is_rear_window_defogger','is_power_steering','gear_box']

for col in required_cols:
    if col not in input_df.columns:
        # You can set default value 0 for numeric, 'Unknown' for categorical
        input_df[col] = 0

# Ensure correct column order (important for model)
input_df = input_df[ list(input_df.columns) + [c for c in required_cols if c not in input_df.columns] ]

# Predict button
if st.button("Predict Claim Probability"):
    prob = model.predict_proba(input_df)[:,1][0]
    st.success(f"Predicted Claim Probability: {prob:.2%}")
