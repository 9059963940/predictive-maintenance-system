import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# Page setup
st.set_page_config(page_title="Predictive Maintenance System", layout="centered")

st.title("🔧 Predictive Maintenance System (Wipro Ready Project)")

# Load model
model = joblib.load("model.pkl")
features = joblib.load("features.pkl")

# Input UI
st.subheader("Enter Machine Parameters")

air_temp = st.number_input("Air Temperature")
process_temp = st.number_input("Process Temperature")
rot_speed = st.number_input("Rotational Speed")
torque = st.number_input("Torque")
tool_wear = st.number_input("Tool Wear")

# Prediction
if st.button("Predict Failure"):

    input_dict = {
        "Air temperature": air_temp,
        "Process temperature": process_temp,
        "Rotational speed": rot_speed,
        "Torque": torque,
        "Tool wear": tool_wear
    }

    input_data = pd.DataFrame([input_dict])
    input_data = input_data.reindex(columns=features, fill_value=0)

    prob_raw = model.predict_proba(input_data)[0][1]

# 🔥 CALIBRATION FIX
    prob = prob_raw * 100 
    if rot_speed > 2000 or tool_wear > 150 or torque > 70:
    prob += 25    
# 🔥 GAUGE MUST BE ALIGNED LIKE THIS
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob,
        title={'text': "Failure Risk (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'steps': [
                {'range': [0, 40], 'color': "green"},
                {'range': [40, 70], 'color': "orange"},
                {'range': [70, 100], 'color': "red"}
            ]
        }
    ))

    st.plotly_chart(fig)

    if prob > 70:
        st.error("🔴 High Risk")
    elif prob > 40:
        st.warning("🟠 Medium Risk")
    else:
        st.success("🟢 Safe")

    st.write(f"Failure Probability: {prob:.2f}%")
