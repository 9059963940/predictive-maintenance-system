import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="Predictive Maintenance", layout="centered")

st.title("🔧 AI Predictive Maintenance System")

# Load model
model = joblib.load("model.pkl")
features = joblib.load("features.pkl")

st.subheader("Enter Machine Parameters")

air_temp = st.number_input("Air Temperature")
process_temp = st.number_input("Process Temperature")
rot_speed = st.number_input("Rotational Speed")
torque = st.number_input("Torque")
tool_wear = st.number_input("Tool Wear")

if st.button("Predict Failure"):

    input_data = pd.DataFrame([{
        "Air temperature": air_temp,
        "Process temperature": process_temp,
        "Rotational speed": rot_speed,
        "Torque": torque,
        "Tool wear": tool_wear
    }])

    input_data = input_data.reindex(columns=features, fill_value=0)

    prob = model.predict_proba(input_data)[0][1] * 100

    st.subheader("Risk Meter")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob,
        gauge={
            "axis": {"range": [0, 100]},
            "steps": [
                {"range": [0, 40], "color": "green"},
                {"range": [40, 70], "color": "orange"},
                {"range": [70, 100], "color": "red"}
            ]
        }
    ))

    st.plotly_chart(fig)

    if prob > 70:
        st.error("🔴 High Risk of Failure")
    elif prob > 40:
        st.warning("🟠 Medium Risk")
    else:
        st.success("🟢 Low Risk")

    st.write(f"Failure Probability: {prob:.2f}%")
