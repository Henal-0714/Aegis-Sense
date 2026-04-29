import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Page setup
st.set_page_config(page_title="AegisSense AI", layout="wide")

# Title
st.markdown("<h1 style='text-align: center;'>✈️ AegisSense AI Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Predictive Maintenance using LSTM</p>", unsafe_allow_html=True)

# Load model
model = load_model("model.h5")
scaler = joblib.load("scaler.save")

# Sidebar
st.sidebar.header("Controls")

dataset = st.sidebar.selectbox("Select Dataset", ["FD001","FD002","FD003","FD004"])

file_map = {
    "FD001": "train_FD001.txt",
    "FD002": "train_FD002.txt",
    "FD003": "train_FD003.txt",
    "FD004": "train_FD004.txt"
}

# Load dataset
cols = ['id','cycle'] + [f'op{i}' for i in range(1,4)] + [f's{i}' for i in range(1,22)]

df = pd.read_csv(file_map[dataset], sep=' ', header=None)
df = df.iloc[:, :26]
df.columns = cols

# Engine selector
engine = st.sidebar.selectbox("Select Engine ID", sorted(df['id'].unique()))
engine_df = df[df['id'] == engine]

# Layout
col1, col2 = st.columns(2)

# GRAPH 1
with col1:
    st.subheader("📊 Multi-Sensor Trends")

    sensors = st.multiselect(
        "Select Sensors",
        [f's{i}' for i in range(1,22)],
        default=['s1','s2','s3']
    )

    fig, ax = plt.subplots()

    for s in sensors:
        ax.plot(engine_df['cycle'], engine_df[s], label=s)

    ax.set_xlabel("Cycle")
    ax.set_ylabel("Sensor Value")
    ax.legend()

    st.pyplot(fig)

# GRAPH 2
with col2:
    st.subheader("📉 Degradation Curve")

    cycles = engine_df['cycle']
    rul_curve = max(cycles) - cycles

    fig2, ax2 = plt.subplots()
    ax2.plot(cycles, rul_curve, color='red')

    ax2.set_xlabel("Cycle")
    ax2.set_ylabel("Estimated RUL")

    st.pyplot(fig2)

# Prediction
st.markdown("---")
st.subheader("🔮 AI Prediction")

if len(engine_df) >= 30:
    latest = engine_df.tail(30)

    X = latest.drop(['id','cycle'], axis=1)
    X_scaled = scaler.transform(X)
    X_scaled = X_scaled.reshape(1,30,X_scaled.shape[1])

    pred = model.predict(X_scaled)
    rul = float(pred[0][0])

    col3, col4, col5 = st.columns(3)

    with col3:
        st.metric("Predicted RUL", round(rul,2))

    with col4:
        if rul < 20:
            st.error("🔴 Critical")
        elif rul < 50:
            st.warning("🟡 Warning")
        else:
            st.success("🟢 Healthy")

    with col5:
        st.metric("Cycles Observed", len(engine_df))

else:
    st.warning("Not enough cycles for prediction")

# Info
st.markdown("---")
st.subheader("🧠 About Model")

st.write("""
This application uses an LSTM (Long Short-Term Memory) model trained on NASA CMAPSS dataset.
It predicts Remaining Useful Life (RUL) of aircraft engines.
""")
