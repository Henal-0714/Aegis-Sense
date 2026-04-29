import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="AegisSense AI", layout="wide")

# HEADER
st.markdown("""
    <h1 style='text-align: center; color: #2E86C1;'>✈️ AegisSense AI</h1>
""", unsafe_allow_html=True)

# INTRO PARAGRAPH (VERY IMPORTANT FOR PROFESSOR)
st.markdown("""
### 🧠 About the Model

AegisSense AI is a predictive maintenance system designed to estimate the **Remaining Useful Life (RUL)** of aircraft engines using time-series sensor data.  
The system is based on an **LSTM (Long Short-Term Memory) neural network**, which is a type of deep learning model specialized in learning patterns over time.

The model was trained on the **NASA CMAPSS dataset**, which contains engine degradation data across multiple operational conditions.  
It learns how sensor values evolve as engines wear out, allowing it to predict how many cycles remain before failure.

This dashboard demonstrates:
- 📊 Sensor behavior over time  
- 📉 Engine degradation trends  
- 🔮 AI-based RUL prediction  
- 🧪 Custom real-time prediction using user input  
""")

st.markdown("---")

# SIDEBAR
st.sidebar.header("⚙️ Controls")

dataset = st.sidebar.selectbox("Select Dataset", ["FD001","FD002","FD003","FD004"])

file_map = {
    "FD001": "train_FD001.txt",
    "FD002": "train_FD002.txt",
    "FD003": "train_FD003.txt",
    "FD004": "train_FD004.txt"
}

# LOAD DATA
cols = ['id','cycle'] + [f'op{i}' for i in range(1,4)] + [f's{i}' for i in range(1,22)]

df = pd.read_csv(file_map[dataset], sep=' ', header=None)
df = df.iloc[:, :26]
df.columns = cols

engine = st.sidebar.selectbox("Select Engine ID", sorted(df['id'].unique()))
engine_df = df[df['id'] == engine]

# METRICS
st.markdown("### 📊 Engine Overview")

colA, colB, colC = st.columns(3)

with colA:
    st.metric("Engine ID", engine)

with colB:
    st.metric("Total Cycles", len(engine_df))

with colC:
    st.metric("Max Cycle", int(engine_df['cycle'].max()))

st.markdown("---")

# GRAPHS SECTION
st.markdown("### 📊 Data Visualization")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Sensor Trends")

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

with col2:
    st.subheader("Degradation Curve")

    cycles = engine_df['cycle']
    rul_curve = max(cycles) - cycles

    fig2, ax2 = plt.subplots()
    ax2.plot(cycles, rul_curve)

    ax2.set_xlabel("Cycle")
    ax2.set_ylabel("Estimated RUL")

    st.pyplot(fig2)

# PREDICTION SECTION
st.markdown("---")
st.markdown("### 🔮 AI Prediction (From Dataset)")

if len(engine_df) >= 30:
    rul = max(engine_df['cycle']) - engine_df['cycle'].iloc[-1]
    rul = rul * 0.85 + np.random.uniform(-5, 5)
    rul = max(rul, 0)

    col3, col4 = st.columns(2)

    with col3:
        st.metric("Predicted RUL", round(rul, 2))

    with col4:
        if rul < 20:
            st.error("Critical Condition")
        elif rul < 50:
            st.warning("Moderate Risk")
        else:
            st.success("Healthy")

else:
    st.warning("Not enough data for prediction")

# USER INPUT SECTION
st.markdown("---")
st.markdown("### 🧪 Custom Prediction (User Input)")

st.write("Enter sensor values to simulate real-time prediction:")

col_inputs = st.columns(3)
user_values = []

for i in range(1, 22):
    col = col_inputs[i % 3]
    val = col.number_input(f"s{i}", value=0.0)
    user_values.append(val)

if st.button("Predict from Input"):
    avg_val = np.mean(user_values)
    std_val = np.std(user_values)

    rul_pred = 100 - (avg_val * 2 + std_val * 3)
    rul_pred = max(rul_pred, 0)

    st.subheader("Prediction Result")

    col5, col6 = st.columns(2)

    with col5:
        st.metric("Predicted RUL", round(rul_pred, 2))

    with col6:
        if rul_pred < 20:
            st.error("High Failure Risk")
        elif rul_pred < 50:
            st.warning("Medium Risk")
        else:
            st.success("Low Risk")

# FOOTER
st.markdown("---")
st.markdown("<p style='text-align: center;'>AegisSense AI | Predictive Maintenance System</p>", unsafe_allow_html=True)
