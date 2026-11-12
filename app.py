import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------
st.set_page_config(
    page_title="Predictive Maintenance | AI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling (the matching Streamlit theme is in .streamlit/config.toml)
st.markdown(
    """
    <style>
      .stApp { background: linear-gradient(135deg, #0b1220 0%, #101b2d 55%, #111827 100%); }
      [data-testid="stHeader"] { background: rgba(0,0,0,0); }
      [data-testid="stSidebar"] {
          background: linear-gradient(180deg, #111c2e 0%, #17263b 100%);
          border-right: 1px solid #263b55;
      }
      .hero {
          padding: 1.35rem 1.6rem; border-radius: 18px;
          background: linear-gradient(110deg, #153451, #174b55);
          border: 1px solid #2a6471; margin-bottom: 1.2rem;
      }
      .hero h1 { color: #f3f8ff; margin: 0; font-size: 2.1rem; }
      .hero p { color: #c8dce9; margin: .45rem 0 0 0; }
      .metric-card {
          background: #17263b; border: 1px solid #2a405c;
          border-radius: 14px; padding: 1rem 1.1rem;
      }
      .metric-label { color: #a9bfd4; font-size: .9rem; }
      .metric-value { color: #f5f9ff; font-size: 1.65rem; font-weight: 700; }
      div[data-testid="stMetric"] {
          background: #17263b; border: 1px solid #2a405c;
          padding: 14px 16px; border-radius: 14px;
      }
      div[data-testid="stMetricLabel"] { color: #b8cbe0; }
      div[data-testid="stMetricValue"] { color: #f5f9ff; }
      .stButton > button {
          border: 0; border-radius: 10px; font-weight: 700;
          background: linear-gradient(90deg, #0ea5a4, #0d9488);
          color: white; padding: .65rem 1.2rem;
      }
      .stButton > button:hover { background: #0f766e; color: white; }
      .result-box {
          padding: 1.1rem 1.3rem; border-radius: 14px;
          border: 1px solid #35516d; background: #17263b;
      }
      .small-note { color: #a9bfd4; font-size: .86rem; }
      h2, h3 { color: #e8f2ff; }
    </style>
    """,
    unsafe_allow_html=True,
)

MODEL_CANDIDATES = [
    "xgboost_predictive_maintenance.pkl",
    "predictive_maintenance_xgboost.pkl",
]

@st.cache_resource
def load_model():
    for model_path in MODEL_CANDIDATES:
        if os.path.exists(model_path):
            return joblib.load(model_path), model_path
    raise FileNotFoundError(
        "Model file not found. Keep xgboost_predictive_maintenance.pkl "
        "or predictive_maintenance_xgboost.pkl in the same folder as app.py."
    )

try:
    model, loaded_model_path = load_model()
except Exception as exc:
    st.error(f"Unable to load the trained model: {exc}")
    st.stop()

# These names and their order match the XGBoost model trained in the notebook.
FEATURES = [
    "Air_temperature_K",
    "Process_temperature_K",
    "Rotational_speed_rpm",
    "Torque_Nm",
    "Tool_wear_min",
    "Type_L",
    "Type_M",
]

# ------------------------------------------------------------
# Sidebar inputs
# ------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Machine settings")
    st.caption("Enter current machine sensor readings.")
    machine_type = st.selectbox("Machine quality type", ["L", "M", "H"], index=0)
    air_temp = st.number_input("Air temperature (K)", min_value=250.0, max_value=350.0, value=300.0, step=0.1)
    process_temp = st.number_input("Process temperature (K)", min_value=250.0, max_value=400.0, value=310.0, step=0.1)
    rotational_speed = st.number_input("Rotational speed (RPM)", min_value=0, max_value=10000, value=1500, step=10)
    torque = st.number_input("Torque (Nm)", min_value=0.0, max_value=200.0, value=40.0, step=0.5)
    tool_wear = st.number_input("Tool wear (minutes)", min_value=0, max_value=500, value=100, step=1)
    predict_clicked = st.button("🔍 Predict machine condition", use_container_width=True)

# ------------------------------------------------------------
# Main page
# ------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <h1>⚙️ Predictive Maintenance</h1>
      <p>Machine failure prediction powered by your trained XGBoost model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

overview1, overview2, overview3 = st.columns(3)
overview1.metric("Model", "XGBoost")
overview2.metric("Input features", "7")
overview3.metric("Model file", "Loaded ✓")

st.markdown("### Machine condition analysis")
st.write(
    "Adjust the sensor values in the sidebar and select **Predict machine condition** "
    "to estimate whether the machine is likely to fail."
)

if predict_clicked:
    input_row = pd.DataFrame(
        [[
            air_temp,
            process_temp,
            rotational_speed,
            torque,
            tool_wear,
            1 if machine_type == "L" else 0,
            1 if machine_type == "M" else 0,
        ]],
        columns=FEATURES,
    )

    try:
        prediction = int(model.predict(input_row)[0])
        probability = None
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(input_row)[0][1])

        st.markdown("### Prediction result")
        result_col, probability_col = st.columns([1.25, 1])

        with result_col:
            if prediction == 1:
                st.error("⚠️ Machine failure predicted")
                st.markdown(
                    '<div class="result-box"><b>Recommended action</b><br>'
                    'Schedule an inspection and check the machine before continuing operation.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.success("✅ No machine failure predicted")
                st.markdown(
                    '<div class="result-box"><b>Recommended action</b><br>'
                    'Continue normal operation and keep monitoring the sensor readings.</div>',
                    unsafe_allow_html=True,
                )

        with probability_col:
            if probability is not None:
                st.metric("Estimated failure probability", f"{probability * 100:.2f}%")
                st.progress(min(max(probability, 0.0), 1.0))
                st.caption("Model probability for the machine-failure class.")
            else:
                st.info("This model does not provide probability estimates.")

        with st.expander("View model input"):
            display_input = input_row.copy()
            display_input["Machine type"] = machine_type
            st.dataframe(display_input, use_container_width=True, hide_index=True)

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

else:
    st.info("👈 Enter the machine readings in the sidebar, then click **Predict machine condition**.")

st.markdown("---")
left, right = st.columns(2)
with left:
    st.markdown("### About this project")
    st.write(
        "This application uses the AI4I 2020 Predictive Maintenance project model. "
        "The notebook compares multiple classifiers and selects XGBoost as the final model."
    )
with right:
    st.markdown("### Model details")
    st.write(f"**Loaded file:** `{loaded_model_path}`")
    st.write("**Expected features:** Air temperature, process temperature, rotational speed, torque, tool wear, and machine type.")
    st.markdown(
        '<p class="small-note">Prediction is an ML estimate, not a substitute for a qualified maintenance inspection.</p>',
        unsafe_allow_html=True,
    )
