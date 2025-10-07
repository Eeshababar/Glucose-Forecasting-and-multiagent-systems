# app.py
# -------------------------------------------------------
# Advanced Glucose Level Predictor (Streamlit)
# -------------------------------------------------------
# Features:
# - Simple login gate (sets session_state.logged_in, patient_id, patient_name)
# - Unified GEMINI_API_KEY handling
# - Robust CSV loading (semicolon/comma fallback)
# - Predict/visualize glucose + Time-in-Range
# - Graph-based analysis via graph.invoke(), with graceful fallback
# - Defensive checks for user_data files and profiles
# -------------------------------------------------------

import os
import traceback
from pathlib import Path
import streamlit as st

# MUST be the first Streamlit call
st.set_page_config(page_title="🩺 Advanced Glucose Level Predictor", page_icon="🩺", layout="wide")

# Optional: demo mode to bypass login in development
if os.getenv("DEMO_MODE", "0") == "1":
    st.session_state["logged_in"] = True
    st.session_state["patient_id"] = st.session_state.get("patient_id", "demo001")
    st.session_state["patient_name"] = st.session_state.get("patient_name", "Demo User")

# ------------------------
# Login UI
# ------------------------
def login_ui():
    st.title("🔐 Login")
    with st.form("login_form", clear_on_submit=False):
        colA, colB = st.columns(2)
        with colA:
            patient_id = st.text_input("Patient ID", value=st.session_state.get("patient_id", "demo001"))
        with colB:
            patient_name = st.text_input("Patient Name", value=st.session_state.get("patient_name", "Patient"))
        password = st.text_input("Password", type="password", value="")
        submitted = st.form_submit_button("Log in")
    if submitted:
        # Replace with real authentication as needed
        if password.strip():
            st.session_state.logged_in = True
            st.session_state.patient_id = (patient_id or "").strip() or "demo001"
            st.session_state.patient_name = (patient_name or "").strip() or "Patient"
            st.success("✅ Logged in")
            st.rerun()
        else:
            st.error("Please enter a password to continue.")

if not st.session_state.get("logged_in", False):
    login_ui()
    st.stop()

# ------------------------
# Imports after login
# ------------------------
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import plotly.graph_objects as go

# Try to import graph + Command; show a visible warning if unavailable
GRAPH_AVAILABLE = True
try:
    from graph import graph
    from langgraph.types import Command
except Exception as e:
    GRAPH_AVAILABLE = False
    graph_import_error = f"Graph pipeline unavailable: {e}"

# ------------------------
# Device
# ------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------
# Session defaults
# ------------------------
defaults = {
    "google_api_key": "",
    "sender_email": "",
    "sender_app_password": "",
    "analysis_state": "idle",         # 'idle', 'running', 'waiting_input', 'complete'
    "analysis_result": None,
    "interrupt_question": None,
    "graph_config": None,
    "current_patient_data": None,
    "session_carbs": 0.0,
    "session_fats": 0.0,
    "session_protein": 0.0,
    "food_logs": []
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

# ------------------------
# Page head
# ------------------------
st.title("🩺 Advanced Glucose Level Predictor")
st.markdown("*Using AI-powered graph-based analysis for comprehensive glucose monitoring*")

# ------------------------
# Sidebar
# ------------------------
with st.sidebar:
    st.write(f"👤 Welcome, {st.session_state.get('patient_name', 'Patient')}")
    st.write(f"ID: {st.session_state.get('patient_id', '')}")

    st.header("⚙️ Configuration")
    low_range = st.number_input("Low Glucose Range (mg/dL)", value=70, min_value=50, max_value=100)
    high_range = st.number_input("High Glucose Range (mg/dL)", value=180, min_value=150, max_value=250)

    st.header("🍽️ Today's Nutrition")
    st.metric("Carbs", f"{st.session_state.session_carbs:.1f}g")
    st.metric("Protein", f"{st.session_state.session_protein:.1f}g")
    st.metric("Fats", f"{st.session_state.session_fats:.1f}g")
    if st.session_state.food_logs:
        total_calories = sum(item.get("total_calories", 0.0) for item in st.session_state.food_logs)
        st.metric("Total Calories", f"{total_calories:.0f}")

    st.header("🔑 API Configuration")
    # Unify on GEMINI_API_KEY
    current_api_key = os.environ.get("GEMINI_API_KEY", "")
    api_key = st.text_input("Gemini API Key", value=current_api_key, type="password",
                            help="Enter your Google Gemini API key")
    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key
        st.session_state.google_api_key = api_key
        st.success("✅ API key configured")

    sender_email = st.text_input("Sender Email", value=st.session_state.sender_email or "",
                                 help="Email used for emergency reports")
    if sender_email:
        st.session_state.sender_email = sender_email
        st.success("✅ Sender Email configured")

    sender_app_password = st.text_input("Sender App Password", value=st.session_state.sender_app_password or "",
                                        type="password", help="Gmail App Password")
    if sender_app_password:
        st.session_state.sender_app_password = sender_app_password
        st.success("✅ Sender app password configured")

    if st.button("🚪 Logout", type="secondary"):
        # Reset essential session vars
        st.session_state.logged_in = False
        st.session_state.patient_id = None
        st.session_state.patient_name = None
        st.session_state.session_carbs = 0.0
        st.session_state.session_fats = 0.0
        st.session_state.session_protein = 0.0
        st.session_state.food_logs = []
        st.session_state.google_api_key = ""
        st.session_state.sender_email = ""
        st.session_state.sender_app_password = ""
        st.rerun()

# ------------------------
# Helpers
# ------------------------
def check_api_key():
    if not os.environ.get("GEMINI_API_KEY"):
        st.error("❌ Please configure your Gemini API key in the sidebar")
        st.stop()

def safe_read_csv(path_or_buffer):
    """Try semicolon first; if single column -> fallback to comma."""
    try:
        df = pd.read_csv(path_or_buffer, delimiter=';')
        if len(df.columns) == 1:
            if hasattr(path_or_buffer, "seek"):
                path_or_buffer.seek(0)
            df = pd.read_csv(path_or_buffer)
        return df
    except Exception:
        # final fallback attempt
        if hasattr(path_or_buffer, "seek"):
            path_or_buffer.seek(0)
        return pd.read_csv(path_or_buffer)

def get_patient_row(patient_id, df_profiles: pd.DataFrame):
    """Return first matching row as a Series or None."""
    if "patient_id" not in df_profiles.columns:
        return None
    rec = df_profiles[df_profiles["patient_id"] == patient_id]
    if rec.empty:
        return None
    return rec.iloc[0]

# ------------------------
# Data loading
# ------------------------
st.header("📁 Your Data")

current_patient_id = st.session_state.get("patient_id", "") or ""
cgm_df = None
data_found = False

# Try user-specific CSV from repository
if current_patient_id:
    data_path = Path("user_data") / "cgm_data" / f"{current_patient_id}.csv"
    if data_path.exists():
        try:
            df = safe_read_csv(str(data_path))
            # Ensure time
            if "time" not in df.columns:
                st.error("❌ Error: Your data file is missing the 'time' column")
                st.stop()
            df["time"] = pd.to_datetime(df["time"])

            # Glucose column
            if "glucose" in df.columns:
                pass
            elif "gl" in df.columns:
                df["glucose"] = df["gl"]
            else:
                st.error("❌ Error: Your data file must contain either a 'glucose' or 'gl' column")
                st.stop()

            cgm_df = df
            data_found = True
            st.success(f"✅ Your glucose data loaded successfully! ({len(df)} records)")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", len(cgm_df))
            with col2:
                st.metric("Date Range", f"{cgm_df['time'].min().date()} to {cgm_df['time'].max().date()}")
            with col3:
                st.metric("Days of Data", cgm_df["time"].dt.date.nunique())

        except Exception as e:
            st.error(f"❌ Error loading your data file: {e}")
            st.caption(traceback.format_exc())
    else:
        st.warning(f"⚠️ No data file found for your patient ID: {current_patient_id}")

# Fallback: uploader
if not data_found:
    st.info("💡 You can upload your glucose data file below:")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        try:
            df = safe_read_csv(uploaded_file)
            if "time" not in df.columns:
                st.error("❌ Error: CSV must contain a 'time' column")
                st.stop()
            df["time"] = pd.to_datetime(df["time"])

            # attach patient id in file-less mode (optional)
            df["patient_id"] = current_patient_id or "uploaded"

            if "glucose" in df.columns:
                pass
            elif "gl" in df.columns:
                df["glucose"] = df["gl"]
            else:
                st.error("❌ Error: CSV must contain either a 'glucose' or 'gl' column")
                st.stop()

            cgm_df = df
            data_found = True
            st.success("✅ Data uploaded successfully!")
        except Exception as e:
            st.error(f"Error loading file: {e}")
            st.caption(traceback.format_exc())

# ------------------------
# Main analysis (only if we have data)
# ------------------------
if cgm_df is not None and data_found:
    patient_data = cgm_df.sort_values("time")
    st.session_state.current_patient_data = patient_data

    # Summary
    st.subheader("📊 Your Glucose Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Records", len(patient_data))
    with col2:
        if "glucose" in patient_data.columns:
            st.metric("Avg Glucose", f"{patient_data['glucose'].mean():.1f} mg/dL")
    with col3:
        st.metric("Date Range", f"{patient_data['time'].dt.date.nunique()} days")
    with col4:
        if "glucose" in patient_data.columns:
            last_glucose = float(patient_data["glucose"].iloc[-1])
            st.metric("Last Reading", f"{last_glucose:.1f} mg/dL")

    # Trend chart
    if "glucose" in patient_data.columns:
        st.subheader("📈 Your Glucose Trends")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=patient_data["time"],
            y=patient_data["glucose"],
            mode="lines+markers",
            name="Glucose Level",
            line=dict(color="blue", width=2)
        ))
        fig.add_hline(y=low_range, line_dash="dash", line_color="red",
                      annotation_text=f"Low ({low_range})")
        fig.add_hline(y=high_range, line_dash="dash", line_color="red",
                      annotation_text=f"High ({high_range})")
        fig.update_layout(title="Your Glucose Levels Over Time",
                          xaxis_title="Time",
                          yaxis_title="Glucose (mg/dL)",
                          height=400)
        total = len(patient_data)
        in_range = ((patient_data["glucose"] >= low_range) &
                    (patient_data["glucose"] <= high_range)).sum()
        below_range = (patient_data["glucose"] < low_range).sum()
        above_range = (patient_data["glucose"] > high_range).sum()

        tir = (in_range / total) * 100 if total else 0.0
        below_pct = (below_range / total) * 100 if total else 0.0
        above_pct = (above_range / total) * 100 if total else 0.0

        st.plotly_chart(fig, use_container_width=True, key="glucose_chart")

        st.subheader("⏱️ Your Time in Range Stats")
        left, right = st.columns([3, 1])
        pie_data = {
            "Category": ["In Range", "Below Range", "Above Range"],
            "Value": [in_range, below_range, above_range]
        }
        pie_fig = px.pie(pie_data, names="Category", values="Value",
                         color="Category",
                         color_discrete_map={"In Range": "green", "Below Range": "blue", "Above Range": "red"},
                         hole=0.4)
        pie_fig.update_traces(textinfo="percent+label")
        left.plotly_chart(pie_fig, use_container_width=True, key="pie_chart")
        right.metric("In Range", f"{tir:.1f}%")
        right.metric("Below Range", f"{below_pct:.1f}%")
        right.metric("Above Range", f"{above_pct:.1f}%")

    # ------------------------
    # AI-Powered Prediction & Analysis
    # ------------------------
    st.subheader("🔮 AI-Powered Prediction & Analysis")

    if st.session_state.analysis_state == "idle":
        if st.button("🚀 Run Advanced Analysis", type="primary"):
            # Only enforce key if your graph actually needs it
            # check_api_key()
            st.session_state.analysis_state = "running"
            st.rerun()

    elif st.session_state.analysis_state == "running":
        try:
            with st.spinner("Running AI analysis..."):
                # Prepare features
                features = [
                    "glucose", "calories", "heart_rate", "steps",
                    "basal_rate", "bolus_volume_delivered", "carb_input"
                ]
                available_features = [f for f in features if f in patient_data.columns]
                if not available_features:
                    st.error("No required features found in the data")
                    st.session_state.analysis_state = "idle"
                else:
                    data = patient_data[available_features].ffill().bfill()

                    scaler = MinMaxScaler()
                    data_scaled = scaler.fit_transform(data)

                    window_size = min(72, len(data_scaled))  # last 6 hours if 5-min samples
                    window = data_scaled[-window_size:]
                    import numpy as np
                    input_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0)

                    config = {"configurable": {"thread_id": "1"}}
                    st.session_state.graph_config = config

                    # Try to read patient profiles (optional)
                    patient_bio_path = Path("user_data") / "patient_profiles.csv"
                    patient_bio_data = None
                    if patient_bio_path.exists():
                        try:
                            profiles_df = safe_read_csv(str(patient_bio_path))
                            # Normalize column names expected later
                            # Allow flexible casing/spaces by renaming common ones if needed
                            rename_map = {
                                "diabetes_proficiency": "Diabetes Proficiency",
                                "diabetes proficiency": "Diabetes Proficiency",
                                "emergency email": "emergency_email",
                                "emergency contact number": "emergency_contact_number"
                            }
                            lower_cols = {c.lower(): c for c in profiles_df.columns}
                            for k, v in rename_map.items():
                                if k in lower_cols and v not in profiles_df.columns:
                                    profiles_df.rename(columns={lower_cols[k]: v}, inplace=True)
                            patient_bio_data = get_patient_row(current_patient_id, profiles_df)
                        except Exception as e:
                            st.warning(f"Could not read patient profiles: {e}")

                    # Build graph params with safe defaults
                    def safe_get(row, key, default=None):
                        if row is None:
                            return default
                        return row[key] if key in row else default

                    graph_params = {
                        "patient_id": current_patient_id,
                        "input_tensor": input_tensor.tolist(),
                        "raw_patient_data": patient_data.to_dict(orient="records"),
                        "low_range": low_range,
                        "high_range": high_range,
                        "messages": [],
                        "rag_complete": False,
                        "age": safe_get(patient_bio_data, "age", None),
                        "gender": safe_get(patient_bio_data, "gender", None),
                        "diabetes_proficiency": safe_get(patient_bio_data, "Diabetes Proficiency", None),
                        "emergency_contact_number": safe_get(patient_bio_data, "emergency_contact_number", None),
                        "emergency_email": safe_get(patient_bio_data, "emergency_email", None),
                        "name": safe_get(patient_bio_data, "name", st.session_state.get("patient_name", "Patient")),
                        "id": safe_get(patient_bio_data, "patient_id", current_patient_id),
                        "carbs_grams": st.session_state.session_carbs,
                        "protein_grams": st.session_state.session_protein,
                        "fat_grams": st.session_state.session_fats,
                        "food_logs": st.session_state.food_logs
                    }

                    # Optional email creds
                    current_sender_email = st.session_state.sender_email or None
                    current_sender_app_password = st.session_state.sender_app_password or None
                    if current_sender_email and current_sender_app_password:
                        graph_params.update({
                            "sender_email": current_sender_email,
                            "sender_account_app_password": current_sender_app_password
                        })
                    else:
                        graph_params.update({
                            "sender_email": None,
                            "sender_account_app_password": None
                        })

                    # Invoke the graph if available; otherwise make a simple local result
                    if GRAPH_AVAILABLE:
                        try:
                            result = graph.invoke(graph_params, config)
                        except Exception as ge:
                            st.warning("Graph pipeline failed; falling back to local quick analysis.")
                            st.caption(str(ge))
                            # Minimal fallback result (no interrupts)
                            last_glu = float(patient_data["glucose"].iloc[-1])
                            result = {
                                "predicted_glucose": last_glu,  # naive hold
                                "glucose_level": (
                                    "Normal" if low_range <= last_glu <= high_range
                                    else "Warning" if (last_glu < low_range and last_glu >= low_range - 10)
                                    or (last_glu > high_range and last_glu <= high_range + 10)
                                    else "High" if last_glu > high_range else "Low"
                                ),
                                "trend_note": "Fallback estimate based on last reading.",
                                "emergency": not (low_range <= last_glu <= high_range),
                                "advice": "- Stay hydrated.\n- Recheck glucose in 15 minutes.\n- Follow your care plan.",
                                "routine_plan": "- Maintain consistent meals.\n- Track carbs and activity."
                            }
                    else:
                        st.warning(f"{graph_import_error} — using local quick analysis.")
                        last_glu = float(patient_data["glucose"].iloc[-1])
                        result = {
                            "predicted_glucose": last_glu,  # naive hold
                            "glucose_level": (
                                "Normal" if low_range <= last_glu <= high_range
                                else "Warning" if (last_glu < low_range and last_glu >= low_range - 10)
                                or (last_glu > high_range and last_glu <= high_range + 10)
                                else "High" if last_glu > high_range else "Low"
                            ),
                            "trend_note": "Fallback estimate based on last reading.",
                            "emergency": not (low_range <= last_glu <= high_range),
                            "advice": "- Stay hydrated.\n- Recheck glucose in 15 minutes.\n- Follow your care plan.",
                            "routine_plan": "- Maintain consistent meals.\n- Track carbs and activity."
                        }

                    # Handle interrupts if any
                    interrupts = result.get("__interrupt__", []) if isinstance(result, dict) else []
                    if interrupts:
                        interrupt_value = interrupts[0].value
                        question = interrupt_value.get("question", "Provide additional information:")
                        st.session_state.interrupt_question = question
                        st.session_state.analysis_result = result
                        st.session_state.analysis_state = "waiting_input"
                        st.rerun()
                    else:
                        st.session_state.analysis_result = result
                        st.session_state.analysis_state = "complete"
                        st.rerun()

        except Exception as e:
            st.error(f"Error during analysis: {e}")
            st.exception(e)
            st.session_state.analysis_state = "idle"

    elif st.session_state.analysis_state == "waiting_input":
        st.warning("⚠️ Additional Information Required")
        user_input = st.text_input(st.session_state.interrupt_question or "Provide additional details:", key="user_interrupt_input")
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Submit Response", type="primary"):
                if user_input:
                    try:
                        with st.spinner("Processing your response..."):
                            if GRAPH_AVAILABLE:
                                result = graph.invoke(Command(resume=user_input), config=st.session_state.graph_config)
                            else:
                                # Fallback: simply record the response and mark complete
                                result = st.session_state.analysis_result or {}
                                result["user_response"] = user_input
                            st.session_state.analysis_result = result
                            st.session_state.analysis_state = "complete"
                            st.rerun()
                    except Exception as e:
                        st.error(f"Failed to process response: {e}")
                        st.exception(e)
                else:
                    st.error("Please provide a response before submitting")
        with col2:
            if st.button("Cancel Analysis"):
                st.session_state.analysis_state = "idle"
                st.session_state.analysis_result = None
                st.session_state.interrupt_question = None
                st.session_state.graph_config = None
                st.rerun()

    elif st.session_state.analysis_state == "complete":
        # Display results
        result = st.session_state.analysis_result or {}
        st.success("✅ Analysis Complete!")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🔮 Prediction Results")
            if "predicted_glucose" in result:
                try:
                    st.metric("Predicted Glucose", f"{float(result['predicted_glucose']):.1f} mg/dL")
                except Exception:
                    st.metric("Predicted Glucose", f"{result['predicted_glucose']}")
            if "glucose_level" in result:
                level = str(result["glucose_level"])
                color = "green" if level == "Normal" else ("orange" if level == "Warning" else "red")
                st.markdown(f"**Glucose Level:** :{color}[{level}]")
            if "trend_note" in result:
                st.info(f"📈 **Trend:** {result['trend_note']}")

        with col2:
            st.subheader("🚨 Emergency Status")
            if bool(result.get("emergency", False)):
                st.error("🚨 EMERGENCY DETECTED")
            else:
                st.success("✅ No Emergency Detected")

        # Forecast chart
        if "glucose" in patient_data.columns and "predicted_glucose" in result:
            try:
                st.subheader("📈 Your Glucose Forecast")
                last_time = patient_data["time"].iloc[-1]
                future_time = last_time + pd.Timedelta(minutes=5)

                forecast_fig = go.Figure()
                forecast_fig.add_trace(go.Scatter(
                    x=patient_data["time"],
                    y=patient_data["glucose"],
                    mode="lines+markers",
                    name="Your Glucose Level",
                    line=dict(color="blue", width=2)
                ))
                last_val = float(patient_data["glucose"].iloc[-1])
                pred_val = float(result["predicted_glucose"])
                forecast_fig.add_trace(go.Scatter(
                    x=[last_time, future_time],
                    y=[last_val, pred_val],
                    mode="lines+markers",
                    name="Forecast",
                    line=dict(color="orange", dash="dot", width=2),
                    marker=dict(color="orange", size=10)
                ))
                forecast_fig.add_hline(y=low_range, line_dash="dash", line_color="red",
                                       annotation_text=f"Low ({low_range})")
                forecast_fig.add_hline(y=high_range, line_dash="dash", line_color="red",
                                       annotation_text=f"High ({high_range})")
                forecast_fig.update_layout(
                    title="Your Glucose Levels with Forecast",
                    xaxis_title="Time",
                    yaxis_title="Glucose (mg/dL)",
                    height=400
                )
                start_zoom = last_time - pd.Timedelta(hours=3)
                end_zoom = future_time + pd.Timedelta(minutes=15)
                forecast_fig.update_xaxes(range=[start_zoom, end_zoom])
                st.plotly_chart(forecast_fig, use_container_width=True, key="forecast_chart")
            except Exception as e:
                st.warning(f"Could not render forecast chart: {e}")

        if "advice" in result and result["advice"]:
            st.subheader("🧠 AI Recommendations")
            st.markdown(str(result["advice"]))

        if "routine_plan" in result and result["routine_plan"]:
            st.subheader("📅 Personalized Routine Plan")
            st.markdown(str(result["routine_plan"]))

        if st.button("🔄 Run New Analysis"):
            st.session_state.analysis_state = "idle"
            st.session_state.analysis_result = None
            st.session_state.interrupt_question = None
            st.session_state.graph_config = None
            st.rerun()

else:
    if current_patient_id:
        st.info(f"👆 Please upload your glucose data file or ensure it exists at: `user_data/cgm_data/{current_patient_id}.csv`")
    else:
        st.error("❌ No patient ID found. Please log in again.")

# Footer
st.markdown("---")
st.markdown("*Your Personal Advanced Glucose Monitoring System powered by AI*")
