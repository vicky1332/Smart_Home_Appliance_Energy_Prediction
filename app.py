import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ---------------------------------------------------------
# Page Configuration & Custom CSS Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Smart Home Energy Consumption Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme aesthetic & cursor pointer fix
st.markdown("""
<style>
    /* Metric Cards */
    .stMetric {
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
        padding: 16px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
    }
    
    /* Selectbox Pointer Cursor Fix */
    .stSelectbox, .stSelectbox div, .stSelectbox input, 
    div[data-baseweb="select"], div[data-baseweb="select"] *,
    div[role="combobox"], div[role="combobox"] * {
        cursor: pointer !important;
    }

    .metric-container {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
    }
    .metric-value {
        font-size: 3.2rem;
        font-weight: 800;
        color: #60A5FA;
        margin: 12px 0;
        letter-spacing: -1px;
    }
    .metric-unit {
        font-size: 1.2rem;
        color: #94A3B8;
        font-weight: 600;
    }
    .badge-low {
        background-color: rgba(16, 185, 129, 0.2);
        border: 1px solid #10B981;
        color: #34D399;
        padding: 6px 16px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.95rem;
        display: inline-block;
    }
    .badge-moderate {
        background-color: rgba(245, 158, 11, 0.2);
        border: 1px solid #F59E0B;
        color: #FBBF24;
        padding: 6px 16px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.95rem;
        display: inline-block;
    }
    .badge-high {
        background-color: rgba(249, 115, 22, 0.2);
        border: 1px solid #F97316;
        color: #FB923C;
        padding: 6px 16px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.95rem;
        display: inline-block;
    }
    .badge-peak {
        background-color: rgba(239, 68, 68, 0.2);
        border: 1px solid #EF4444;
        color: #FCA5A5;
        padding: 6px 16px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.95rem;
        display: inline-block;
    }
    
    /* Sidebar Styled Cards */
    .sidebar-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 16px;
    }
    .sidebar-status {
        color: #34D399;
        font-weight: 700;
        font-size: 1.05rem;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 8px;
    }
    .sidebar-metric-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 8px;
        margin-top: 10px;
        text-align: center;
    }
    .sidebar-metric-item {
        background: #0F172A;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 8px 4px;
    }
    .sidebar-metric-label {
        font-size: 0.75rem;
        color: #94A3B8;
        font-weight: 600;
    }
    .sidebar-metric-val {
        font-size: 0.95rem;
        color: #F8FAFC;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Feature Metadata Mapping (Human-Friendly Labels & Metadata)
# ---------------------------------------------------------
FEATURE_METADATA = {
    "lights": {
        "label": "Lighting Energy Use",
        "unit": "Wh",
        "group": "💡 Lighting System",
        "description": "Energy consumption of indoor light fixtures",
        "min": 0.0, "max": 70.0, "default": 0.0, "step": 10.0
    },
    "T1": {
        "label": "Kitchen Temperature",
        "unit": "°C",
        "group": "🏠 Living Areas",
        "description": "Temperature measured in the kitchen area",
        "min": 15.0, "max": 30.0, "default": 21.6, "step": 0.1
    },
    "RH_1": {
        "label": "Kitchen Humidity",
        "unit": "%",
        "group": "🏠 Living Areas",
        "description": "Relative humidity in the kitchen area",
        "min": 20.0, "max": 70.0, "default": 39.7, "step": 0.1
    },
    "T2": {
        "label": "Living Room Temperature",
        "unit": "°C",
        "group": "🏠 Living Areas",
        "description": "Temperature measured in the living room",
        "min": 15.0, "max": 32.0, "default": 20.0, "step": 0.1
    },
    "RH_2": {
        "label": "Living Room Humidity",
        "unit": "%",
        "group": "🏠 Living Areas",
        "description": "Relative humidity in the living room",
        "min": 20.0, "max": 65.0, "default": 40.5, "step": 0.1
    },
    "T3": {
        "label": "Laundry Room Temperature",
        "unit": "°C",
        "group": "🧺 Utility Areas",
        "description": "Temperature measured in the laundry room",
        "min": 15.0, "max": 32.0, "default": 22.1, "step": 0.1
    },
    "RH_3": {
        "label": "Laundry Room Humidity",
        "unit": "%",
        "group": "🧺 Utility Areas",
        "description": "Relative humidity in the laundry room",
        "min": 20.0, "max": 60.0, "default": 38.5, "step": 0.1
    },
    "T4": {
        "label": "Office Room Temperature",
        "unit": "°C",
        "group": "🏠 Living Areas",
        "description": "Temperature measured in the home office",
        "min": 15.0, "max": 30.0, "default": 20.7, "step": 0.1
    },
    "RH_4": {
        "label": "Office Room Humidity",
        "unit": "%",
        "group": "🏠 Living Areas",
        "description": "Relative humidity in the home office",
        "min": 20.0, "max": 60.0, "default": 38.4, "step": 0.1
    },
    "T5": {
        "label": "Bathroom Temperature",
        "unit": "°C",
        "group": "🧺 Utility Areas",
        "description": "Temperature measured in the bathroom",
        "min": 15.0, "max": 30.0, "default": 19.4, "step": 0.1
    },
    "RH_5": {
        "label": "Bathroom Humidity",
        "unit": "%",
        "group": "🧺 Utility Areas",
        "description": "Relative humidity in the bathroom",
        "min": 25.0, "max": 100.0, "default": 49.1, "step": 0.1
    },
    "T6": {
        "label": "North Exterior Temperature",
        "unit": "°C",
        "group": "🌤️ Outdoor & Weather",
        "description": "Outdoor temperature measured at north facade of building",
        "min": -10.0, "max": 35.0, "default": 7.3, "step": 0.1
    },
    "RH_6": {
        "label": "North Exterior Humidity",
        "unit": "%",
        "group": "🌤️ Outdoor & Weather",
        "description": "Outdoor humidity measured at north facade of building",
        "min": 0.0, "max": 100.0, "default": 55.3, "step": 0.1
    },
    "T7": {
        "label": "Ironing Room Temperature",
        "unit": "°C",
        "group": "🛌 Bedrooms & Personal Rooms",
        "description": "Temperature measured in the ironing room",
        "min": 15.0, "max": 30.0, "default": 20.0, "step": 0.1
    },
    "RH_7": {
        "label": "Ironing Room Humidity",
        "unit": "%",
        "group": "🛌 Bedrooms & Personal Rooms",
        "description": "Relative humidity in the ironing room",
        "min": 20.0, "max": 60.0, "default": 34.9, "step": 0.1
    },
    "T8": {
        "label": "Teenager Room Temperature",
        "unit": "°C",
        "group": "🛌 Bedrooms & Personal Rooms",
        "description": "Temperature measured in teenager room 2",
        "min": 15.0, "max": 30.0, "default": 22.1, "step": 0.1
    },
    "RH_8": {
        "label": "Teenager Room Humidity",
        "unit": "%",
        "group": "🛌 Bedrooms & Personal Rooms",
        "description": "Relative humidity in teenager room 2",
        "min": 20.0, "max": 65.0, "default": 42.4, "step": 0.1
    },
    "T9": {
        "label": "Parents Room Temperature",
        "unit": "°C",
        "group": "🛌 Bedrooms & Personal Rooms",
        "description": "Temperature measured in the parents bedroom",
        "min": 14.0, "max": 28.0, "default": 19.4, "step": 0.1
    },
    "RH_9": {
        "label": "Parents Room Humidity",
        "unit": "%",
        "group": "🛌 Bedrooms & Personal Rooms",
        "description": "Relative humidity in the parents bedroom",
        "min": 20.0, "max": 60.0, "default": 40.9, "step": 0.1
    },
    "T_out": {
        "label": "Outdoor Weather Station Temperature",
        "unit": "°C",
        "group": "🌤️ Outdoor & Weather",
        "description": "Ambient outdoor temperature from weather station",
        "min": -10.0, "max": 35.0, "default": 6.9, "step": 0.1
    },
    "Press_mm_hg": {
        "label": "Barometric Pressure",
        "unit": "mm Hg",
        "group": "🌤️ Outdoor & Weather",
        "description": "Atmospheric pressure measured at weather station",
        "min": 720.0, "max": 780.0, "default": 756.1, "step": 0.1
    },
    "RH_out": {
        "label": "Outdoor Weather Station Humidity",
        "unit": "%",
        "group": "🌤️ Outdoor & Weather",
        "description": "Outdoor relative humidity from weather station",
        "min": 20.0, "max": 100.0, "default": 83.7, "step": 0.1
    },
    "Windspeed": {
        "label": "Wind Speed",
        "unit": "m/s",
        "group": "🌤️ Outdoor & Weather",
        "description": "Outdoor wind speed measured at weather station",
        "min": 0.0, "max": 15.0, "default": 3.7, "step": 0.1
    },
    "Visibility": {
        "label": "Outdoor Visibility",
        "unit": "km",
        "group": "🌤️ Outdoor & Weather",
        "description": "Outdoor atmospheric visibility distance",
        "min": 1.0, "max": 70.0, "default": 40.0, "step": 0.5
    },
    "Tdewpoint": {
        "label": "Dew Point Temperature",
        "unit": "°C",
        "group": "🌤️ Outdoor & Weather",
        "description": "Outdoor dew point temperature",
        "min": -10.0, "max": 20.0, "default": 3.4, "step": 0.1
    }
}

PRESET_SCENARIOS = {
    "Average Day (Normal Occupancy)": {
        "lights": 0.0, "T1": 21.6, "RH_1": 39.7, "T2": 20.0, "RH_2": 40.5,
        "T3": 22.1, "RH_3": 38.5, "T4": 20.7, "RH_4": 38.4, "T5": 19.4,
        "RH_5": 49.1, "T6": 7.3, "RH_6": 55.3, "T7": 20.0, "RH_7": 34.9,
        "T8": 22.1, "RH_8": 42.4, "T9": 19.4, "RH_9": 40.9, "T_out": 6.9,
        "Press_mm_hg": 756.1, "RH_out": 83.7, "Windspeed": 3.7, "Visibility": 40.0, "Tdewpoint": 3.4
    },
    "Cold Winter Morning (High Heating & Lights)": {
        "lights": 40.0, "T1": 20.0, "RH_1": 44.0, "T2": 19.0, "RH_2": 46.0,
        "T3": 20.5, "RH_3": 42.0, "T4": 19.5, "RH_4": 43.0, "T5": 18.5,
        "RH_5": 58.0, "T6": -2.5, "RH_6": 88.0, "T7": 19.0, "RH_7": 41.0,
        "T8": 20.0, "RH_8": 45.0, "T9": 18.0, "RH_9": 44.0, "T_out": -3.0,
        "Press_mm_hg": 765.0, "RH_out": 92.0, "Windspeed": 6.5, "Visibility": 25.0, "Tdewpoint": -4.0
    },
    "Hot Summer Afternoon (High Outdoor Temp)": {
        "lights": 0.0, "T1": 24.5, "RH_1": 35.0, "T2": 25.0, "RH_2": 34.0,
        "T3": 25.5, "RH_3": 33.0, "T4": 24.0, "RH_4": 35.0, "T5": 23.5,
        "RH_5": 45.0, "T6": 26.0, "RH_6": 28.0, "T7": 24.0, "RH_7": 32.0,
        "T8": 25.0, "RH_8": 36.0, "T9": 23.0, "RH_9": 35.0, "T_out": 25.5,
        "Press_mm_hg": 752.0, "RH_out": 42.0, "Windspeed": 2.5, "Visibility": 60.0, "Tdewpoint": 12.0
    },
    "Peak Evening Usage (Active Appliances)": {
        "lights": 50.0, "T1": 22.5, "RH_1": 48.0, "T2": 21.5, "RH_2": 45.0,
        "T3": 23.5, "RH_3": 46.0, "T4": 21.0, "RH_4": 42.0, "T5": 21.0,
        "RH_5": 75.0, "T6": 9.0, "RH_6": 65.0, "T7": 21.0, "RH_7": 40.0,
        "T8": 22.5, "RH_8": 48.0, "T9": 20.0, "RH_9": 45.0, "T_out": 8.5,
        "Press_mm_hg": 754.0, "RH_out": 85.0, "Windspeed": 4.0, "Visibility": 40.0, "Tdewpoint": 5.0
    },
    "Eco / Away Mode (Low Activity)": {
        "lights": 0.0, "T1": 18.0, "RH_1": 32.0, "T2": 17.5, "RH_2": 33.0,
        "T3": 18.5, "RH_3": 31.0, "T4": 17.5, "RH_4": 32.0, "T5": 17.0,
        "RH_5": 40.0, "T6": 5.0, "RH_6": 45.0, "T7": 17.5, "RH_7": 30.0,
        "T8": 18.0, "RH_8": 34.0, "T9": 17.0, "RH_9": 33.0, "T_out": 4.5,
        "Press_mm_hg": 758.0, "RH_out": 75.0, "Windspeed": 2.0, "Visibility": 45.0, "Tdewpoint": 1.0
    }
}

SCENARIO_OPTIONS = [
    "🛠️ Custom / Manual Entry",
    "Average Day (Normal Occupancy)",
    "Cold Winter Morning (High Heating & Lights)",
    "Hot Summer Afternoon (High Outdoor Temp)",
    "Peak Evening Usage (Active Appliances)",
    "Eco / Away Mode (Low Activity)"
]


# ---------------------------------------------------------
# Load Model Artifact
# ---------------------------------------------------------
@st.cache_resource
def load_model_artifact(model_path="smart_home_appliance_energy_model.pkl"):
    if not os.path.exists(model_path):
        st.error(f"Model file `{model_path}` not found in repository. Please ensure `{model_path}` is committed to GitHub.")
        return None
    try:
        artifact = joblib.load(model_path)
        if isinstance(artifact, dict) and "model" in artifact and "features" in artifact:
            return artifact
        else:
            st.error("Model artifact format is invalid.")
            return None
    except Exception as e:
        st.error(f"Error loading model artifact: {e}")
        return None


artifact = load_model_artifact()
expected_features = artifact["features"] if artifact else list(FEATURE_METADATA.keys())

# Initialize input slider states if missing
for feat_key in expected_features:
    if f"input_{feat_key}" not in st.session_state:
        st.session_state[f"input_{feat_key}"] = float(FEATURE_METADATA[feat_key]["default"])

if "preset_selector" not in st.session_state:
    st.session_state["preset_selector"] = "Average Day (Normal Occupancy)"

# Initialize active prediction input state dictionary
if "active_prediction_input" not in st.session_state:
    st.session_state["active_prediction_input"] = {
        k: float(v["default"]) for k, v in FEATURE_METADATA.items()
    }

if "custom_modified_pending" not in st.session_state:
    st.session_state["custom_modified_pending"] = False


# Callbacks for clean Streamlit state updates without exceptions
def on_preset_change():
    chosen = st.session_state.get("preset_selector")
    if chosen in PRESET_SCENARIOS:
        preset_vals = PRESET_SCENARIOS[chosen]
        for key, val in preset_vals.items():
            st.session_state[f"input_{key}"] = float(val)
        # For preset scenarios, update prediction immediately
        st.session_state["active_prediction_input"] = {
            k: float(st.session_state[f"input_{k}"]) for k in expected_features
        }
        st.session_state["custom_modified_pending"] = False


def on_slider_change():
    st.session_state["preset_selector"] = "🛠️ Custom / Manual Entry"
    # User is manually editing values — require clicking Predict button!
    st.session_state["custom_modified_pending"] = True


def reset_to_defaults():
    default_vals = PRESET_SCENARIOS["Average Day (Normal Occupancy)"]
    for key, val in default_vals.items():
        st.session_state[f"input_{key}"] = float(val)
    st.session_state["preset_selector"] = "Average Day (Normal Occupancy)"
    st.session_state["active_prediction_input"] = {
        k: float(st.session_state[f"input_{k}"]) for k in expected_features
    }
    st.session_state["custom_modified_pending"] = False


# ---------------------------------------------------------
# Sidebar Layout & Model Status Card
# ---------------------------------------------------------
st.sidebar.title("⚡ Smart Home Energy")
st.sidebar.markdown("Predict household appliance energy consumption in real-time.")

if artifact:
    metrics = artifact.get("metrics", {})
    mae_val = metrics.get('MAE', 0)
    rmse_val = metrics.get('RMSE', 0)
    r2_val = metrics.get('R2', 0)
    model_name = artifact.get('model_name', 'Gradient Boosting Regressor')

    st.sidebar.markdown(
        f"""
        <div class="sidebar-card">
            <div class="sidebar-status">🟢 ML Model Active</div>
            <div style="font-size: 0.85rem; color: #CBD5E1; font-weight: 600;">{model_name}</div>
            <div style="font-size: 0.75rem; color: #94A3B8; margin-bottom: 8px;">UCI Appliance Energy Model</div>
            <div class="sidebar-metric-grid">
                <div class="sidebar-metric-item">
                    <div class="sidebar-metric-label">MAE</div>
                    <div class="sidebar-metric-val">{mae_val:.1f} Wh</div>
                </div>
                <div class="sidebar-metric-item">
                    <div class="sidebar-metric-label">RMSE</div>
                    <div class="sidebar-metric-val">{rmse_val:.1f} Wh</div>
                </div>
                <div class="sidebar-metric-item">
                    <div class="sidebar-metric-label">R²</div>
                    <div class="sidebar-metric-val">{r2_val:.3f}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.sidebar.error("❌ Model artifact missing!")

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Scenario Presets")

st.sidebar.selectbox(
    "Choose a pre-configured scenario or enter custom values:",
    options=SCENARIO_OPTIONS,
    key="preset_selector",
    on_change=on_preset_change
)

st.sidebar.button(
    "🔄 Reset to Default Scenario",
    use_container_width=True,
    on_click=reset_to_defaults
)

st.sidebar.info(
    "💡 **Note:** Feature codes (e.g. T1, RH_1) from the UCI dataset have been mapped to intuitive home sensor names (e.g. Kitchen Temperature, Outdoor Humidity)."
)


# ---------------------------------------------------------
# Main App Header
# ---------------------------------------------------------
st.title("🏠 Smart Home Appliance Energy Prediction")
st.markdown(
    "Estimate **household appliance energy consumption** (Wh per 10-minute interval) using real-time indoor room climate sensors, lighting activity, and outdoor weather parameters."
)

# Tabs Interface
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Interactive Predictor",
    "📊 Model Analytics & Importance",
    "📁 Batch CSV Prediction",
    "📖 Sensor & Feature Glossary"
])

# ---------------------------------------------------------
# Tab 1: Interactive Predictor
# ---------------------------------------------------------
with tab1:
    st.markdown("### 🎛️ Input Environmental & House Sensor Values")
    st.caption("Select a scenario preset above or manually adjust any sensor sliders below.")

    input_data = {}

    # Group features by category
    groups = {
        "💡 Lighting System": ["lights"],
        "🏠 Living Areas": ["T1", "RH_1", "T2", "RH_2", "T4", "RH_4"],
        "🛌 Bedrooms & Personal Rooms": ["T7", "RH_7", "T8", "RH_8", "T9", "RH_9"],
        "🧺 Utility Areas": ["T3", "RH_3", "T5", "RH_5"],
        "🌤️ Outdoor & Weather": ["T6", "RH_6", "T_out", "RH_out", "Press_mm_hg", "Windspeed", "Visibility", "Tdewpoint"]
    }

    cols = st.columns(2)
    col_idx = 0

    for group_name, feat_list in groups.items():
        with cols[col_idx % 2]:
            with st.expander(f"{group_name} ({len(feat_list)} sensors)", expanded=True):
                for feat in feat_list:
                    meta = FEATURE_METADATA[feat]
                    val = st.slider(
                        label=f"{meta['label']} ({meta['unit']})",
                        min_value=float(meta["min"]),
                        max_value=float(meta["max"]),
                        step=float(meta["step"]),
                        key=f"input_{feat}",
                        on_change=on_slider_change,
                        help=f"Dataset code: {feat} — {meta['description']}"
                    )
                    input_data[feat] = val
        col_idx += 1

    st.markdown("---")

    # Inform user if custom modifications require clicking Predict
    if st.session_state.get("custom_modified_pending", False):
        st.info("ℹ️ **Custom parameters modified.** Click **🚀 Predict Appliance Energy Consumption** below to recalculate predictions.")
    
    # Explicit Prediction Action Button
    predict_btn = st.button("🚀 Predict Appliance Energy Consumption", type="primary", use_container_width=True)

    if predict_btn:
        # Update active prediction input when user explicitly clicks Predict!
        st.session_state["active_prediction_input"] = {
            k: float(st.session_state[f"input_{k}"]) for k in expected_features
        }
        st.session_state["custom_modified_pending"] = False

    # Perform prediction on active_prediction_input
    active_input_df = pd.DataFrame([st.session_state["active_prediction_input"]])[expected_features]

    if artifact and "model" in artifact:
        model = artifact["model"]
        predicted_wh = model.predict(active_input_df)[0]
        predicted_wh = max(0.0, predicted_wh)  # Energy cannot be negative

        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            # Severity badge logic
            if predicted_wh < 60:
                badge_html = '<span class="badge-low">🟢 Low Energy Consumption</span>'
                level_desc = "Appliance energy demand is minimal. House is operating efficiently."
            elif predicted_wh < 120:
                badge_html = '<span class="badge-moderate">🟡 Moderate Consumption</span>'
                level_desc = "Standard baseline household appliance usage."
            elif predicted_wh < 220:
                badge_html = '<span class="badge-high">🟠 High Consumption</span>'
                level_desc = "Above average appliance usage (heavy kitchen/laundry appliances active)."
            else:
                badge_html = '<span class="badge-peak">🔴 Peak Usage Alert</span>'
                level_desc = "High energy demand spike detected across appliances!"

            st.markdown(
                f"""
                <div class="metric-container">
                    <div style="font-size:1.1rem; font-weight:600; color:#94A3B8;">Predicted Energy Use (10-min interval)</div>
                    <div class="metric-value">{predicted_wh:.1f} <span class="metric-unit">Wh</span></div>
                    <div>{badge_html}</div>
                    <div style="margin-top:12px; font-size:0.95rem; color:#CBD5E1;">{level_desc}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Energy Equivalencies
            st.markdown("#### 💡 Energy Equivalent Insights")
            hourly_est_wh = predicted_wh * 6
            daily_est_kwh = (hourly_est_wh * 24) / 1000.0
            led_bulbs = int(hourly_est_wh / 10)

            st.markdown(
                f"""
                <div style="display: flex; gap: 10px; margin-top: 8px; flex-wrap: wrap;">
                    <div style="flex: 1; min-width: 110px; background: #1E293B; border: 1px solid #334155; border-radius: 12px; padding: 14px 10px; text-align: center;">
                        <div style="font-size: 0.75rem; color: #94A3B8; font-weight: 600; text-transform: uppercase;">Hourly Rate</div>
                        <div style="font-size: 1.3rem; font-weight: 700; color: #F8FAFC; margin-top: 4px;">{hourly_est_wh:,.0f}</div>
                        <div style="font-size: 0.75rem; color: #60A5FA; font-weight: 600;">Wh / hour</div>
                    </div>
                    <div style="flex: 1; min-width: 110px; background: #1E293B; border: 1px solid #334155; border-radius: 12px; padding: 14px 10px; text-align: center;">
                        <div style="font-size: 0.75rem; color: #94A3B8; font-weight: 600; text-transform: uppercase;">Daily Projected</div>
                        <div style="font-size: 1.3rem; font-weight: 700; color: #F8FAFC; margin-top: 4px;">{daily_est_kwh:.2f}</div>
                        <div style="font-size: 0.75rem; color: #60A5FA; font-weight: 600;">kWh / day</div>
                    </div>
                    <div style="flex: 1; min-width: 110px; background: #1E293B; border: 1px solid #334155; border-radius: 12px; padding: 14px 10px; text-align: center;">
                        <div style="font-size: 0.75rem; color: #94A3B8; font-weight: 600; text-transform: uppercase;">LED Equivalent</div>
                        <div style="font-size: 1.3rem; font-weight: 700; color: #F8FAFC; margin-top: 4px;">{led_bulbs:,}</div>
                        <div style="font-size: 0.75rem; color: #60A5FA; font-weight: 600;">bulbs (10W)</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with res_col2:
            # Plotly Gauge Chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=predicted_wh,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Appliance Power Gauge (Wh / 10-min)", 'font': {'size': 16, 'color': '#F8FAFC'}},
                number={'suffix': " Wh", 'font': {'color': '#60A5FA', 'size': 32}},
                gauge={
                    'axis': {'range': [0, 500], 'tickwidth': 1, 'tickcolor': "#94A3B8"},
                    'bar': {'color': "#3B82F6"},
                    'bgcolor': "#1E293B",
                    'borderwidth': 2,
                    'bordercolor': "#334155",
                    'steps': [
                        {'range': [0, 60], 'color': 'rgba(16, 185, 129, 0.25)'},
                        {'range': [60, 120], 'color': 'rgba(245, 158, 11, 0.25)'},
                        {'range': [120, 220], 'color': 'rgba(249, 115, 22, 0.25)'},
                        {'range': [220, 500], 'color': 'rgba(239, 68, 68, 0.25)'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 300
                    }
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=50, b=20),
                height=280
            )
            st.plotly_chart(fig_gauge, use_container_width=True)


# ---------------------------------------------------------
# Tab 2: Model Analytics & Feature Importance
# ---------------------------------------------------------
with tab2:
    st.markdown("### 📊 Model Performance & Feature Importances")
    st.write(
        "The model was trained using **Gradient Boosting Regression** on the UCI Appliances Energy Prediction dataset."
    )

    if artifact:
        # Metrics Display
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Mean Absolute Error (MAE)", f"{artifact['metrics'].get('MAE', 0):.2f} Wh")
        m2.metric("Root Mean Sq Error (RMSE)", f"{artifact['metrics'].get('RMSE', 0):.2f} Wh")
        m3.metric("R² Score", f"{artifact['metrics'].get('R2', 0):.4f}")
        m4.metric("Features Trained", f"{len(expected_features)} Sensors")

        st.markdown("---")
        st.markdown("### 🔍 Feature Importance Ranking")

        # Extract Feature Importances directly from model pipeline
        try:
            model_pipeline = artifact["model"]
            gb_model = model_pipeline.named_steps["model"]
            importances = gb_model.feature_importances_
            all_feats = artifact.get("features", expected_features)

            importance_df = pd.DataFrame({
                "Technical Code": all_feats,
                "Friendly Name": [FEATURE_METADATA.get(f, {}).get("label", f) for f in all_feats],
                "Importance": importances
            }).sort_values("Importance", ascending=True)

            fig_imp = px.bar(
                importance_df.tail(15),
                x="Importance",
                y="Friendly Name",
                orientation="h",
                title="Top Feature Importances in Energy Prediction",
                hover_data=["Technical Code"],
                color="Importance",
                color_continuous_scale="Blues"
            )
            fig_imp.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#F8FAFC'),
                height=480,
                xaxis_title="Relative Importance Score",
                yaxis_title="Sensor / Feature"
            )
            st.plotly_chart(fig_imp, use_container_width=True)

        except Exception as err:
            st.error(f"Error loading feature importances: {err}")

        # Hyperparameter JSON View
        with st.expander("⚙️ View Best Model Hyperparameters"):
            st.json(artifact.get("best_params", {}))


# ---------------------------------------------------------
# Tab 3: Batch CSV Prediction
# ---------------------------------------------------------
with tab3:
    st.markdown("### 📁 Batch Prediction via CSV Upload")
    st.markdown(
        "Upload a `.csv` file containing sensor measurements to generate predictions for multiple observation intervals at once."
    )

    # Offer sample template download
    sample_rows = []
    for scenario_name, sc_vals in PRESET_SCENARIOS.items():
        row = {"Scenario_Name": scenario_name}
        row.update(sc_vals)
        sample_rows.append(row)
    sample_df = pd.DataFrame(sample_rows)
    
    st.download_button(
        label="📥 Download Sample Batch Input CSV Template",
        data=sample_df.to_csv(index=False).encode('utf-8'),
        file_name="sample_smart_home_energy_inputs.csv",
        mime="text/csv"
    )

    st.markdown("---")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"CSV uploaded successfully! ({len(batch_df)} rows found)")
            st.dataframe(batch_df.head(), use_container_width=True)

            # Map friendly names back to technical feature names if needed
            friendly_to_tech = {v["label"]: k for k, v in FEATURE_METADATA.items()}
            renamed_df = batch_df.rename(columns=friendly_to_tech)

            missing_cols = [col for col in expected_features if col not in renamed_df.columns]

            if missing_cols:
                st.error(f"Missing required sensor columns in CSV: `{missing_cols}`")
                st.info("Ensure your CSV includes either technical feature codes (T1, RH_1...) or friendly labels (Kitchen Temperature...).")
            else:
                input_batch = renamed_df[expected_features]
                predictions = artifact["model"].predict(input_batch)
                predictions = np.maximum(0.0, predictions)

                result_df = batch_df.copy()
                result_df["Predicted_Appliance_Energy_Wh"] = np.round(predictions, 2)

                st.markdown("#### 📊 Prediction Results Preview")
                st.dataframe(result_df, use_container_width=True)

                # Export predictions
                csv_bytes = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="💾 Download Predictions CSV",
                    data=csv_bytes,
                    file_name="smart_home_energy_predictions.csv",
                    mime="text/csv",
                    type="primary"
                )

        except Exception as e:
            st.error(f"Error processing CSV file: {e}")


# ---------------------------------------------------------
# Tab 4: Sensor & Feature Glossary
# ---------------------------------------------------------
with tab4:
    st.markdown("### 📖 Feature & Sensor Reference Glossary")
    st.markdown(
        "Below is the complete dictionary mapping dataset technical variables (`T1`, `RH_1`, etc.) to human-friendly descriptions, room locations, and measurement units."
    )

    glossary_data = []
    for code, meta in FEATURE_METADATA.items():
        glossary_data.append({
            "Dataset Feature Code": f"`{code}`",
            "Human-Friendly Name": meta["label"],
            "Category Group": meta["group"],
            "Unit": meta["unit"],
            "Valid Range (Dataset)": f"{meta['min']} – {meta['max']} {meta['unit']}",
            "Median Default": f"{meta['default']} {meta['unit']}",
            "Description": meta["description"]
        })

    glossary_df = pd.DataFrame(glossary_data)
    st.dataframe(glossary_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 📚 Dataset & Project Context")
    st.markdown("""
    - **Dataset Source:** UCI Machine Learning Repository — *Appliances Energy Prediction Data Set*
    - **Target Variable:** `Appliances` (Energy use in Wh per 10-minute sampling interval)
    - **Target Environment:** Experimental low-energy house monitored over 4.5 months with ZigBee wireless sensor nodes.
    - **Primary Predictors:** Indoor temperatures & relative humidity across 9 rooms, outdoor meteorological conditions (temperature, pressure, humidity, wind speed, dew point), and lighting system energy consumption.
    """)
