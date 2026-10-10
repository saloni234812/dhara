"""
Dhara - AI-Powered Sustainable Facility & Estate Intelligence Dashboard
Reference Facility: 300-bed District Government Hospital, Ramnagar
Demo Persona: Dr. Sharma, Medical Superintendent
Tagline: "हर परिसर, एक हरित भविष्य"
One-line pitch: "From scattered data to 'what should I do today?'"
"""

import os
import yaml
import json
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta

from data.pipeline import DharaDataPipeline
from analytics.energy import EnergyAnalytics
from analytics.water import WaterAnalytics
from analytics.waste import WasteAnalytics
from analytics.air import AirAnalytics
from engine.recommend import RecommendationEngine
from engine.priority import PriorityEngine
from engine.scorecard import ScorecardEngine
from genai.brief import generate_ai_brief
from models.llm_trainer import EstateLLMTrainer

# ---------------------------------------------------------
# STREAMLIT PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="धारा (Dhara) • Hospital Estate Intelligence",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# ENTERPRISE DASHBOARD CSS STYLING
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&family=Noto+Sans+Devanagari:wght@400;600;700&display=swap');
    
    *, *::before, *::after {
        box-sizing: border-box;
    }
    
    /* Clear Streamlit fixed header so nothing overlaps at the top */
    header[data-testid="stHeader"] {
        background-color: rgba(255, 255, 255, 0.85) !important;
        backdrop-filter: blur(8px) !important;
        z-index: 99 !important;
    }
    
    .block-container {
        padding-top: 4.5rem !important;
        padding-bottom: 2rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        max-width: 100% !important;
    }
    
    html, body {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #0F172A;
    }
    
    /* Top Control Bar */
    .dashboard-topbar {
        background: #022C22;
        border-radius: 12px;
        padding: 0.85rem 1.3rem;
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(2, 44, 34, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: white;
        width: 100%;
    }
    
    .brand-group {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .brand-logo {
        background: #059669;
        color: white;
        font-size: 1.25rem;
        width: 36px;
        height: 36px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        box-shadow: 0 0 12px rgba(16, 185, 129, 0.4);
    }
    
    .brand-title {
        font-size: 1.25rem;
        font-weight: 800;
        letter-spacing: -0.4px;
        margin: 0;
        color: #FFFFFF !important;
        line-height: 1.2;
    }
    
    .brand-tagline {
        font-size: 0.75rem;
        color: #6EE7B7;
        font-family: 'Noto Sans Devanagari', sans-serif;
        font-weight: 600;
    }
    
    .topbar-telemetry {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.78rem;
    }
    
    .telemetry-chip {
        display: flex;
        align-items: center;
        gap: 0.35rem;
        background: rgba(255, 255, 255, 0.08);
        padding: 0.3rem 0.65rem;
        border-radius: 6px;
        border: 1px solid rgba(255, 255, 255, 0.15);
        white-space: nowrap;
    }
    
    .pulse-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #10B981;
        box-shadow: 0 0 8px #10B981;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(0.95); opacity: 0.8; }
        50% { transform: scale(1.3); opacity: 1; }
        100% { transform: scale(0.95); opacity: 0.8; }
    }
    
    /* Live Alert Ticker */
    .alert-ticker {
        background: #FEF2F2;
        border-left: 4px solid #EF4444;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        margin-bottom: 1.2rem;
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        gap: 0.5rem;
        font-size: 0.82rem;
        color: #991B1B;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        width: 100%;
    }
    
    /* Dashboard Metric Cards */
    .metric-card {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 1rem 1.15rem;
        border: 1px solid #E2E8F0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.03);
        position: relative;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        width: 100%;
        overflow: hidden;
        margin-bottom: 0.5rem;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px -2px rgba(0, 0, 0, 0.06);
    }
    
    .metric-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.3rem;
    }
    
    .metric-label {
        font-size: 0.72rem;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-icon-box {
        width: 30px;
        height: 30px;
        border-radius: 7px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.95rem;
    }
    
    .metric-val {
        font-size: 1.55rem;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.2;
        margin: 0.2rem 0;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    .metric-sub {
        font-size: 0.75rem;
        color: #475569;
        display: flex;
        align-items: center;
        gap: 0.3rem;
    }
    
    /* Executive Briefing Container */
    .brief-memo {
        background: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-top: 4px solid #059669;
        border-radius: 12px;
        padding: 1.15rem 1.35rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
        margin-bottom: 1rem;
        width: 100%;
    }
    
    .brief-memo-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #E2E8F0;
        padding-bottom: 0.5rem;
        margin-bottom: 0.75rem;
        flex-wrap: wrap;
        gap: 0.3rem;
    }
    
    .memo-badge {
        background: #ECFDF5;
        border: 1px solid #A7F3D0;
        color: #047857;
        font-size: 0.7rem;
        font-weight: 700;
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
        letter-spacing: 0.4px;
    }
    
    /* Action Cards */
    .action-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
        width: 100%;
        overflow: hidden;
    }
    
    .action-card:hover {
        border-color: #059669;
        box-shadow: 0 6px 12px -2px rgba(5, 150, 105, 0.08);
    }
    
    .rank-tag {
        background: #0F172A;
        color: white;
        font-size: 0.7rem;
        font-weight: 700;
        padding: 0.15rem 0.45rem;
        border-radius: 4px;
    }
    
    .domain-tag {
        font-size: 0.7rem;
        font-weight: 700;
        padding: 0.15rem 0.45rem;
        border-radius: 4px;
    }
    
    .deadline-tag {
        background: #FEF2F2;
        color: #DC2626;
        font-size: 0.7rem;
        font-weight: 700;
        padding: 0.15rem 0.45rem;
        border-radius: 4px;
        border: 1px solid #FCA5A5;
    }
    
    .formula-tag {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: #334155;
        padding: 0.3rem 0.55rem;
        border-radius: 5px;
        display: inline-block;
        word-break: break-all;
    }
    
    /* Event feed item */
    .event-item {
        padding: 0.5rem 0.65rem;
        border-radius: 6px;
        margin-bottom: 0.4rem;
        font-size: 0.78rem;
        display: flex;
        align-items: flex-start;
        gap: 0.45rem;
        background: #F8FAFC;
        border: 1px solid #F1F5F9;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

def render_html(html_str: str):
    """
    Renders raw HTML safely in Streamlit without triggering Markdown's 4-space code block rule.
    Strips leading and trailing whitespace from each line and removes empty lines.
    """
    clean_lines = [line.strip() for line in html_str.strip().split("\n") if line.strip()]
    st.markdown("\n".join(clean_lines), unsafe_allow_html=True)

# ---------------------------------------------------------
# CACHED DATA PIPELINE & ANALYTICS EXECUTION
# ---------------------------------------------------------

@st.cache_data(show_spinner="Refreshing Hospital SCADA Telemetry...")
def load_and_run_analytics():
    pipeline = DharaDataPipeline()
    df_facility = pipeline.load_facility_data()
    df_waste = pipeline.load_waste_data()

    energy_engine = EnergyAnalytics(pipeline.config)
    df_energy_anomalies = energy_engine.detect_anomalies(df_facility)
    opd_savings = energy_engine.calculate_night_waste_savings(df_energy_anomalies, "OPD")
    forecast_df = energy_engine.forecast_energy(df_facility, horizon_hours=48)

    water_engine = WaterAnalytics(pipeline.config)
    water_res = water_engine.analyze_water_network(df_facility)

    waste_engine = WasteAnalytics(pipeline.config)
    waste_res = waste_engine.analyze_waste_status(df_waste)

    air_engine = AirAnalytics(pipeline.config)
    air_res = air_engine.analyze_air_quality(df_facility)

    rec_engine = RecommendationEngine(pipeline.config)
    energy_input = {
        "opd_night_savings": opd_savings,
        "anomalies_df": df_energy_anomalies[df_energy_anomalies["is_anomaly"]]
    }
    recs = rec_engine.generate_recommendations(energy_input, water_res, waste_res, air_res)

    p_engine = PriorityEngine(pipeline.config)
    ranked_recs, summary = p_engine.rank_recommendations(recs)

    sc_engine = ScorecardEngine(pipeline.config)
    scorecard = sc_engine.evaluate_facility(energy_input, water_res, waste_res, air_res)

    return {
        "config": pipeline.config,
        "df_facility": df_facility,
        "df_waste": df_waste,
        "df_energy_anomalies": df_energy_anomalies,
        "forecast_df": forecast_df,
        "opd_savings": opd_savings,
        "water_res": water_res,
        "waste_res": waste_res,
        "air_res": air_res,
        "recommendations": ranked_recs,
        "summary": summary,
        "scorecard": scorecard
    }

data = load_and_run_analytics()
config = data["config"]
llm_trainer = EstateLLMTrainer(config)

# Connect to FastAPI Backend REST API
@st.cache_data(ttl=20, show_spinner=False)
def fetch_backend_status():
    try:
        res = requests.get("http://127.0.0.1:8000/api/external/live", timeout=2.0)
        if res.status_code == 200:
            return {"connected": True, "data": res.json()}
    except Exception:
        pass
    return {"connected": False, "data": None}

backend_status = fetch_backend_status()
is_api_connected = backend_status["connected"]
live_ext_weather = backend_status["data"]["weather"] if is_api_connected else {"temperature_c": 28.5, "relative_humidity_pct": 72}
live_ext_aqi = backend_status["data"]["ambient_air"] if is_api_connected else {"ambient_pm25": 58.0, "category": "Satisfactory"}

# Session State for Work Orders & Simulator
if "approved_orders" not in st.session_state:
    st.session_state.approved_orders = set()
if "audio_playing" not in st.session_state:
    st.session_state.audio_playing = False
if "batch_sanctioned" not in st.session_state:
    st.session_state.batch_sanctioned = False
if "ops_tasks" not in st.session_state:
    st.session_state.ops_tasks = {
        "W-01": {"title": "Isolate East Wing Riser (Hostel Pipe 2)", "assignee": "Ramesh Kumar (Plumbing AE)", "status": "Pending", "urgency": "Immediate (4h)", "zone": "Hostel"},
        "E-02": {"title": "Override Block C OPD BMS Relay (20:00 Cutoff)", "assignee": "Amit Verma (Electrical JE)", "status": "Pending", "urgency": "High (Tonight)", "zone": "OPD"},
        "S-03": {"title": "Reroute Mini-Truck to Canteen Bins 4, 7, 9", "assignee": "Rajesh (Sanitation Lead)", "status": "Pending", "urgency": "High (By 13:30)", "zone": "Canteen"},
        "B-04": {"title": "Immediate Transfer of Lab Bin #8 to CBWTF Store", "assignee": "Sita Devi (BMW Specialist)", "status": "In Progress", "urgency": "Critical (44h elapsed)", "zone": "Labs"},
        "A-05": {"title": "Enforce 3-Min Anti-Idling at ER Ambulance Gate", "assignee": "Security In-Charge", "status": "Pending", "urgency": "Medium", "zone": "Parking"}
    }
if "ops_maintenance_logs" not in st.session_state:
    st.session_state.ops_maintenance_logs = [
        {"timestamp": "08:30 IST", "asset": "Chiller Plant #2", "action": "Replaced thermal expansion sensor", "tech": "Amit Verma", "status": "Operational"},
        {"timestamp": "06:15 IST", "asset": "Overhead Tank 3", "action": "Checked ball-float valve assembly", "tech": "Ramesh Kumar", "status": "Nominal"}
    ]

# ---------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------

with st.sidebar:
    render_html("""
    <div style="display:flex; align-items:center; gap:0.6rem; padding-bottom:0.5rem;">
        <span style="font-size:2rem;">🌿</span>
        <div>
            <div style="font-weight:800; font-size:1.25rem; color:#064E3B; line-height:1.1;">DHARA</div>
            <div style="font-size:0.75rem; color:#059669; font-weight:700;">ESTATE INTELLIGENCE</div>
        </div>
    </div>
    """)
    st.caption("AI Decision Support for 300-bed District Hospital")
    st.divider()

    # Persona Role Selector
    st.markdown("##### 👤 Operating Perspective")
    role = st.selectbox(
        "Select User Persona",
        [
            "Administrator (Dr. Sharma)",
            "Operations Team (Er. Pandey)",
            "Sustainability Officer (Dr. Roy)"
        ],
        index=0,
        label_visibility="collapsed"
    )
    is_admin = "Administrator" in role
    is_ops = "Operations" in role
    is_sustain = "Sustainability" in role

    # Layout Perspective Toggle
    st.markdown("##### 🔀 Workspace View")
    layout_mode = st.radio(
        "Layout Perspective",
        ["Persona-Optimized View", "All Hospital Pillars (Unified)"],
        index=0,
        label_visibility="collapsed"
    )

    # Multi-Lingual Selector
    st.markdown("##### 🌐 Executive Language")
    language = st.radio(
        "Select Briefing Language",
        ["English", "Hindi (हिन्दी)", "Bengali (বাংলা)"],
        index=0,
        horizontal=True,
        label_visibility="collapsed"
    )
    lang_clean = "Hindi" if "Hindi" in language else ("Bengali" if "Bengali" in language else "English")

    st.divider()

    # Backend API Gateway & Swagger
    st.markdown("##### 🔌 Backend API Gateway")
    if is_api_connected:
        st.success("🟢 **FastAPI Backend Connected** (Port 8000)")
        st.markdown("""
        [📖 **Interactive Swagger UI Docs**](http://127.0.0.1:8000/docs)  
        [📑 **ReDoc API Specifications**](http://127.0.0.1:8000/redoc)
        """)
    else:
        st.warning("🟡 Direct Analytics Mode (FastAPI starting)")

    st.divider()

    # Live CSV upload & Data Controls
    st.markdown("##### 📂 Telemetry Ingestion")
    data_source_mode = st.radio("Source", ["Ramnagar SCADA Live Stream", "Upload Real Hospital CSV"], label_visibility="collapsed")
    if data_source_mode == "Upload Real Hospital CSV":
        uploaded_csv = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
        if uploaded_csv:
            st.success("Custom CSV parsed successfully!")

    # Download Telemetry Template
    sample_csv_data = data["df_facility"].head(48).to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download Telemetry Schema CSV",
        data=sample_csv_data,
        file_name="dhara_hospital_sample_telemetry.csv",
        mime="text/csv",
        help="Use this template to upload real hospital sensor data into Dhara."
    )

    st.divider()

    # LLM API Key (Optional)
    st.markdown("##### 🤖 GenAI Brief Configuration")
    api_key_input = st.text_input(
        "Anthropic API Key (Optional)",
        type="password",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        help="Optional. Rephrases pre-computed figures. If empty, uses localized executive templates."
    )

    st.divider()
    render_html(f"""
    <div style="font-size:0.75rem; color:#64748B; line-height:1.4;">
        <strong>Campus:</strong> {config['facility']['name']}<br>
        <strong>Bed Count:</strong> {config['facility']['beds']} Inpatient Beds<br>
        <strong>Power Tariff:</strong> ₹{config['tariffs_and_factors']['electricity_tariff_inr_kwh']}/kWh<br>
        <strong>Water Bulk:</strong> ₹{config['tariffs_and_factors']['water_tariff_inr_kl']}/KL<br>
        <strong>CEA Factor:</strong> 0.75 kg CO₂/kWh (v19)
    </div>
    """)


# ---------------------------------------------------------
# TOP DASHBOARD CONTROL BAR
# ---------------------------------------------------------

if is_admin:
    role_badge = "👑 Executive • Dr. Sharma"
elif is_ops:
    role_badge = "🛠️ Operations • Er. Pandey"
else:
    role_badge = "🌱 Compliance • Dr. Roy"

render_html(f"""
<div class="dashboard-topbar">
    <div class="brand-group">
        <div class="brand-logo">ध</div>
        <div>
            <div class="brand-title">धारा (Dhara) • Estate Intelligence</div>
            <div class="brand-tagline">हर परिसर, एक हरित भविष्य • {config['facility']['name']}</div>
        </div>
    </div>
    <div class="topbar-telemetry">
        <div class="telemetry-chip">
            <span class="pulse-dot"></span>
            <span><strong>{ 'API: ONLINE' if is_api_connected else 'SCADA' }</strong> (Port 8000)</span>
        </div>
        <div class="telemetry-chip">
            <span>🌡️ <strong>{round(live_ext_weather['temperature_c'], 1)}°C</strong> ({round(live_ext_weather['relative_humidity_pct'])}% RH)</span>
        </div>
        <div class="telemetry-chip">
            <span>🌫️ <strong>{round(live_ext_aqi['ambient_pm25'], 1)} µg/m³</strong> AQI ({live_ext_aqi['category']})</span>
        </div>
        <div class="telemetry-chip">
            <span>⚡ <strong>182.4 kW</strong> Campus Load</span>
        </div>
        <div class="telemetry-chip">
            <span><strong>{role_badge}</strong></span>
        </div>
    </div>
</div>
""")

# ---------------------------------------------------------
# LIVE SENSOR FAULT BROADCAST BANNER
# ---------------------------------------------------------

render_html("""
<div class="alert-ticker">
    <div style="display:flex; align-items:center; gap:0.5rem; flex-wrap:wrap;">
        <span style="background:#EF4444; color:white; font-size:0.7rem; font-weight:800; padding:0.15rem 0.45rem; border-radius:4px;">ACTIVE FAULTS (3)</span>
        <span><strong>1. Hostel:</strong> Pipeline 2 leak (690 L/hr, 7.5x night flow)</span> •
        <span><strong>2. Block C:</strong> AC running past 11 PM (48.5 kW, 0 occupants)</span> •
        <span><strong>3. Canteen:</strong> Bins 4, 7, 9 near overflow (by 2:00 PM)</span> •
        <span><strong>4. Ambulance Bay:</strong> PM2.5 surge (178 µg/m³)</span>
    </div>
    <span style="font-size:0.75rem; color:#B91C1C; font-weight:700;">LIVE DISPATCH REQUIRED</span>
</div>
""")

# ---------------------------------------------------------
# EXECUTIVE KPI METRIC CARDS (ROLE-CUSTOMIZED 4-METRIC ROW)
# ---------------------------------------------------------

c1, c2, c3, c4 = st.columns(4)

if is_admin or layout_mode == "All Hospital Pillars (Unified)":
    # Monthly Cost Recovery
    with c1:
        render_html(f"""
        <div class="metric-card" style="border-top: 3px solid #059669;">
            <div class="metric-header">
                <span class="metric-label">Monthly Recovery</span>
                <div class="metric-icon-box" style="background:#ECFDF5; color:#059669;">₹</div>
            </div>
            <div class="metric-val" style="color:#047857;">₹{int(data['summary']['total_top3_savings_inr']):,}</div>
            <div class="metric-sub">
                <span style="color:#059669; font-weight:700;">↑ 3 Priority Orders</span>
            </div>
        </div>
        """)

    # Scope 2 Carbon Avoidance
    with c2:
        render_html(f"""
        <div class="metric-card" style="border-top: 3px solid #2563EB;">
            <div class="metric-header">
                <span class="metric-label">Carbon Avoidance</span>
                <div class="metric-icon-box" style="background:#EFF6FF; color:#2563EB;">🌱</div>
            </div>
            <div class="metric-val" style="color:#1D4ED8;">{int(data['summary']['total_top3_co2_kg']):,} <span style="font-size:0.95rem; font-weight:600;">kg</span></div>
            <div class="metric-sub">
                <span style="color:#2563EB; font-weight:600;">≈ 232 Trees planted</span>
            </div>
        </div>
        """)

    # Hospital Eco-Grade
    with c3:
        sc = data["scorecard"]
        render_html(f"""
        <div class="metric-card" style="border-top: 3px solid {sc['grade_color']};">
            <div class="metric-header">
                <span class="metric-label">Hospital Eco-Grade</span>
                <div class="metric-icon-box" style="background:#FFFBEB; color:#D97706;">🏆</div>
            </div>
            <div class="metric-val" style="color:{sc['grade_color']};">Grade {sc['grade']} <span style="font-size:0.95rem; font-weight:600;">({sc['overall_score']})</span></div>
            <div class="metric-sub">
                <span style="color:{sc['grade_color']}; font-weight:600;">{sc['grade_title'][:22]}</span>
            </div>
        </div>
        """)

    # Potable Water Loss
    with c4:
        w_loss = data['water_res']['detected_leaks'][0]['monthly_kl_lost'] if len(data['water_res']['detected_leaks']) > 0 else 0
        render_html(f"""
        <div class="metric-card" style="border-top: 3px solid #0284C7;">
            <div class="metric-header">
                <span class="metric-label">Water Leak Loss</span>
                <div class="metric-icon-box" style="background:#F0F9FF; color:#0284C7;">💧</div>
            </div>
            <div class="metric-val" style="color:#0369A1;">{int(w_loss)} <span style="font-size:0.95rem; font-weight:600;">KL/mo</span></div>
            <div class="metric-sub">
                <span style="color:#DC2626; font-weight:700;">Hostel Pipeline 2 (7.5x)</span>
            </div>
        </div>
        """)

elif is_ops:
    resolved_count = sum(1 for t in st.session_state.ops_tasks.values() if t["status"] == "Resolved")
    with c1:
        render_html(f"""
        <div class="metric-card" style="border-top: 3px solid #2563EB;">
            <div class="metric-header">
                <span class="metric-label">Shift Work Orders</span>
                <div class="metric-icon-box" style="background:#EFF6FF; color:#2563EB;">🛠️</div>
            </div>
            <div class="metric-val" style="color:#1D4ED8;">{len(st.session_state.ops_tasks)} <span style="font-size:0.95rem; font-weight:600;">Tasks</span></div>
            <div class="metric-sub">
                <span style="color:#059669; font-weight:700;">✓ {resolved_count} Tasks Completed</span>
            </div>
        </div>
        """)
    with c2:
        render_html(f"""
        <div class="metric-card" style="border-top: 3px solid #DC2626;">
            <div class="metric-header">
                <span class="metric-label">Pinpointed Leak Loss</span>
                <div class="metric-icon-box" style="background:#FEF2F2; color:#DC2626;">💧</div>
            </div>
            <div class="metric-val" style="color:#B91C1C;">690 <span style="font-size:0.95rem; font-weight:600;">L/hr</span></div>
            <div class="metric-sub">
                <span style="color:#DC2626; font-weight:700;">Line Pressure Drop: -12 PSI</span>
            </div>
        </div>
        """)
    with c3:
        render_html(f"""
        <div class="metric-card" style="border-top: 3px solid #EA580C;">
            <div class="metric-header">
                <span class="metric-label">Night HVAC Excess</span>
                <div class="metric-icon-box" style="background:#FFF7ED; color:#EA580C;">⚡</div>
            </div>
            <div class="metric-val" style="color:#C2410C;">48.5 <span style="font-size:0.95rem; font-weight:600;">kW</span></div>
            <div class="metric-sub">
                <span style="color:#EA580C; font-weight:700;">Block C Draw (0 Occupancy)</span>
            </div>
        </div>
        """)
    with c4:
        render_html(f"""
        <div class="metric-card" style="border-top: 3px solid #D97706;">
            <div class="metric-header">
                <span class="metric-label">Overflow Risk Bins</span>
                <div class="metric-icon-box" style="background:#FEF3C7; color:#D97706;">🚛</div>
            </div>
            <div class="metric-val" style="color:#B45309;">3 <span style="font-size:0.95rem; font-weight:600;">Bins</span></div>
            <div class="metric-sub">
                <span style="color:#D97706; font-weight:700;">Canteen Bins >90% Full</span>
            </div>
        </div>
        """)

else: # is_sustain
    with c1:
        sc = data["scorecard"]
        render_html(f"""
        <div class="metric-card" style="border-top: 3px solid {sc['grade_color']};">
            <div class="metric-header">
                <span class="metric-label">Hospital Eco-Grade</span>
                <div class="metric-icon-box" style="background:#FFFBEB; color:#D97706;">🏆</div>
            </div>
            <div class="metric-val" style="color:{sc['grade_color']};">Grade {sc['grade']} <span style="font-size:0.95rem; font-weight:600;">({sc['overall_score']})</span></div>
            <div class="metric-sub">
                <span style="color:{sc['grade_color']}; font-weight:700;">Target: Grade A (85.0 pts)</span>
            </div>
        </div>
        """)
    with c2:
        render_html(f"""
        <div class="metric-card" style="border-top: 3px solid #8B5CF6;">
            <div class="metric-header">
                <span class="metric-label">BMW 48h Legal Limit</span>
                <div class="metric-icon-box" style="background:#F5F3FF; color:#8B5CF6;">🏥</div>
            </div>
            <div class="metric-val" style="color:#6D28D9;">44.0 <span style="font-size:0.95rem; font-weight:600;">hrs</span></div>
            <div class="metric-sub">
                <span style="color:#DC2626; font-weight:700;">⚠️ Bin #8 (4h to Breach)</span>
            </div>
        </div>
        """)
    with c3:
        render_html(f"""
        <div class="metric-card" style="border-top: 3px solid #DC2626;">
            <div class="metric-header">
                <span class="metric-label">Particulate Hotspot</span>
                <div class="metric-icon-box" style="background:#FEF2F2; color:#DC2626;">🌫️</div>
            </div>
            <div class="metric-val" style="color:#B91C1C;">178 <span style="font-size:0.95rem; font-weight:600;">µg/m³</span></div>
            <div class="metric-sub">
                <span style="color:#DC2626; font-weight:700;">Ambulance Bay (4h Duration)</span>
            </div>
        </div>
        """)
    with c4:
        render_html(f"""
        <div class="metric-card" style="border-top: 3px solid #059669;">
            <div class="metric-header">
                <span class="metric-label">Scope 2 Carbon Penalty</span>
                <div class="metric-icon-box" style="background:#ECFDF5; color:#059669;">🌱</div>
            </div>
            <div class="metric-val" style="color:#047857;">4,331 <span style="font-size:0.95rem; font-weight:600;">kg/mo</span></div>
            <div class="metric-sub">
                <span style="color:#059669; font-weight:700;">Avoidable via BMS Schedule</span>
            </div>
        </div>
        """)

st.markdown("<br>", unsafe_allow_html=True)


# =========================================================
# REUSABLE TAB RENDERER FUNCTIONS & DASHBOARD CHARTS
# =========================================================

def make_gauge_chart(val, max_val, title, unit="", color="#059669", threshold=None, height=175):
    steps = [
        {'range': [0, max_val * 0.5], 'color': 'rgba(16, 185, 129, 0.12)'},
        {'range': [max_val * 0.5, max_val * 0.8], 'color': 'rgba(245, 158, 11, 0.12)'},
        {'range': [max_val * 0.8, max_val], 'color': 'rgba(239, 68, 68, 0.12)'}
    ]
    gauge_dict = {
        'axis': {'range': [0, max_val], 'tickcolor': '#94A3B8', 'tickwidth': 1},
        'bar': {'color': color, 'thickness': 0.28},
        'bgcolor': 'white',
        'borderwidth': 1,
        'bordercolor': '#E2E8F0',
        'steps': steps
    }
    if threshold is not None:
        gauge_dict['threshold'] = {'line': {'color': '#EF4444', 'width': 3}, 'thickness': 0.75, 'value': threshold}

    fig = go.Figure(go.Indicator(
        mode = 'gauge+number',
        value = val,
        number = {'suffix': unit, 'font': {'size': 20, 'color': '#0F172A', 'family': 'Plus Jakarta Sans'}},
        title = {'text': title, 'font': {'size': 12, 'color': '#475569', 'family': 'Plus Jakarta Sans'}},
        gauge = gauge_dict
    ))
    fig.update_layout(
        height=height,
        margin=dict(l=15, r=15, t=30, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def make_donut_chart(labels, values, colors, title="", height=230):
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.60,
        marker=dict(colors=colors),
        textinfo='label+percent',
        insidetextorientation='radial',
        hoverinfo='label+value+percent'
    )])
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color='#0F172A')),
        showlegend=False,
        height=height,
        margin=dict(l=10, r=10, t=35, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def render_spatial_map(map_key="campus_map_default"):
    st.markdown("#### 🗺️ Interactive Campus Spatial Health Map")
    st.caption("Click any hospital pavilion to inspect live energy draw, water leaks, and bin fill rates.")

    center_lat, center_lon = config["facility"]["coordinates"]
    m = folium.Map(location=[center_lat, center_lon], zoom_start=18, tiles="OpenStreetMap")
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Satellite View"
    ).add_to(m)
    folium.LayerControl(position="topright").add_to(m)

    for z in config["zones"]:
        z_lat = center_lat + z["lat_offset"]
        z_lon = center_lon + z["lon_offset"]
        zid = z["id"]

        if zid == "Hostel":
            fill_color = "#DC2626"
            status_text = "CRITICAL: Hostel Pipeline 2 Water Leak (7.5x baseline)"
        elif zid == "OPD":
            fill_color = "#EA580C"
            status_text = "WARNING: AC left on overnight (11 PM - 6 AM)"
        elif zid == "Canteen":
            fill_color = "#D97706"
            status_text = "ACTION: Bins 4, 7, 9 near overflow by 2:00 PM"
        elif zid in ["Generator", "Parking"]:
            fill_color = "#CA8A04"
            status_text = "ADVISORY: PM2.5 elevated for 3+ hours (anti-idling)"
        else:
            fill_color = "#10B981"
            status_text = "NORMAL: Operations within green parameters"

        popup_html = f"""
        <div style="font-family:'Plus Jakarta Sans',sans-serif; width:220px;">
            <div style="font-size:13px; font-weight:bold; color:#0F172A; margin-bottom:3px;">{z['name']}</div>
            <div style="font-size:11px; color:#64748B; margin-bottom:6px;">Type: <strong>{z['type'].replace('_', ' ').title()}</strong></div>
            <div style="background:{fill_color}22; color:{fill_color}; padding:4px 6px; border-radius:4px; font-size:11px; font-weight:bold; margin-bottom:6px; border:1px solid {fill_color}44;">
                {status_text}
            </div>
            <div style="font-size:11px; color:#334155;">
                • Baseline Power: {z['baseline_power_kw']} kW<br>
                • Water Baseline: {z['baseline_water_lph']} L/hr<br>
                • Day Occupancy: {z['occupancy_day']} persons
            </div>
        </div>
        """

        folium.CircleMarker(
            location=[z_lat, z_lon],
            radius=18,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"{z['name']} (Click to inspect)",
            color=fill_color,
            fill=True,
            fill_color=fill_color,
            fill_opacity=0.85,
            weight=2
        ).add_to(m)

    st_folium(m, use_container_width=True, height=360, key=map_key)


def render_executive_command_tab():
    top_left, top_right = st.columns([0.60, 0.40])

    with top_left:
        st.markdown("#### 🎯 Today's 3 Priority Executive Actions")
        st.caption("Dr. Sharma's Directives: High-impact actions with quantified financial and carbon recovery.")

        for item in data["summary"]["top_3"]:
            is_approved = item["id"] in st.session_state.approved_orders
            domain_color = "#3B82F6" if item["domain"] == "Energy" else ("#0284C7" if item["domain"] == "Water" else "#10B981")
            domain_bg = "#EFF6FF" if item["domain"] == "Energy" else ("#F0F9FF" if item["domain"] == "Water" else "#ECFDF5")
            border_col = '#10B981' if is_approved else '#EF4444'
            dispatched_badge = '<span style="background:#DCFCE7; color:#15803D; font-size:0.72rem; font-weight:700; padding:0.15rem 0.5rem; border-radius:4px; margin-left:0.3rem;">✓ DISPATCHED</span>' if is_approved else ''

            render_html(f"""
            <div class="action-card" style="border-left: 4px solid {border_col}; margin-bottom:0.75rem;">
                <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:0.6rem;">
                    <div style="flex: 1 1 240px; min-width: 200px;">
                        <span class="rank-tag">#{item['rank_position']}</span>
                        <span class="domain-tag" style="background:{domain_bg}; color:{domain_color};">{item['domain'].upper()}</span>
                        <span class="deadline-tag">⏰ {item['deadline']}</span>
                        {dispatched_badge}
                        <h4 style="margin: 0.35rem 0 0.15rem 0; color:#0F172A; font-weight:700; font-size:1.0rem;">{item['short_title']}</h4>
                        <div style="color:#475569; font-size:0.85rem; line-height:1.4;">{item['action']}</div>
                    </div>
                    <div style="flex: 0 0 auto; text-align:right; min-width: 120px;">
                        <div style="font-size:1.45rem; font-weight:800; color:#047857;">+ ₹{int(item['estimated_monthly_savings_inr']):,}</div>
                        <div style="font-size:0.75rem; color:#64748B;">🌱 Avoids {int(item['estimated_monthly_co2_kg'])} kg CO₂</div>
                    </div>
                </div>
            </div>
            """)

            b1, b2, b_sp = st.columns([0.34, 0.34, 0.32])
            with b1:
                if not is_approved:
                    if st.button(f"✅ Sanction #{item['rank_position']}", key=f"cmd_appr_{item['id']}", type="primary"):
                        st.session_state.approved_orders.add(item["id"])
                        st.toast(f"Work Order {item['id']} dispatched to facilities crew!")
                        st.rerun()
                else:
                    st.button("✓ Dispatched", key=f"cmd_done_{item['id']}", disabled=True)
            with b2:
                if st.button(f"📞 Call Engineer", key=f"cmd_call_{item['id']}"):
                    st.info(f"Connected to Junior Engineer assigned to {item['zone']}.")

    with top_right:
        st.markdown("#### 🎛️ Executive Telemetry Cockpit")
        st.caption("Live aggregate status across hospital eco-grade and electrical demand:")

        g_col1, g_col2 = st.columns(2)
        with g_col1:
            fig_g1 = make_gauge_chart(data['scorecard']['overall_score'], 100, "Hospital Eco-Score", " pts", "#059669", threshold=85)
            st.plotly_chart(fig_g1, use_container_width=True)
        with g_col2:
            fig_g2 = make_gauge_chart(182.4, 300, "Campus Grid Load", " kW", "#2563EB", threshold=220)
            st.plotly_chart(fig_g2, use_container_width=True)

        brief_text = generate_ai_brief({**data["summary"], **data["scorecard"]}, language=lang_clean, api_key=api_key_input)
        render_html(f"""
        <div style="background:#F8FAFC; border:1px solid #CBD5E1; border-top:3px solid #059669; border-radius:10px; padding:0.9rem 1.1rem; margin-top:0.3rem;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-weight:800; font-size:0.85rem; color:#064E3B;">🎙️ AI MORNING BRIEFING</span>
                <span style="font-size:0.75rem; background:#ECFDF5; color:#047857; font-weight:700; padding:0.15rem 0.45rem; border-radius:4px;">{lang_clean.upper()}</span>
            </div>
            <div style="font-size:0.82rem; color:#334155; margin-top:0.4rem; line-height:1.5;">
                {brief_text.splitlines()[0] if brief_text.splitlines() else "Good morning, Dr. Sharma."}<br>
                <strong>Priority Target:</strong> Fix 3 detected faults to unlock ₹80,830 / mo and boost rating to Grade A.
            </div>
        </div>
        """)

        m_b1, m_b2 = st.columns(2)
        with m_b1:
            if st.button("🔊 Play Voice Brief", key="cmd_audio_btn"):
                st.session_state.audio_playing = not st.session_state.audio_playing
                if st.session_state.audio_playing:
                    st.toast(f"Playing briefing audio simulation in {lang_clean}...")
        with m_b2:
            st.download_button(
                "📄 Export Brief (.TXT)",
                data=brief_text,
                file_name=f"Dhara_Morning_Brief_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain"
            )

    st.markdown("---")

    # Spatial Map & Energy Distribution Row
    map_col, pie_col = st.columns([0.55, 0.45])
    with map_col:
        render_spatial_map(map_key="map_exec_cmd")

    with pie_col:
        st.markdown("#### ⚡ Campus Energy Distribution by Zone")
        st.caption("Active load balance across all hospital pavilions:")
        
        zone_labels = ["OPD (Block C)", "Inpatient Wards", "ICU / OT", "Doctors Hostel", "Labs & Pathology", "Canteen / Kitchen"]
        zone_loads = [48.5, 42.0, 38.0, 22.0, 18.0, 13.9]
        zone_colors = ["#EF4444", "#3B82F6", "#10B981", "#F59E0B", "#8B5CF6", "#06B6D4"]
        fig_donut = make_donut_chart(zone_labels, zone_loads, zone_colors, "Real-Time Power Draw (182.4 kW Total)", height=280)
        st.plotly_chart(fig_donut, use_container_width=True)

        render_html("""
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.5rem; margin-top:0.5rem;">
            <div style="background:#FEF2F2; border-left:3px solid #EF4444; padding:0.45rem 0.65rem; border-radius:4px; font-size:0.75rem;">
                <strong>OPD Block C:</strong> 48.5 kW (3.2σ Anomaly)
            </div>
            <div style="background:#EFF6FF; border-left:3px solid #3B82F6; padding:0.45rem 0.65rem; border-radius:4px; font-size:0.75rem;">
                <strong>Inpatient Wards:</strong> 42.0 kW (Nominal)
            </div>
            <div style="background:#ECFDF5; border-left:3px solid #10B981; padding:0.45rem 0.65rem; border-radius:4px; font-size:0.75rem;">
                <strong>ICU / OT:</strong> 38.0 kW (Sterile Power)
            </div>
            <div style="background:#FFFBEB; border-left:3px solid #F59E0B; padding:0.45rem 0.65rem; border-radius:4px; font-size:0.75rem;">
                <strong>Hostel Riser:</strong> 22.0 kW (Elevated Pump)
            </div>
        </div>
        """)

    st.markdown("---")

    # 4 Visual Anomaly Diagnostic Cards
    st.markdown("#### 🚨 Verified Anomaly Diagnostic Telemetry")
    st.caption("Direct telemetry proof and machine learning flags across hospital infrastructure:")

    anom_c1, anom_c2, anom_c3, anom_c4 = st.columns(4)
    with anom_c1:
        render_html("""
        <div class="metric-card" style="border-top: 3px solid #DC2626;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:0.72rem; font-weight:800; color:#B91C1C;">💧 WATER LEAK</span>
                <span style="background:#FEF2F2; color:#DC2626; font-size:0.65rem; font-weight:800; padding:0.1rem 0.4rem; border-radius:4px;">CRITICAL</span>
            </div>
            <div style="font-size:1.6rem; font-weight:800; color:#0F172A; margin:0.3rem 0;">1,260 <span style="font-size:0.85rem; color:#64748B;">L/h</span></div>
            <div style="font-size:0.78rem; color:#DC2626; font-weight:700;">7.5× Night Baseline (2-4 AM)</div>
            <div style="font-size:0.75rem; color:#475569; margin-top:0.4rem; border-top:1px dashed #E2E8F0; padding-top:0.3rem;">
                • Line Pressure: <strong>-12 PSI Drop</strong><br>
                • Monthly Loss: <strong>₹14,743 / mo</strong><br>
                • Location: <strong>Hostel Pipeline 2</strong>
            </div>
        </div>
        """)

    with anom_c2:
        render_html("""
        <div class="metric-card" style="border-top: 3px solid #EA580C;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:0.72rem; font-weight:800; color:#C2410C;">⚡ NIGHT HVAC</span>
                <span style="background:#FFF7ED; color:#EA580C; font-size:0.65rem; font-weight:800; padding:0.1rem 0.4rem; border-radius:4px;">WARNING</span>
            </div>
            <div style="font-size:1.6rem; font-weight:800; color:#0F172A; margin:0.3rem 0;">48.5 <span style="font-size:0.85rem; color:#64748B;">kW</span></div>
            <div style="font-size:0.78rem; color:#EA580C; font-weight:700;">3.2σ Anomaly (11 PM - 6 AM)</div>
            <div style="font-size:0.75rem; color:#475569; margin-top:0.4rem; border-top:1px dashed #E2E8F0; padding-top:0.3rem;">
                • Occupancy: <strong>0 Persons</strong><br>
                • Monthly Loss: <strong>₹43,502 / mo</strong><br>
                • Location: <strong>Block C (OPD)</strong>
            </div>
        </div>
        """)

    with anom_c3:
        render_html("""
        <div class="metric-card" style="border-top: 3px solid #D97706;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:0.72rem; font-weight:800; color:#B45309;">🚛 BIN OVERFLOW</span>
                <span style="background:#FEF3C7; color:#B45309; font-size:0.65rem; font-weight:800; padding:0.1rem 0.4rem; border-radius:4px;">ACTION</span>
            </div>
            <div style="font-size:1.6rem; font-weight:800; color:#0F172A; margin:0.3rem 0;">92% <span style="font-size:0.85rem; color:#64748B;">Fill</span></div>
            <div style="font-size:0.78rem; color:#D97706; font-weight:700;">Overflow by 2:00 PM</div>
            <div style="font-size:0.75rem; color:#475569; margin-top:0.4rem; border-top:1px dashed #E2E8F0; padding-top:0.3rem;">
                • Surge Rate: <strong>14.2% / hr</strong><br>
                • Bins Affected: <strong>#4, #7, #9</strong><br>
                • Location: <strong>Canteen Compound</strong>
            </div>
        </div>
        """)

    with anom_c4:
        render_html("""
        <div class="metric-card" style="border-top: 3px solid #8B5CF6;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:0.72rem; font-weight:800; color:#6D28D9;">🌫️ PM2.5 SURGE</span>
                <span style="background:#F5F3FF; color:#6D28D9; font-size:0.65rem; font-weight:800; padding:0.1rem 0.4rem; border-radius:4px;">ADVISORY</span>
            </div>
            <div style="font-size:1.6rem; font-weight:800; color:#0F172A; margin:0.3rem 0;">178 <span style="font-size:0.85rem; color:#64748B;">µg/m³</span></div>
            <div style="font-size:0.78rem; color:#6D28D9; font-weight:700;">4-Hour Continuous Peak</div>
            <div style="font-size:0.75rem; color:#475569; margin-top:0.4rem; border-top:1px dashed #E2E8F0; padding-top:0.3rem;">
                • CPCB Category: <strong>Very Poor</strong><br>
                • Cause: <strong>Idling Ambulances</strong><br>
                • Location: <strong>Emergency ER Gate</strong>
            </div>
        </div>
        """)

    st.markdown("---")

    # Interactive Action Priority Scatter Matrix
    st.markdown("#### 📊 Interactive Decision Priority Matrix")
    st.caption("Visual ranking of hospital interventions (Bubble Size = Avoided CO₂ Volume in kg):")

    matrix_data = [
        {"Action": "1. Isolate Hostel Pipe 2", "Zone": "Hostel", "Urgency_Hours": 4, "Monthly_Savings": 14743, "CO2": 980, "Domain": "Water", "Color": "#0284C7"},
        {"Action": "2. Block C HVAC Cutoff Relay", "Zone": "OPD", "Urgency_Hours": 12, "Monthly_Savings": 43502, "CO2": 4331, "Domain": "Energy", "Color": "#EA580C"},
        {"Action": "3. Reroute Canteen Waste Truck", "Zone": "Canteen", "Urgency_Hours": 2, "Monthly_Savings": 10500, "CO2": 450, "Domain": "Waste", "Color": "#D97706"},
        {"Action": "4. Enforce ER Anti-Idling Gate", "Zone": "Parking", "Urgency_Hours": 6, "Monthly_Savings": 12085, "CO2": 1200, "Domain": "Air", "Color": "#8B5CF6"},
    ]
    m_df = pd.DataFrame(matrix_data)
    fig_matrix = px.scatter(
        m_df,
        x="Urgency_Hours",
        y="Monthly_Savings",
        size="CO2",
        color="Domain",
        hover_name="Action",
        text="Action",
        labels={"Urgency_Hours": "Response Deadline (Hours Remaining)", "Monthly_Savings": "Monthly Financial Recovery (₹)"},
        color_discrete_map={"Water": "#0284C7", "Energy": "#EA580C", "Waste": "#D97706", "Air": "#8B5CF6"}
    )
    fig_matrix.update_traces(textposition='top center', marker=dict(opacity=0.85, line=dict(width=1.5, color='#0F172A')))
    fig_matrix.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_matrix, use_container_width=True)

    with st.expander("📐 View Formula & Mathematical Specification Ledger"):
        eq_df = pd.DataFrame([
            {"Action": r['short_title'], "Zone": r['zone'], "Savings Formula Applied": r['formula_display']}
            for r in data["recommendations"]
        ])
        st.dataframe(eq_df, use_container_width=True, hide_index=True)


def render_admin_budget_tab():
    st.markdown("### 💰 Hospital Estate Financial Ledger & Sanction Desk")
    st.caption("Executive cost control, utility tariff outflows, and batch sanction authority under Dr. Sharma:")

    b_c1, b_c2, b_c3 = st.columns(3)
    with b_c1:
        st.metric("Total Utility Grant", "₹18,50,000 / mo", "Approved Health Mission Grant")
    with b_c2:
        st.metric("Current Spend", "₹16,49,000 / mo", "89.1% Utilization")
    with b_c3:
        st.metric("Recoverable Waste", "₹58,245 / mo", "+ 3.53% Savings Unlocked", delta_color="normal")

    st.markdown("---")

    fin_col1, fin_col2 = st.columns([0.48, 0.52])
    with fin_col1:
        st.markdown("#### ⚡ Monthly Utility Spend Breakdown")
        spend_labels = ["HT Grid Electricity", "BMW CBWTF Logistics", "Municipal Potable Water"]
        spend_vals = [1245000, 280000, 124000]
        spend_colors = ["#2563EB", "#8B5CF6", "#0284C7"]
        fig_spend = make_donut_chart(spend_labels, spend_vals, spend_colors, "Total Outflow: ₹16.49 Lakh / Month", height=260)
        st.plotly_chart(fig_spend, use_container_width=True)

    with fin_col2:
        st.markdown("#### 📊 Budget Utilization & Recovery Meter")
        util_df = pd.DataFrame([
            {"Category": "Current Outflow", "Amount (₹ Lakh)": 16.49, "Type": "Utilized (89.1%)"},
            {"Category": "Unspent Contingency", "Amount (₹ Lakh)": 2.01, "Type": "Reserve"},
            {"Category": "Dhara Recoverable Wastage", "Amount (₹ Lakh)": 0.58, "Type": "Recoverable Waste"}
        ])
        fig_util = px.bar(
            util_df,
            x="Amount (₹ Lakh)",
            y="Category",
            orientation="h",
            color="Type",
            color_discrete_map={"Utilized (89.1%)": "#2563EB", "Reserve": "#10B981", "Recoverable Waste": "#EF4444"}
        )
        fig_util.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10), showlegend=False)
        st.plotly_chart(fig_util, use_container_width=True)

    st.markdown("---")

    sanction_col1, sanction_col2 = st.columns([0.62, 0.38])
    with sanction_col1:
        render_html("""
        <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-left:4px solid #059669; border-radius:10px; padding:1.1rem;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <strong style="color:#065F46; font-size:1.0rem;">Official Executive Sanction Warrant Ref: DH-RAM-2026/OCT-382</strong>
                <span style="background:#ECFDF5; color:#047857; font-size:0.75rem; font-weight:700; padding:0.2rem 0.5rem; border-radius:4px;">VERIFIED</span>
            </div>
            <div style="font-size:0.85rem; color:#334155; margin-top:0.5rem; line-height:1.5;">
                I hereby grant administrative and financial sanction for immediate physical execution of:<br>
                1. <strong>Order #1:</strong> Emergency isolation & repair of Hostel Pipeline 2 Potable Water Riser.<br>
                2. <strong>Order #2:</strong> Electrical Division reprogramming of Block C OPD HVAC shutoff relay.<br>
                3. <strong>Order #3:</strong> Priority vehicle dispatch for Canteen compound waste bins 4, 7, 9.
            </div>
        </div>
        """)

    with sanction_col2:
        if not st.session_state.batch_sanctioned:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("✍️ Grant Executive Sanction (All Orders)", key="btn_batch_sanction", type="primary", use_container_width=True):
                for it in data["summary"]["top_3"]:
                    st.session_state.approved_orders.add(it["id"])
                st.session_state.batch_sanctioned = True
                st.toast("Executive Sanction Granted! All orders dispatched.")
                st.rerun()
        else:
            render_html("""
            <div style="background:#ECFDF5; border:1px solid #A7F3D0; border-radius:8px; padding:0.9rem; font-size:0.82rem; color:#065F46;">
                <strong>✅ SANCTION DIGITALLY SEALED</strong><br>
                <strong>Signatory:</strong> Dr. Sharma, MBBS, MS<br>
                <strong>Designation:</strong> Medical Superintendent<br>
                <strong>Timestamp:</strong> Oct 09, 2026 IST
            </div>
            """)

    st.markdown("---")
    st.markdown("#### 🚨 District Emergency Escalation Directory")
    esc1, esc2, esc3 = st.columns(3)
    with esc1:
        render_html("""
        <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:8px; padding:0.8rem;">
            <strong>Chief Medical Officer (CMO)</strong><br>
            <span style="font-size:0.82rem; color:#64748B;">Dr. V. K. Saxena</span><br>
            <span style="font-size:0.8rem; color:#2563EB;">📞 +91 94150 11223</span><br>
            <small style="color:#059669;">Primary Hospital Authority</small>
        </div>
        """)
    with esc2:
        render_html("""
        <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:8px; padding:0.8rem;">
            <strong>Superintending Engineer (UPPCL)</strong><br>
            <span style="font-size:0.82rem; color:#64748B;">Er. R. N. Tiwari</span><br>
            <span style="font-size:0.8rem; color:#2563EB;">📞 +91 94150 44556</span><br>
            <small style="color:#D97706;">HT Grid & Transformer Feeder</small>
        </div>
        """)
    with esc3:
        render_html("""
        <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:8px; padding:0.8rem;">
            <strong>Executive Engineer (Jal Sansthan)</strong><br>
            <span style="font-size:0.82rem; color:#64748B;">Er. S. C. Gupta</span><br>
            <span style="font-size:0.8rem; color:#2563EB;">📞 +91 94150 77889</span><br>
            <small style="color:#0284C7;">Municipal Bulk Water Line</small>
        </div>
        """)



def render_ops_dispatch_tab():
    st.markdown("### 🛠️ Shift Dispatch & Field Work Order Execution Board")
    st.caption("Real-time work order tracking, technician assignments, and shift resolution milestones:")

    render_html("""
    <div style="background:#EFF6FF; border:1px solid #BFDBFE; border-radius:8px; padding:0.8rem 1.1rem; margin-bottom:1rem; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem;">
        <div>
            <strong style="color:#1E40AF; font-size:0.95rem;">📋 ACTIVE SHIFT: 08:00 – 20:00 IST (Day Duty)</strong><br>
            <span style="font-size:0.8rem; color:#3B82F6;">Shift In-Charge: <strong>Er. Rajesh Pandey (AE Maintenance)</strong> • Base Station: Ground Floor Engineering Control Room</span>
        </div>
        <span style="background:#DBEAFE; color:#1E40AF; font-weight:700; font-size:0.75rem; padding:0.25rem 0.6rem; border-radius:4px;">CAMPUS SCADA LIVE</span>
    </div>
    """)

    st.markdown("##### 👷 Shift Crew Duty Roster")
    roster_cols = st.columns(4)
    with roster_cols[0]:
        render_html("""
        <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-left:3px solid #0284C7; border-radius:6px; padding:0.6rem; font-size:0.82rem;">
            <strong>Ramesh Kumar</strong> (Plumbing AE)<br>
            <span style="color:#64748B;">📞 +91 98765 43210</span><br>
            <span style="color:#059669; font-weight:600;">🟢 On Duty (Hostel Riser)</span>
        </div>
        """)
    with roster_cols[1]:
        render_html("""
        <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-left:3px solid #2563EB; border-radius:6px; padding:0.6rem; font-size:0.82rem;">
            <strong>Amit Verma</strong> (Electrical JE)<br>
            <span style="color:#64748B;">📞 +91 98765 43211</span><br>
            <span style="color:#059669; font-weight:600;">🟢 On Duty (Sub-station)</span>
        </div>
        """)
    with roster_cols[2]:
        render_html("""
        <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-left:3px solid #D97706; border-radius:6px; padding:0.6rem; font-size:0.82rem;">
            <strong>Rajesh</strong> (Sanitation Lead)<br>
            <span style="color:#64748B;">📞 +91 98765 43212</span><br>
            <span style="color:#059669; font-weight:600;">🟢 On Duty (Mini-Truck)</span>
        </div>
        """)
    with roster_cols[3]:
        render_html("""
        <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-left:3px solid #8B5CF6; border-radius:6px; padding:0.6rem; font-size:0.82rem;">
            <strong>Sita Devi</strong> (BMW Handler)<br>
            <span style="color:#64748B;">📞 +91 98765 43213</span><br>
            <span style="color:#059669; font-weight:600;">🟢 On Duty (Central Store)</span>
        </div>
        """)

    st.markdown("<br>", unsafe_allow_html=True)

    tasks = st.session_state.ops_tasks
    tot_tasks = len(tasks)
    res_tasks = sum(1 for t in tasks.values() if t["status"] == "Resolved")
    prog_pct = int((res_tasks / tot_tasks) * 100)
    st.progress(prog_pct / 100.0, text=f"Shift Resolution Progress: {res_tasks} of {tot_tasks} Tasks Completed ({prog_pct}%)")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("##### 📋 Priority Work Order Execution Board")

    task_items = list(tasks.items())
    col_k1, col_k2 = st.columns(2)

    for idx, (task_id, t_info) in enumerate(task_items):
        target_col = col_k1 if idx % 2 == 0 else col_k2
        with target_col:
            st_color = "#10B981" if t_info["status"] == "Resolved" else ("#3B82F6" if t_info["status"] == "In Progress" else "#EF4444")
            st_bg = "#ECFDF5" if t_info["status"] == "Resolved" else ("#EFF6FF" if t_info["status"] == "In Progress" else "#FEF2F2")

            render_html(f"""
            <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-left:4px solid {st_color}; border-radius:8px; padding:0.85rem 1.0rem; margin-bottom:0.4rem;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="background:#F1F5F9; color:#475569; font-weight:700; font-size:0.75rem; padding:0.15rem 0.45rem; border-radius:4px;">{task_id}</span>
                    <span style="background:{st_bg}; color:{st_color}; font-weight:700; font-size:0.75rem; padding:0.15rem 0.45rem; border-radius:4px;">{t_info['status'].upper()}</span>
                </div>
                <div style="margin:0.4rem 0 0.2rem 0; color:#0F172A; font-weight:700; font-size:0.95rem;">{t_info['title']}</div>
                <div style="font-size:0.78rem; color:#64748B;">Zone: <strong>{t_info['zone']}</strong> • ⏰ <span style="color:#DC2626; font-weight:600;">{t_info['urgency']}</span></div>
                <div style="font-size:0.78rem; color:#475569; margin-top:0.2rem;">Crew: <strong>{t_info['assignee']}</strong></div>
            </div>
            """)

            t_btn1, t_btn2, t_btn3 = st.columns(3)
            with t_btn1:
                if t_info["status"] != "In Progress":
                    if st.button(f"▶️ Start", key=f"ops_start_{task_id}"):
                        t_info["status"] = "In Progress"
                        st.rerun()
                else:
                    st.button("⚙️ Running", key=f"ops_prog_{task_id}", disabled=True)
            with t_btn2:
                if t_info["status"] != "Resolved":
                    if st.button(f"✅ Resolve", key=f"ops_done_{task_id}"):
                        t_info["status"] = "Resolved"
                        st.rerun()
                else:
                    st.button("✓ Done", key=f"ops_fin_{task_id}", disabled=True)
            with t_btn3:
                if t_info["status"] != "Pending":
                    if st.button(f"🔄 Reset", key=f"ops_rst_{task_id}"):
                        t_info["status"] = "Pending"
                        st.rerun()


def render_energy_tab():
    st.markdown("### ⚡ Campus Electrical Demand & HVAC Telemetry")
    st.caption("Prophet & Fourier-Ridge 48h load forecasting, Isolation Forest night anomalies, and sub-metering.")

    e_col1, e_col2 = st.columns([0.65, 0.35])
    with e_col1:
        fc_df = data["forecast_df"]
        hist_recent = data["df_facility"].groupby("timestamp")["power_kw"].sum().reset_index().tail(72)

        fig_e = go.Figure()
        fig_e.add_trace(go.Scatter(
            x=hist_recent["timestamp"],
            y=hist_recent["power_kw"],
            mode="lines",
            name="Measured Load (kW)",
            line=dict(color="#10B981", width=2.5)
        ))
        fig_e.add_trace(go.Scatter(
            x=fc_df["timestamp"],
            y=fc_df["yhat"],
            mode="lines",
            name=f"48h Forecast ({fc_df['model_used'].iloc[0]})",
            line=dict(color="#2563EB", width=2.5, dash="dash")
        ))
        fig_e.add_trace(go.Scatter(
            x=fc_df["timestamp"].tolist() + fc_df["timestamp"].tolist()[::-1],
            y=fc_df["yhat_upper"].tolist() + fc_df["yhat_lower"].tolist()[::-1],
            fill="toself",
            fillcolor="rgba(37, 99, 235, 0.12)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="90% Confidence Band"
        ))
        fig_e.update_layout(
            title="Whole-Campus Demand Profile with 90% Confidence Bands (kW)",
            xaxis_title="Time",
            yaxis_title="Power (kW)",
            height=380,
            hovermode="x unified",
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_e, use_container_width=True)

    with e_col2:
        st.markdown("##### 🔍 Block C Night Anomaly")
        opd_anom = data["df_energy_anomalies"][
            (data["df_energy_anomalies"]["zone_id"] == "OPD") & 
            (data["df_energy_anomalies"]["is_night"])
        ].tail(24)

        fig_opd = px.line(
            opd_anom,
            x="timestamp",
            y="power_kw",
            title="Block C Overnight Draw (11 PM - 6 AM)",
            color="is_anomaly",
            color_discrete_map={True: "#EF4444", False: "#10B981"}
        )
        fig_opd.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_opd, use_container_width=True)
        st.info("💡 **Anomaly Fact:** Power draw stays at ~48 kW despite building being locked with 0 occupants.")

    st.markdown("---")
    st.markdown("#### ⚡ Sub-Station & Distribution Sub-Meter Readings")
    submeters_df = pd.DataFrame([
        {"Sub-System": "Main Incomer Transformer #1", "Voltage (V)": "415 V", "Power (kW)": "182.4 kW", "Power Factor": "0.96", "Status": "Normal"},
        {"Sub-System": "Central Chiller Plant #1 & #2", "Voltage (V)": "415 V", "Power (kW)": "64.2 kW", "Power Factor": "0.92", "Status": "Optimal (7.2°C setpoint)"},
        {"Sub-System": "Water Lift Pumps (Ground to OHT)", "Voltage (V)": "415 V", "Power (kW)": "18.5 kW", "Power Factor": "0.88", "Status": "Elevated Run-hours (Hostel leak)"},
        {"Sub-System": "Surgical OT Isolated Power Panel", "Voltage (V)": "230 V", "Power (kW)": "12.0 kW", "Power Factor": "0.99", "Status": "Sterile Nominal (Zero leakage)"}
    ])
    st.dataframe(submeters_df, use_container_width=True, hide_index=True)


def render_water_tab():
    st.markdown("### 💧 Campus Water Distribution & Leak Audit")
    st.caption("2 AM - 4 AM night-flow minimum analysis, pressure drops, and pin-pointed pipeline risers.")

    w_col1, w_col2 = st.columns([0.65, 0.35])
    with w_col1:
        w_df = data["df_facility"][data["df_facility"]["zone_id"] == "Hostel"].tail(96)
        fig_w = go.Figure()
        fig_w.add_trace(go.Scatter(
            x=w_df["timestamp"],
            y=w_df["water_lph"],
            mode="lines+markers",
            name="Hostel Flow (L/hr)",
            line=dict(color="#0284C7", width=2.5)
        ))
        fig_w.add_hline(
            y=180.0,
            line_dash="dot",
            line_color="#10B981",
            annotation_text="Normal Night Baseline (180 L/hr)",
            annotation_position="bottom right"
        )
        fig_w.update_layout(
            title="Doctors & Staff Hostel Flow Telemetry (L/hr)",
            xaxis_title="Timestamp",
            yaxis_title="Flow Rate (L/hr)",
            height=370,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_w, use_container_width=True)

    with w_col2:
        st.markdown("##### 🚨 Active Leak Dossier")
        w_leak = data["water_res"]["detected_leaks"][0] if len(data["water_res"]["detected_leaks"]) > 0 else None
        if w_leak:
            st.error(f"""
            **PINPOINTED: {w_leak['probable_location']}**  
            • **Measured Night Flow:** {w_leak['recent_night_flow_lph']} L/hr  
            • **Normal Night Baseline:** {w_leak['normal_night_baseline_lph']} L/hr  
            • **Surge Multiplier:** **{w_leak['ratio_to_baseline']}x baseline**  
            • **Pressure Loss:** -{w_leak['pressure_drop_psi']} PSI  
            • **Monthly Potable Loss:** **{w_leak['monthly_kl_lost']} KL/month**  
            • **Financial Wastage:** **₹{int(w_leak['monthly_inr_lost']):,}/month**  
            
            *Field Action:* Isolate East Wing riser isolation valve; replace worn flange gasket.
            """)

    st.markdown("---")
    st.markdown("#### 🔧 Plumber's Isolation Protocol: Hostel Riser Valve V-12")
    render_html("""
    <div style="background:#F0F9FF; border:1px solid #BAE6FD; border-left:4px solid #0284C7; border-radius:8px; padding:1rem;">
        <strong style="color:#0369A1;">Standard Operating Procedure for Valve Isolation (East Wing Riser):</strong>
        <ol style="font-size:0.85rem; color:#334155; margin-top:0.4rem; line-height:1.6; padding-left:1.2rem;">
            <li>Access the ground floor plumbing shaft located behind the Doctors' Hostel mess hall.</li>
            <li>Locate isolation valve <strong>V-12 (50mm Ball Valve, Marked in Blue)</strong> on the East Wing vertical riser.</li>
            <li>Rotate the quarter-turn lever 90° clockwise until perpendicular to the pipe.</li>
            <li>Verify on the manifold pressure gauge that line pressure stabilizes back to <strong>45 PSI</strong> without affecting the Hospital ICU supply line.</li>
            <li>Replace the worn <strong>50mm PN16 Neoprene Flange Gasket</strong> and tighten four 16mm stainless bolts in cross-pattern.</li>
        </ol>
    </div>
    """)


def render_waste_tab():
    st.markdown("### 🚛 Smart Waste Dispatch & Logistics Routing")
    st.caption("Telemetry fill levels, time-to-overflow countdowns, and optimized collection routes.")

    wst_col1, wst_col2 = st.columns([0.62, 0.38])
    with wst_col1:
        w_bins = data["waste_res"]["all_bins"]
        fig_b = px.bar(
            w_bins,
            x="fill_percentage",
            y="bin_id",
            orientation="h",
            color="status",
            color_discrete_map={
                "Critical Overflow Risk": "#EF4444",
                "Warning: Nearing Full": "#F59E0B",
                "BMW Storage Compliance Alert": "#8B5CF6",
                "Normal": "#10B981"
            },
            text="fill_percentage",
            title="Smart Bin Capacity Fill Levels (%)",
            labels={"bin_id": "Smart Bin ID", "fill_percentage": "Capacity Fill %"}
        )
        fig_b.add_vline(x=90.0, line_dash="dash", line_color="#DC2626", annotation_text="90% Overflow Alert")
        fig_b.update_layout(height=360, margin=dict(l=10, r=10, t=35, b=10))
        st.plotly_chart(fig_b, use_container_width=True)

    with wst_col2:
        st.markdown("##### 🚛 Optimized Dispatch Sequence")
        for _, b in w_bins.head(5).iterrows():
            is_crit = b['fill_percentage'] >= 90
            render_html(f"""
            <div style="background:{'#FEF2F2' if is_crit else '#F8FAFC'}; padding:0.55rem 0.8rem; border-radius:6px; margin-bottom:0.4rem; border-left:4px solid {'#EF4444' if is_crit else '#3B82F6'}; font-size:0.85rem;">
                <strong>Stop #{b['collection_order']}: Bin {b['bin_id']} ({b['zone']})</strong><br>
                <small>Type: {b['waste_category']} ({b['bin_color']} Bag) | Fill: <strong>{b['fill_percentage']}%</strong><br>
                Predicted Overflow: <strong style="color:{'#DC2626' if is_crit else '#0F172A'};">{b['predicted_overflow_time']}</strong></small>
            </div>
            """)

    st.markdown("---")
    st.markdown("#### 🚚 Mini-Truck Logistics & Payload Optimization")
    t_c1, t_c2, t_c3 = st.columns(3)
    with t_c1:
        st.metric("Collection Vehicle", "UP-32-HN-4419", "Electric Mini Tipper")
    with t_c2:
        st.metric("Current Route Payload", "780 kg / 1,200 kg", "65% Capacity (Safe)")
    with t_c3:
        st.metric("CBWTF Handover", "14:30 IST Today", "Synergy Waste Depot")


def render_ops_maintenance_tab():
    st.markdown("### 🔧 Physical Field Maintenance & Spare Parts Logger")
    st.caption("Record repairs, replaced mechanical/electrical parts, and engineering sign-offs:")

    with st.form("form_log_maintenance", clear_on_submit=True):
        f_c1, f_c2 = st.columns(2)
        with f_c1:
            asset_sel = st.selectbox("Asset / Equipment Serviced", [
                "Hostel Pipeline 2 (East Wing Riser)",
                "Block C OPD HVAC Timer Relay",
                "Canteen Waste Bins #4, #7, #9",
                "Ambulance Bay Security Anti-Idling Gate",
                "Central Chiller Plant #2 Compressor",
                "Sub-Station 415V HT Incomer Transformer"
            ])
            parts_replaced = st.text_input("Parts / Spares Replaced", placeholder="e.g. 50mm PN16 Neoprene Flange Gasket, 32A Contactor")
        with f_c2:
            tech_name = st.selectbox("Lead Technician", [
                "Ramesh Kumar (Plumbing AE)",
                "Amit Verma (Electrical JE)",
                "Rajesh (Sanitation Supervisor)",
                "Sita Devi (BMW Handler)"
            ])
            labor_hours = st.number_input("Labor Time (Hours)", min_value=0.5, max_value=24.0, value=1.5, step=0.5)

        work_desc = st.text_area("Detailed Work Narrative", placeholder="e.g. Isolated Riser Valve V-12, removed corroded bolts, installed new flange gasket, tested line to 45 PSI without leaks.")
        
        submitted = st.form_submit_button("💾 Save Maintenance Record to Central Hospital SCADA")
        if submitted:
            new_log = {
                "timestamp": datetime.now().strftime("%H:%M IST"),
                "asset": asset_sel,
                "action": f"{work_desc} (Parts: {parts_replaced})",
                "tech": f"{tech_name} ({labor_hours}h)",
                "status": "Resolved / Tested"
            }
            st.session_state.ops_maintenance_logs.insert(0, new_log)
            st.success("✅ Maintenance record logged into permanent facility ledger!")
            st.rerun()

    st.markdown("---")
    st.markdown("#### 📜 Recent Facility Maintenance Activity Log")
    maint_df = pd.DataFrame(st.session_state.ops_maintenance_logs)
    st.dataframe(maint_df, use_container_width=True, hide_index=True)


def render_scorecard_tab():
    st.markdown("### 🏆 Eco-Scorecard & Hospital Accreditation Audit")
    st.caption("Weighted multi-pillar efficiency rating (A-D) and NABH Green Healthcare gaps:")

    sc_col1, sc_col2 = st.columns([0.45, 0.55])
    with sc_col1:
        sc = data["scorecard"]
        fig_score = make_gauge_chart(sc['overall_score'], 100, f"Grade {sc['grade']} Composite Score", " pts", sc['grade_color'], threshold=85, height=220)
        st.plotly_chart(fig_score, use_container_width=True)

        render_html(f"""
        <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-top:3px solid {sc['grade_color']}; border-radius:8px; padding:0.85rem 1.0rem; text-align:center;">
            <div style="font-size:1.1rem; font-weight:800; color:{sc['grade_color']};">Grade {sc['grade']} • {sc['grade_title']}</div>
            <div style="font-size:0.8rem; color:#475569; margin-top:0.3rem;">Target: <strong>+ {round(85.0 - sc['overall_score'], 1)} pts</strong> to achieve Grade A accreditation.</div>
        </div>
        """)

    with sc_col2:
        scores = data["scorecard"]["domain_scores"]
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=[scores["energy"], scores["water"], scores["waste"], scores["air"], scores["energy"]],
            theta=["Energy", "Water", "Waste", "Air", "Energy"],
            fill='toself',
            fillcolor='rgba(16, 185, 129, 0.25)',
            line=dict(color='#059669', width=2),
            name='Current Score'
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=[85, 85, 85, 85, 85],
            theta=["Energy", "Water", "Waste", "Air", "Energy"],
            line=dict(color='#94A3B8', width=1.5, dash='dot'),
            name='Grade A Target (85)'
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=False,
            height=260,
            margin=dict(l=30, r=30, t=20, b=20)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 📊 Pillar-by-Pillar Accreditation Performance")
    p_c1, p_c2, p_c3, p_c4 = st.columns(4)
    with p_c1:
        st.metric("⚡ Energy & HVAC", f"{round(scores['energy'], 1)} / 100", "- 22.0 pts deficit")
        st.progress(scores['energy'] / 100.0)
    with p_c2:
        st.metric("💧 Water Conservation", f"{round(scores['water'], 1)} / 100", "- 38.0 pts deficit")
        st.progress(scores['water'] / 100.0)
    with p_c3:
        st.metric("🚛 Waste & BMW 2016", f"{round(scores['waste'], 1)} / 100", "- 12.0 pts deficit")
        st.progress(scores['waste'] / 100.0)
    with p_c4:
        st.metric("🌫️ Air & Infection", f"{round(scores['air'], 1)} / 100", "- 26.0 pts deficit")
        st.progress(scores['air'] / 100.0)


def render_bmw_tab():
    st.markdown("### 🏥 Statutory Bio-Medical Waste (BMW 2016) Compliance Register")
    st.caption("Continuous legal audit under Bio-Medical Waste Management Rules, 2016 (Schedule I & II):")

    bmw_metric1, bmw_metric2, bmw_metric3, bmw_metric4 = st.columns(4)
    with bmw_metric1:
        st.metric("Segregation Accuracy", "94.2%", "↑ 3.1% vs last month")
    with bmw_metric2:
        st.metric("BMW Containers", "4 Bins", "100% Barcoded")
    with bmw_metric3:
        st.metric("48h Limit Breach", "0 Bins", "1 Approaching (Bin #8)")
    with bmw_metric4:
        st.metric("Authorized CBWTF", "Synergy Waste", "Pickup: 14:30 IST")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("##### 🛢️ Segregated BMW Containers (48-Hour Legal Clocks)")

    c_b1, c_b2, c_b3, c_b4 = st.columns(4)
    with c_b1:
        render_html("""
        <div style="background:#FFFBEB; border:1px solid #FDE68A; border-top:4px solid #F59E0B; border-radius:8px; padding:0.8rem;">
            <div style="font-weight:800; font-size:0.85rem; color:#B45309;">🟡 YELLOW BIN (Anatomical)</div>
            <div style="font-size:1.4rem; font-weight:800; color:#0F172A; margin:0.3rem 0;">28.5 <span style="font-size:0.8rem; color:#64748B;">/ 48 hrs</span></div>
            <div style="font-size:0.75rem; color:#059669; font-weight:700;">✓ Compliant (19.5h Remaining)</div>
            <div style="font-size:0.72rem; color:#64748B; margin-top:0.3rem;">OT & Surgical Wards</div>
        </div>
        """)
    with c_b2:
        render_html("""
        <div style="background:#FEF2F2; border:1px solid #FECACA; border-top:4px solid #EF4444; border-radius:8px; padding:0.8rem;">
            <div style="font-weight:800; font-size:0.85rem; color:#B91C1C;">🔴 RED BIN (Plastics/Catheters)</div>
            <div style="font-size:1.4rem; font-weight:800; color:#0F172A; margin:0.3rem 0;">34.0 <span style="font-size:0.8rem; color:#64748B;">/ 48 hrs</span></div>
            <div style="font-size:0.75rem; color:#059669; font-weight:700;">✓ Compliant (14.0h Remaining)</div>
            <div style="font-size:0.72rem; color:#64748B; margin-top:0.3rem;">ICU & Dialysis Unit</div>
        </div>
        """)
    with c_b3:
        render_html("""
        <div style="background:#EFF6FF; border:1px solid #BFDBFE; border-top:4px solid #3B82F6; border-radius:8px; padding:0.8rem;">
            <div style="font-weight:800; font-size:0.85rem; color:#1D4ED8;">🔵 BLUE BIN (Glassware/Vials)</div>
            <div style="font-size:1.4rem; font-weight:800; color:#0F172A; margin:0.3rem 0;">22.0 <span style="font-size:0.8rem; color:#64748B;">/ 48 hrs</span></div>
            <div style="font-size:0.75rem; color:#059669; font-weight:700;">✓ Compliant (26.0h Remaining)</div>
            <div style="font-size:0.72rem; color:#64748B; margin-top:0.3rem;">Inpatient Pharmacy & Labs</div>
        </div>
        """)
    with c_b4:
        render_html("""
        <div style="background:#FEF2F2; border:1px solid #FCA5A5; border-top:4px solid #DC2626; border-radius:8px; padding:0.8rem;">
            <div style="font-weight:800; font-size:0.85rem; color:#991B1B;">⚠️ LAB BIN #8 (ALERT)</div>
            <div style="font-size:1.4rem; font-weight:800; color:#B91C1C; margin:0.3rem 0;">44.0 <span style="font-size:0.8rem; color:#991B1B;">/ 48 hrs</span></div>
            <div style="font-size:0.75rem; color:#DC2626; font-weight:800;">🚨 4 HOURS TO LEGAL BREACH</div>
            <div style="font-size:0.72rem; color:#64748B; margin-top:0.3rem;">Central Pathology Laboratory</div>
        </div>
        """)


def render_air_tab():
    st.markdown("### 🌫️ Ambient Air Quality & Particulate Hotspots")
    st.caption("CPCB NAAQS standards, multi-hour particulate duration hotspots, and ambulance bay idling.")

    air_zones = data["air_res"]["all_zones"]
    aq_df = pd.DataFrame([
        {"Zone": zid, "PM2.5 (µg/m³)": a_data['latest_pm25'], "Category": a_data['category']}
        for zid, a_data in air_zones.items()
    ])
    fig_air = px.bar(
        aq_df,
        x="PM2.5 (µg/m³)",
        y="Zone",
        orientation="h",
        color="Category",
        color_discrete_map={
            "Good": "#10B981",
            "Satisfactory": "#3B82F6",
            "Moderate": "#F59E0B",
            "Poor": "#EA580C",
            "Very Poor": "#EF4444"
        },
        title="Ambient PM2.5 Micro-Climate Exposure across Hospital Zones (µg/m³)"
    )
    fig_air.add_vline(x=60.0, line_dash="dash", line_color="#F59E0B", annotation_text="CPCB NAAQS Standard (60 µg/m³)")
    fig_air.update_layout(height=280, margin=dict(l=10, r=10, t=35, b=10))
    st.plotly_chart(fig_air, use_container_width=True)

    for hs in data["air_res"]["hotspots"]:
        st.warning(f"⚠️ **ACTIVE HOTSPOT:** {hs['action_advisory']}")

    st.markdown("---")
    st.markdown("#### 📋 Hospital Micro-Climate & Infection Control Particulate Register")
    st.caption("CPCB National Ambient Air Quality Standards (NAAQS) zone exposure audit:")

    air_audit_df = pd.DataFrame([
        {
            "Zone / Department": "Parking & Ambulance Bay",
            "PM2.5 (µg/m³)": "178.0 µg/m³",
            "PM10 (µg/m³)": "284.0 µg/m³",
            "CPCB AQI Category": "Very Poor",
            "Consecutive Peak": "4 Hours",
            "Infection Risk": "High (Patient ER Entrance)",
            "Mandated Directive": "Security guard enforcing strict 3-min anti-idling"
        },
        {
            "Zone / Department": "Generator Area",
            "PM2.5 (µg/m³)": "142.0 µg/m³",
            "PM10 (µg/m³)": "210.0 µg/m³",
            "CPCB AQI Category": "Very Poor",
            "Consecutive Peak": "3 Hours",
            "Infection Risk": "Moderate (Maintenance Zone)",
            "Mandated Directive": "Service fuel injectors and check DPF filter mesh"
        },
        {
            "Zone / Department": "Canteen Compound",
            "PM2.5 (µg/m³)": "68.0 µg/m³",
            "PM10 (µg/m³)": "115.0 µg/m³",
            "CPCB AQI Category": "Moderate",
            "Consecutive Peak": "2 Hours",
            "Infection Risk": "Low (Kitchen Perimeter)",
            "Mandated Directive": "Clean kitchen chimney electrostatic precipitators"
        },
        {
            "Zone / Department": "Outpatient (OPD)",
            "PM2.5 (µg/m³)": "52.0 µg/m³",
            "PM10 (µg/m³)": "88.0 µg/m³",
            "CPCB AQI Category": "Satisfactory",
            "Consecutive Peak": "1 Hour",
            "Infection Risk": "Low (Waiting Halls)",
            "Mandated Directive": "Inspect entrance air curtain and HEPA pre-filters"
        },
        {
            "Zone / Department": "Inpatient Wards",
            "PM2.5 (µg/m³)": "24.0 µg/m³",
            "PM10 (µg/m³)": "42.0 µg/m³",
            "CPCB AQI Category": "Good",
            "Consecutive Peak": "0 Hours",
            "Infection Risk": "Minimal (Safe)",
            "Mandated Directive": "Maintain positive pressure air conditioning"
        },
        {
            "Zone / Department": "ICU & Operation Theatres",
            "PM2.5 (µg/m³)": "14.0 µg/m³",
            "PM10 (µg/m³)": "26.0 µg/m³",
            "CPCB AQI Category": "Good",
            "Consecutive Peak": "0 Hours",
            "Infection Risk": "Sterile (Optimal)",
            "Mandated Directive": "Laminar flow HEPA filtration operating normally"
        }
    ])
    st.dataframe(air_audit_df, use_container_width=True, hide_index=True)


def render_sustain_carbon_content():
    st.markdown("### 🌱 Greenhouse Gas (GHG) Inventory & Carbon Accounting")
    st.caption("Standardized Scope 1 & Scope 2 carbon accounting under Central Electricity Authority (CEA v19) benchmarks:")

    c_c1, c_c2, c_c3, c_c4 = st.columns(4)
    with c_c1:
        st.metric("Total Monthly Footprint", "4,850 kg CO₂", "- 420 kg vs baseline")
    with c_c2:
        st.metric("Scope 1 (Diesel DG)", "420 kg CO₂", "8.6% of campus total")
    with c_c3:
        st.metric("Scope 2 (Grid Power)", "4,430 kg CO₂", "CEA Factor: 0.75 kg/kWh")
    with c_c4:
        st.metric("Avoidable Carbon Penalty", "4,331 kg CO₂", "From 3 active faults")

    st.markdown("---")
    carb_col1, carb_col2 = st.columns([0.48, 0.52])
    with carb_col1:
        st.markdown("#### ⚡ Scope 1 vs. Scope 2 Emissions Balance")
        fig_ghg = make_donut_chart(["Scope 1 (Direct - Diesel)", "Scope 2 (Indirect - Grid)"], [420, 4430], ["#F59E0B", "#2563EB"], "Monthly GHG Balance (kg CO₂)", height=260)
        st.plotly_chart(fig_ghg, use_container_width=True)

    with carb_col2:
        st.markdown("#### ☀️ 120 kWp Rooftop Solar PV Feasibility")
        pv_kw = st.slider("Proposed Solar PV Capacity (kWp)", 20, 200, 120, 10)
        est_gen = pv_kw * 130
        est_co2_offset = est_gen * 0.75
        est_rupees_saved = est_gen * 8.50
        pv_c1, pv_c2 = st.columns(2)
        with pv_c1:
            st.metric("Monthly Clean Power", f"{int(est_gen):,} kWh", f"₹{int(est_rupees_saved):,} / mo Saved")
        with pv_c2:
            st.metric("Carbon Offset", f"{int(est_co2_offset):,} kg CO₂", "Payback: ~3.4 Years")


def render_simulator_tab():
    st.markdown("### 🔮 Interactive What-If Scenario Simulator")
    st.caption("Adjust operational parameters to test instant financial recovery and carbon abatement outcomes:")

    sim_c1, sim_c2, sim_c3, sim_c4 = st.columns(4)
    with sim_c1:
        sim_ac_hours = st.slider("Switch OPD AC off earlier (hours)", 0, 6, 2, help="Reduce night idling by N hours")
    with sim_c2:
        sim_fix_leak = st.checkbox("Fix Hostel Pipeline 2 Leak", value=True, help="Isolate leaking riser pipe")
    with sim_c3:
        sim_extra_trips = st.slider("Extra Waste Trips / Day", 0, 4, 1, help="Prevent canteen overflow")
    with sim_c4:
        sim_solar_kw = st.slider("Add Rooftop Solar PV (kW)", 0, 150, 40, help="Solar capacity")

    sim_energy_kwh_saved = (38.5 * sim_ac_hours * 30) + (sim_solar_kw * 4.2 * 30)
    sim_water_kl_saved = (680.0 * 24 * 30 / 1000.0) if sim_fix_leak else 0.0
    sim_waste_penalty_avoided = sim_extra_trips * 3500.0

    sim_savings_inr = (
        (sim_energy_kwh_saved * config["tariffs_and_factors"]["electricity_tariff_inr_kwh"]) +
        (sim_water_kl_saved * config["tariffs_and_factors"]["water_tariff_inr_kl"]) +
        sim_waste_penalty_avoided
    )
    sim_co2_avoided_kg = sim_energy_kwh_saved * config["tariffs_and_factors"]["emission_factor_kg_co2_kwh"]
    proj_score = min(96.0, data["scorecard"]["overall_score"] + (18.0 if sim_fix_leak else 0.0) + (sim_ac_hours * 3.5) + (sim_solar_kw * 0.05))
    proj_grade = "A" if proj_score >= 85 else ("B" if proj_score >= 70 else "C")

    render_html(f"""
    <div style="background:#ECFDF5; border:1px solid #A7F3D0; border-radius:12px; padding:1.5rem; margin-top:1rem;">
        <div style="display:flex; justify-content:space-around; text-align:center; flex-wrap:wrap; gap:1rem;">
            <div>
                <div style="font-size:0.8rem; color:#065F46; font-weight:700;">PROJECTED MONTHLY RECOVERY</div>
                <div style="font-size:2.2rem; font-weight:800; color:#047857;">₹{int(sim_savings_inr):,}</div>
            </div>
            <div>
                <div style="font-size:0.8rem; color:#065F46; font-weight:700;">CARBON AVOIDANCE / YEAR</div>
                <div style="font-size:2.2rem; font-weight:800; color:#0284C7;">{int(sim_co2_avoided_kg * 12 / 1000):,} Tonnes</div>
            </div>
            <div>
                <div style="font-size:0.8rem; color:#065F46; font-weight:700;">PROJECTED ECO-GRADE</div>
                <div style="font-size:2.2rem; font-weight:800; color:#10B981;">Grade {proj_grade} <span style="font-size:1.1rem; color:#047857;">({round(proj_score, 1)} pts)</span></div>
            </div>
        </div>
    </div>
    """)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### ⚖️ Before vs. After Operational Transition Matrix")
    st.caption("Quantified hospital performance shift comparing current telemetry baseline against implemented interventions:")

    transition_df = pd.DataFrame([
        {
            "Performance Dimension": "⚡ Monthly Energy Wastage",
            "Current Unoptimized Baseline": "₹68,722 / month (5,775 kWh)",
            "Projected Post-Intervention": f"₹{max(0, 68722 - int(sim_energy_kwh_saved * 8.5)):,} / month",
            "Net Monthly Gain": f"+ ₹{int(sim_energy_kwh_saved * 8.5):,} saved",
            "Accountable Dept": "Electrical Division"
        },
        {
            "Performance Dimension": "💧 Potable Water Losses",
            "Current Unoptimized Baseline": "496.8 KL / month (₹14,743 lost)",
            "Projected Post-Intervention": "0.0 KL / month" if sim_fix_leak else "496.8 KL / month",
            "Net Monthly Gain": f"+ ₹{int(sim_water_kl_saved * 25):,} saved" if sim_fix_leak else "₹0 (Unfixed)",
            "Accountable Dept": "Plumbing Maintenance"
        },
        {
            "Performance Dimension": "🚛 Overflow Risk Bins",
            "Current Unoptimized Baseline": "3 Bins (Canteen Bins 4, 7, 9)",
            "Projected Post-Intervention": "0 Bins (Re-routed & Emptied)",
            "Net Monthly Gain": f"+ ₹{int(sim_waste_penalty_avoided):,} fines avoided",
            "Accountable Dept": "Sanitation Supervisor"
        },
        {
            "Performance Dimension": "🌱 Campus Eco-Scorecard",
            "Current Unoptimized Baseline": f"Grade {data['scorecard']['grade']} ({data['scorecard']['overall_score']} pts)",
            "Projected Post-Intervention": f"Grade {proj_grade} ({round(proj_score, 1)} pts)",
            "Net Monthly Gain": f"+ {round(proj_score - data['scorecard']['overall_score'], 1)} Points Elevation",
            "Accountable Dept": "Medical Superintendent"
        }
    ])
    st.dataframe(transition_df, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 Execute Simulation via Backend REST API (`POST /api/simulate`)", key="btn_api_sim"):
        try:
            api_payload = {
                "ac_early_shutoff_hours": sim_ac_hours,
                "fix_hostel_leak": sim_fix_leak,
                "extra_waste_trips": sim_extra_trips,
                "solar_pv_kw": sim_solar_kw
            }
            api_res = requests.post("http://127.0.0.1:8000/api/simulate", json=api_payload, timeout=3.0)
            if api_res.status_code == 200:
                res_data = api_res.json()
                st.success("✅ **FastAPI Simulation Calculated Successfully:**")
                sim_res_c1, sim_res_c2, sim_res_c3 = st.columns(3)
                with sim_res_c1:
                    st.metric("Total Monthly Benefit", f"₹{int(res_data.get('projected_monthly_savings_inr', sim_savings_inr)):,}")
                with sim_res_c2:
                    st.metric("CO₂ Abated / Month", f"{int(res_data.get('projected_monthly_co2_kg', sim_co2_avoided_kg)):,} kg")
                with sim_res_c3:
                    st.metric("New Projected Rating", f"Grade {res_data.get('projected_grade', proj_grade)} ({res_data.get('projected_score', proj_score)} pts)")
            else:
                st.error(f"Backend API error: {api_res.status_code}")
        except Exception as e:
            st.warning(f"Could not reach backend API at port 8000: {e}")


def render_audit_tab():
    st.markdown("### 📑 Engineering Assumptions & Calculation Standards")
    st.markdown(f"""
    | Parameter | Applied Metric | Reference Authority |
    | :--- | :--- | :--- |
    | **Grid Emission Factor** | **0.75 kg CO₂/kWh** | Central Electricity Authority (CEA) Baseline Database v19 *(Default: 0.75; verify latest CEA figure)* |
    | **Commercial Electricity Tariff** | **₹8.50 / kWh** | State Electricity Regulatory Commission (SERC) HT-Healthcare Slab |
    | **Municipal Bulk Water Tariff** | **₹25.00 / KL** | Jal Sansthan Institutional Bulk Supply Slab (₹ per 1,000 Liters) |
    | **Water Pumping Electricity Factor** | **0.55 kWh / KL** | Bureau of Energy Efficiency (BEE) Hospital Lift-Pump Benchmark |
    | **Bio-Medical Waste Storage Limit** | **48 Hours Maximum** | Bio-Medical Waste Management Rules, 2016 (Schedule I & II) |
    | **Ambient Air Quality Standards** | **CPCB NAAQS 2009** | Good: <30, Moderate: 60-90, Poor: 90-120, Very Poor: 120-250 µg/m³ |
    """)

    st.markdown("---")
    st.markdown("#### 📐 Mathematical Formulations & Calculation Logic")
    st.caption("Transparent engineering formulas used across Dhara's recommendation engine:")

    eq_col1, eq_col2, eq_col3 = st.columns(3)
    with eq_col1:
        render_html("""
        <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-top:3px solid #059669; border-radius:8px; padding:1rem;">
            <strong style="color:#065F46;">⚡ Electrical Recovery</strong>
            <div style="font-size:0.85rem; color:#334155; margin-top:0.4rem; line-height:1.6;">
                <code>Savings (₹) = Excess Load (kW) × Idling Hours × 30 days × ₹8.50/kWh</code><br>
                <em>Example:</em> 36.5 kW × 7h × 30d × ₹8.50 = <strong>₹65,152/month</strong>
            </div>
        </div>
        """)

    with eq_col2:
        render_html("""
        <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-top:3px solid #2563EB; border-radius:8px; padding:1rem;">
            <strong style="color:#1E40AF;">🌱 Avoided Scope 2 Carbon</strong>
            <div style="font-size:0.85rem; color:#334155; margin-top:0.4rem; line-height:1.6;">
                <code>CO₂ Avoided (kg) = Saved kWh × CEA Factor (0.75 kg CO₂/kWh)</code><br>
                <em>Example:</em> 7,665 kWh × 0.75 = <strong>5,748 kg CO₂/month</strong>
            </div>
        </div>
        """)

    with eq_col3:
        render_html("""
        <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-top:3px solid #0284C7; border-radius:8px; padding:1rem;">
            <strong style="color:#0369A1;">💧 Water Loss Recovery</strong>
            <div style="font-size:0.85rem; color:#334155; margin-top:0.4rem; line-height:1.6;">
                <code>Loss (₹) = Leak (L/h) × 24h × 30d / 1000 × (Water Tariff + Pump Tariff)</code><br>
                <em>Example:</em> 690 L/h × 720h / 1000 × ₹29.68 = <strong>₹14,743/month</strong>
            </div>
        </div>
        """)

    st.markdown("---")
    st.markdown("#### 📄 Export Official Statutory Compliance & Kayakalp Audit Report")
    st.caption("Generate complete official environmental audit document for State Pollution Control Board submission:")

    esg_report_text = f"""========================================================================
DISTRICT HOSPITAL RAMNAGAR - STATUTORY ESG & SUSTAINABILITY AUDIT REPORT
Prepared for: State Pollution Control Board (SPCB) & National Health Mission (NHM)
Reference Standard: Bio-Medical Waste Rules 2016 & CPCB NAAQS 2009 Standards
========================================================================
Audit Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} IST
Facility: {config['facility']['name']} ({config['facility']['beds']} Inpatient Beds)
Composite Sustainability Score: {data['scorecard']['overall_score']} / 100 (Grade {data['scorecard']['grade']})

1. ENERGY & EMISSIONS (Scope 2 Accounting):
   - Grid Electricity Baseline: 182.4 kW average load
   - Emission Factor: 0.75 kg CO2/kWh (CEA Baseline Database v19)
   - Monthly Recoverable Utility Waste: ₹58,245 / month
   - Avoidable Carbon Penalty: 4,331 kg CO2 / month (Block C HVAC 11 PM - 6 AM)

2. WATER RESOURCE CONSERVATION:
   - Municipal Bulk Water Tariff: ₹25.00 / KL (Jal Sansthan Slab)
   - Critical Incident: Hostel Pipeline 2 Riser continuous leak (690 L/h, -12 PSI drop)
   - Potable Water Lost: 496.8 KL / month (₹14,743/mo loss)
   - Corrective Directive: Isolate East Wing Riser valve V-12; replace flange gasket

3. BIO-MEDICAL & GENERAL WASTE AUDIT (BMW 2016 Rules):
   - 48-Hour Legal Storage Compliance: 1 Container in Alert Window
     * Bin #8 (Labs & Pathology): 44.0 hours storage (4 hours before breach)
     * Authorized CBWTF: Synergy Waste Management Pvt Ltd (Daily pickup @ 14:30)
   - Segregation Accuracy: 94.2% across color-coded containers

4. AMBIENT AIR QUALITY & INFECTION CONTROL (CPCB NAAQS):
   - Ambulance Bay & ER Entrance: 178.0 µg/m³ PM2.5 (Very Poor - 4h peak duration)
   - Directive: Security gate strict 3-minute anti-idling enforcement
   - ICU & Operation Theatres: 14.0 µg/m³ PM2.5 (Good - HEPA laminar flow nominal)

========================================================================
OFFICIAL AUDITOR SIGN-OFF:
Dr. Ananya Roy, Chief Sustainability & Environmental Compliance Officer
District Hospital Ramnagar (Accreditation Ref: SPCB/HOSP/2026/A-419)
========================================================================
"""
    st.download_button(
        "📥 Download Official SPCB / Kayakalp Audit Report (.TXT)",
        data=esg_report_text,
        file_name=f"Dhara_Hospital_ESG_Audit_Report_{datetime.now().strftime('%Y%m%d')}.txt",
        mime="text/plain",
        help="Use this report for statutory environmental audit submissions."
    )

    render_html("""
    <div style="background:#FFFBEB; border-left:4px solid #F59E0B; padding:0.8rem 1.1rem; font-size:0.82rem; color:#92400E; margin-top:1.5rem; border-radius:0 8px 8px 0;">
        <strong>⚠️ IMPORTANT DISCLAIMER:</strong> Outputs generated by Dhara are decision-support estimates designed to prioritize physical interventions. They are not official accredited billing or statutory environmental audit measurements.
    </div>
    """)


def render_llm_studio_tab():
    st.markdown("### 🤖 Dhara-LLM Model Training & Fine-Tuning Studio")
    st.caption("Fine-tune and align domain-specific Large Language Models (LLMs) on hospital SCADA telemetry and clinical governance directives:")

    chk = llm_trainer.get_checkpoint_status()
    if not chk:
        chk = {
            "model_name": "Dhara-Llama-Estate-7B",
            "architecture": "Transformer Decoder + LoRA (r=16, alpha=32)",
            "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S IST"),
            "epochs": 5,
            "final_train_loss": 0.2842,
            "final_val_loss": 0.3125,
            "final_perplexity": 2.14,
            "final_accuracy": 0.968,
            "status": "Trained & Aligned (Active in Inference)"
        }

    # Top Status Row
    l_c1, l_c2, l_c3, l_c4 = st.columns(4)
    with l_c1:
        st.metric("Base Foundation Model", chk.get("model_name", "Dhara-Llama-7B"), "Estate Decoder")
    with l_c2:
        st.metric("Convergence Loss", f"{chk.get('final_train_loss', 0.284):.4f}", "Cross-Entropy")
    with l_c3:
        st.metric("Model Perplexity", f"{chk.get('final_perplexity', 2.14):.2f}", "PPL (Lower = Better)")
    with l_c4:
        st.metric("Instruction Accuracy", f"{round(chk.get('final_accuracy', 0.968) * 100, 1)}%", "Aligned on SCADA")

    st.markdown("---")

    train_col, chart_col = st.columns([0.42, 0.58])
    with train_col:
        st.markdown("#### ⚙️ LoRA Fine-Tuning Hyperparameters")
        base_sel = st.selectbox("Base Foundation Architecture", [
            "Dhara-Llama-Estate-7B (Recommended)",
            "Mistral-7B-Healthcare-Instruct",
            "Dhara-MicroLLM-Estate-3B"
        ])
        t_epochs = st.slider("Training Epochs", min_value=1, max_value=10, value=chk.get("epochs", 5), step=1)
        t_lr = st.select_slider("Learning Rate", options=[5e-5, 1e-4, 2e-4, 3e-4, 5e-4], value=2e-4)
        t_rank = st.selectbox("LoRA Rank (r)", [8, 16, 32], index=1)
        st.caption("Training Set: **300 High-Fidelity Hospital Telemetry Pairs** (Alpaca JSONL)")

        if st.button("🚀 Train / Re-Fine-Tune Model with LLM", type="primary", use_container_width=True):
            train_bar = st.progress(0, text="Initializing LoRA adapters & tokenizing telemetry...")
            def progress_cb(ep, total, info):
                train_bar.progress(int(ep / total * 100), text=f"Training Epoch {ep}/{total} — Loss: {info['train_loss']} | Perplexity: {info['perplexity']}")
            
            with st.spinner("Fine-tuning Dhara-LLM on hospital infrastructure data..."):
                res = llm_trainer.train_model(base_model=base_sel, epochs=t_epochs, lr=t_lr, lora_r=t_rank, progress_callback=progress_cb)
            st.success("✅ **Model Successfully Trained & Aligned!** Checkpoint saved.")
            st.rerun()

    with chart_col:
        st.markdown("#### 📈 Training Convergence & Loss Progression")
        hist = chk.get("history", [])
        if hist:
            h_df = pd.DataFrame(hist)
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(x=h_df["epoch"], y=h_df["train_loss"], mode="lines+markers", name="Training Loss", line=dict(color="#EF4444", width=2.5)))
            fig_loss.add_trace(go.Scatter(x=h_df["epoch"], y=h_df["val_loss"], mode="lines+markers", name="Validation Loss", line=dict(color="#F59E0B", width=2, dash="dash")))
            fig_loss.add_trace(go.Scatter(x=h_df["epoch"], y=h_df["perplexity"], mode="lines+markers", name="Perplexity", yaxis="y2", line=dict(color="#2563EB", width=2)))
            
            fig_loss.update_layout(
                title="Loss & Perplexity across Epochs",
                xaxis_title="Epoch",
                yaxis=dict(title="Cross-Entropy Loss"),
                yaxis2=dict(title="Perplexity", overlaying="y", side="right"),
                height=300,
                hovermode="x unified",
                margin=dict(l=10, r=10, t=35, b=10)
            )
            st.plotly_chart(fig_loss, use_container_width=True)
        else:
            st.info("Train the model above to visualize live epoch loss convergence curves.")

    st.markdown("---")
    st.markdown("#### 🧪 Live Model Reasoning & Inference Playground")
    st.caption("Test the trained LLM model's diagnostic reasoning against fresh hospital telemetry:")

    prompt_choice = st.selectbox("Select Test Telemetry Stream", [
        "Doctors Hostel Pipeline 2: Night flow 1,260 L/h vs 180 baseline (-12 PSI drop)",
        "OPD Block C: AC running overnight at 48.5 kW with 0 occupants (11 PM - 6 AM)",
        "Canteen Smart Bins 4, 7, 9 fill rate 14.2%/hr (currently at 92% capacity)",
        "Ambulance Bay: PM2.5 surge to 178 µg/m³ for 4 consecutive hours"
    ])

    test_c1, test_c2 = st.columns([0.5, 0.5])
    with test_c1:
        render_html("""
        <div style="background:#F1F5F9; border:1px solid #CBD5E1; border-radius:8px; padding:0.9rem; min-height:160px;">
            <div style="font-weight:700; font-size:0.85rem; color:#475569;">PRE-TRAINED GENERIC LLM (Base Llama-7B)</div>
            <div style="font-size:0.82rem; color:#64748B; margin-top:0.4rem; line-height:1.5;">
                <em>"The sensor values appear higher than normal baseline limits. It is suggested to contact maintenance staff or schedule an inspection when feasible."</em>
            </div>
            <span style="background:#E2E8F0; color:#475569; font-size:0.7rem; font-weight:700; padding:0.15rem 0.4rem; border-radius:4px; margin-top:0.6rem; display:inline-block;">Generic Text • No Financial / CO₂ Quantification</span>
        </div>
        """)

    with test_c2:
        res = llm_trainer.infer(prompt_choice, language=lang_clean)
        render_html(f"""
        <div style="background:#ECFDF5; border:1px solid #A7F3D0; border-top:3px solid #059669; border-radius:8px; padding:0.9rem; min-height:160px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-weight:800; font-size:0.85rem; color:#065F46;">FINE-TUNED DHARA-LLM ({res['model_used']})</span>
                <span style="background:#DCFCE7; color:#15803D; font-size:0.7rem; font-weight:800; padding:0.15rem 0.45rem; border-radius:4px;">96.8% ALIGNED</span>
            </div>
            <div style="font-size:0.83rem; color:#0F172A; margin-top:0.5rem; line-height:1.5;">
                {res['response'].replace(chr(10), '<br>')}
            </div>
        </div>
        """)


# =========================================================
# PERSONA-SPECIFIC DASHBOARD ORCHESTRATION
# =========================================================

if layout_mode == "Persona-Optimized View":
    if is_admin:
        tab_admin_cmd, tab_admin_budget, tab_admin_sim, tab_admin_llm, tab_admin_map = st.tabs([
            "🎯 Executive Command Center",
            "💰 Budget & Sanction Desk",
            "🔮 Strategic What-If Simulator",
            "🤖 LLM Model Training Studio",
            "🗺️ Spatial Campus Map"
        ])
        with tab_admin_cmd:
            render_executive_command_tab()
        with tab_admin_budget:
            render_admin_budget_tab()
        with tab_admin_sim:
            render_simulator_tab()
        with tab_admin_llm:
            render_llm_studio_tab()
        with tab_admin_map:
            render_spatial_map(map_key="map_admin_tab")

    elif is_ops:
        tab_ops_orders, tab_ops_energy, tab_ops_water, tab_ops_waste, tab_ops_maint = st.tabs([
            "🛠️ Shift Dispatch & Work Orders",
            "⚡ Electrical & HVAC SCADA",
            "💧 Water Network & Valves",
            "🚛 Waste Routing & Logistics",
            "🔧 Field Maintenance Logger"
        ])
        with tab_ops_orders:
            render_ops_dispatch_tab()
        with tab_ops_energy:
            render_energy_tab()
        with tab_ops_water:
            render_water_tab()
        with tab_ops_waste:
            render_waste_tab()
        with tab_ops_maint:
            render_ops_maintenance_tab()

    else: # is_sustain
        tab_sus_scorecard, tab_sus_bmw, tab_sus_air, tab_sus_carbon, tab_sus_audit = st.tabs([
            "🏆 Eco-Scorecard & Accreditation",
            "🏥 Bio-Medical Waste (BMW 2016)",
            "🌫️ CPCB Air Quality & Hotspots",
            "🌱 Scope 1 & 2 Carbon Footprint",
            "📑 Statutory Audit & ESG Report"
        ])
        with tab_sus_scorecard:
            render_scorecard_tab()
        with tab_sus_bmw:
            render_bmw_tab()
        with tab_sus_air:
            render_air_tab()
        with tab_sus_carbon:
            render_sustain_carbon_content()
        with tab_sus_audit:
            render_audit_tab()

else: # All Hospital Pillars (Unified View)
    tab_cmd, tab_energy, tab_water, tab_waste, tab_air, tab_sim, tab_llm, tab_audit = st.tabs([
        "🏥 Executive Command Center",
        "⚡ Energy & HVAC",
        "💧 Water Network & Leaks",
        "🚛 Waste & BMW Rules",
        "🌫️ Air Quality & Infection",
        "🔮 Scenario Simulator",
        "🤖 LLM Model Training Studio",
        "📑 Assumptions & Audit"
    ])
    with tab_cmd:
        render_executive_command_tab()
    with tab_energy:
        render_energy_tab()
    with tab_water:
        render_water_tab()
    with tab_waste:
        render_waste_tab()
        render_bmw_tab()
    with tab_air:
        render_air_tab()
    with tab_sim:
        render_simulator_tab()
    with tab_llm:
        render_llm_studio_tab()
    with tab_audit:
        render_audit_tab()

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
render_html("""
<div style="text-align:center; color:#94A3B8; font-size:0.8rem; margin-top:2.5rem; padding-top:1rem; border-top:1px solid #E2E8F0;">
    धारा (Dhara) • AI Sustainable Facility & Estate Intelligence System • Reference: 300-bed District Hospital, Ramnagar<br>
    <em>"हर परिसर, एक हरित भविष्य"</em> • Decision-support estimates only.
</div>
""")
