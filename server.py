"""
Dhara - FastAPI Backend REST API Server
Powers the AI Sustainable Facility & Estate Intelligence System
Connects hospital telemetry, predictive models, and external APIs (Weather, AQI, CEA Grid).
"""

import os
from datetime import datetime
from fastapi import FastAPI, Query, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from data.pipeline import DharaDataPipeline
from analytics.energy import EnergyAnalytics
from analytics.water import WaterAnalytics
from analytics.waste import WasteAnalytics
from analytics.air import AirAnalytics
from engine.recommend import RecommendationEngine
from engine.priority import PriorityEngine
from engine.scorecard import ScorecardEngine
from genai.brief import generate_ai_brief
from backend.external_apis import ExternalAPIClient

app = FastAPI(
    title="Dhara Estate Intelligence API",
    description="Backend REST API for 300-bed District Hospital Sustainability & Telemetry",
    version="1.0.0"
)

# Enable CORS for Streamlit and web frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize engines and cached pipeline
pipeline = DharaDataPipeline()
external_client = ExternalAPIClient(
    lat=pipeline.config["facility"]["coordinates"][0],
    lon=pipeline.config["facility"]["coordinates"][1]
)

# In-memory storage for approved work orders
approved_work_orders = []

class WorkOrderApprovalRequest(BaseModel):
    order_id: str
    approved_by: Optional[str] = "Dr. Sharma (Medical Superintendent)"
    notes: Optional[str] = "Dispatched via Dhara Command Center"

class SimulationRequest(BaseModel):
    ac_early_shutoff_hours: int = 2
    fix_hostel_leak: bool = True
    extra_waste_trips: int = 1
    solar_pv_kw: int = 40

# --- ROUTES ---

@app.get("/")
def api_root():
    return {
        "system": "Dhara - Estate Intelligence Backend",
        "tagline": "हर परिसर, एक हरित भविष्य",
        "facility": pipeline.config["facility"]["name"],
        "status": "online",
        "active_version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "endpoints": [
            "/api/external/live",
            "/api/telemetry/live",
            "/api/energy/forecast",
            "/api/water/leak-audit",
            "/api/waste/bins",
            "/api/air/hotspots",
            "/api/recommendations",
            "/api/scorecard",
            "/api/brief",
            "/api/simulate"
        ]
    }

@app.get("/api/external/live")
def get_live_external_data():
    """Fetches real-time ambient weather & AQI for hospital coordinates."""
    weather = external_client.get_live_weather()
    aqi = external_client.get_live_ambient_aqi()
    grid = external_client.get_grid_carbon_intensity()
    return {
        "facility_coordinates": pipeline.config["facility"]["coordinates"],
        "weather": weather,
        "ambient_air": aqi,
        "grid_baseline": grid
    }

@app.get("/api/telemetry/live")
def get_live_telemetry():
    """Returns aggregated campus telemetry snapshot across hospital zones."""
    df_facility = pipeline.load_facility_data()
    latest_snapshot = pipeline.get_latest_snapshot(df_facility)
    
    total_power = float(latest_snapshot["power_kw"].sum())
    total_water = float(latest_snapshot["water_lph"].sum())
    avg_pm25 = float(latest_snapshot["pm25"].mean())
    
    zones_data = latest_snapshot[["zone_id", "power_kw", "water_lph", "pm25", "occupancy"]].to_dict(orient="records")
    
    return {
        "timestamp": datetime.now().isoformat(),
        "campus_total_power_kw": round(total_power, 1),
        "campus_total_water_lph": round(total_water, 1),
        "campus_avg_pm25": round(avg_pm25, 1),
        "zones": zones_data
    }

@app.get("/api/energy/forecast")
def get_energy_forecast(horizon_hours: int = Query(48, ge=12, le=72)):
    """Returns load forecast with 90% confidence bands."""
    df_facility = pipeline.load_facility_data()
    energy_engine = EnergyAnalytics(pipeline.config)
    fc_df = energy_engine.forecast_energy(df_facility, horizon_hours=horizon_hours)
    
    # Format for JSON
    fc_df["timestamp"] = fc_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return {
        "horizon_hours": horizon_hours,
        "model_used": fc_df["model_used"].iloc[0],
        "forecast": fc_df.to_dict(orient="records")
    }

@app.get("/api/water/leak-audit")
def get_water_leak_audit():
    """Runs 2 AM - 4 AM night-flow check and returns detected leaks."""
    df_facility = pipeline.load_facility_data()
    water_engine = WaterAnalytics(pipeline.config)
    audit = water_engine.analyze_water_network(df_facility)
    return audit

@app.get("/api/waste/bins")
def get_waste_bins_status():
    """Returns bin fill levels, overflow countdowns, and BMW compliance."""
    df_waste = pipeline.load_waste_data()
    waste_engine = WasteAnalytics(pipeline.config)
    status = waste_engine.analyze_waste_status(df_waste)
    
    # Convert dataframes to dictionaries
    return {
        "total_bins": status["total_bins"],
        "urgent_pickup_count": status["urgent_pickup_count"],
        "suggested_first_stop": status["suggested_first_stop"],
        "plain_advisory": status["plain_advisory"],
        "all_bins": status["all_bins"].to_dict(orient="records"),
        "critical_bins": status["critical_bins"].to_dict(orient="records"),
        "bmw_alerts": status["bmw_alerts"].to_dict(orient="records")
    }

@app.get("/api/air/hotspots")
def get_air_hotspots():
    """Returns CPCB ambient air quality categories and active hotspots."""
    df_facility = pipeline.load_facility_data()
    air_engine = AirAnalytics(pipeline.config)
    res = air_engine.analyze_air_quality(df_facility)
    return res

@app.get("/api/recommendations")
def get_recommendations():
    """Returns ranked priority actions using: Rank = Savings x Urgency x Ease."""
    df_facility = pipeline.load_facility_data()
    df_waste = pipeline.load_waste_data()

    energy_engine = EnergyAnalytics(pipeline.config)
    df_energy_anom = energy_engine.detect_anomalies(df_facility)
    opd_savings = energy_engine.calculate_night_waste_savings(df_energy_anom, "OPD")

    water_engine = WaterAnalytics(pipeline.config)
    water_res = water_engine.analyze_water_network(df_facility)

    waste_engine = WasteAnalytics(pipeline.config)
    waste_res = waste_engine.analyze_waste_status(df_waste)

    air_engine = AirAnalytics(pipeline.config)
    air_res = air_engine.analyze_air_quality(df_facility)

    rec_engine = RecommendationEngine(pipeline.config)
    recs = rec_engine.generate_recommendations(
        {"opd_night_savings": opd_savings, "anomalies_df": df_energy_anom[df_energy_anom["is_anomaly"]]},
        water_res,
        waste_res,
        air_res
    )

    p_engine = PriorityEngine(pipeline.config)
    ranked_recs, summary = p_engine.rank_recommendations(recs)

    # Format datetime objects for JSON serialization
    for r in ranked_recs:
        if "deadline_datetime" in r and isinstance(r["deadline_datetime"], datetime):
            r["deadline_datetime"] = r["deadline_datetime"].isoformat()

    return {
        "summary": summary,
        "recommendations": ranked_recs,
        "approved_orders": approved_work_orders
    }

@app.post("/api/workorders/approve")
def approve_work_order(req: WorkOrderApprovalRequest):
    """Signs and records an approved work order dispatch."""
    receipt = {
        "order_id": req.order_id,
        "approved_by": req.approved_by,
        "approved_at": datetime.now().isoformat(),
        "notes": req.notes,
        "dispatch_status": "DISPATCHED_TO_FACILITIES"
    }
    approved_work_orders.append(receipt)
    return {
        "status": "success",
        "message": f"Work order {req.order_id} signed and dispatched.",
        "receipt": receipt
    }

@app.get("/api/scorecard")
def get_scorecard():
    """Returns weighted A-D facility grade and radar pillar scores."""
    df_facility = pipeline.load_facility_data()
    df_waste = pipeline.load_waste_data()

    energy_engine = EnergyAnalytics(pipeline.config)
    df_energy_anom = energy_engine.detect_anomalies(df_facility)
    opd_savings = energy_engine.calculate_night_waste_savings(df_energy_anom, "OPD")

    water_res = WaterAnalytics(pipeline.config).analyze_water_network(df_facility)
    waste_res = WasteAnalytics(pipeline.config).analyze_waste_status(df_waste)
    air_res = AirAnalytics(pipeline.config).analyze_air_quality(df_facility)

    sc_engine = ScorecardEngine(pipeline.config)
    scorecard = sc_engine.evaluate_facility(
        {"opd_night_savings": opd_savings, "anomalies_df": df_energy_anom[df_energy_anom["is_anomaly"]]},
        water_res,
        waste_res,
        air_res
    )
    return scorecard

@app.get("/api/brief")
def get_executive_brief(language: str = Query("English", regex="^(English|Hindi|Bengali)$")):
    """Returns AI morning brief in English, Hindi, or Bengali."""
    # Obtain latest summary & scorecard
    recs_data = get_recommendations()
    scorecard = get_scorecard()
    combined_metrics = {**recs_data["summary"], **scorecard}
    
    brief_text = generate_ai_brief(combined_metrics, language=language)
    return {
        "language": language,
        "facility": pipeline.config["facility"]["name"],
        "persona": "Dr. Sharma (Medical Superintendent)",
        "briefing_text": brief_text,
        "generated_at": datetime.now().isoformat()
    }

@app.post("/api/simulate")
def run_simulation(sim: SimulationRequest):
    """Calculates what-if scenario savings and updated Eco-Grade."""
    tariff_elec = pipeline.config["tariffs_and_factors"]["electricity_tariff_inr_kwh"]
    tariff_water = pipeline.config["tariffs_and_factors"]["water_tariff_inr_kl"]
    emission_factor = pipeline.config["tariffs_and_factors"]["emission_factor_kg_co2_kwh"]

    sim_energy_kwh_saved = (38.5 * sim.ac_early_shutoff_hours * 30) + (sim.solar_pv_kw * 4.2 * 30)
    sim_water_kl_saved = (680.0 * 24 * 30 / 1000.0) if sim.fix_hostel_leak else 0.0
    sim_waste_penalty_avoided = sim.extra_waste_trips * 3500.0

    savings_inr = (
        (sim_energy_kwh_saved * tariff_elec) +
        (sim_water_kl_saved * tariff_water) +
        sim_waste_penalty_avoided
    )
    co2_avoided_kg = sim_energy_kwh_saved * emission_factor

    scorecard = get_scorecard()
    base_score = scorecard["overall_score"]
    proj_score = min(96.0, base_score + (18.0 if sim.fix_hostel_leak else 0.0) + (sim.ac_early_shutoff_hours * 3.5) + (sim.solar_pv_kw * 0.05))
    proj_grade = "A" if proj_score >= 85 else ("B" if proj_score >= 70 else "C")

    return {
        "inputs": sim.dict(),
        "projected_monthly_savings_inr": round(savings_inr, 0),
        "projected_annual_co2_avoided_tonnes": round((co2_avoided_kg * 12) / 1000.0, 1),
        "base_grade": scorecard["grade"],
        "projected_grade": proj_grade,
        "projected_score": round(proj_score, 1)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
