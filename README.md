# धारा (Dhara) — AI-Powered Sustainable Facility & Estate Intelligence Dashboard

> **Tagline:** *"हर परिसर, एक हरित भविष्य"*  
> **One-Line Pitch:** *"From scattered data to 'what should I do today?'"*  
> **Reference Facility:** 300-bed District Government Hospital, Ramnagar, India  
> **Demo Persona:** Dr. Sharma, Medical Superintendent (needs plain-language decisions, not complex charts)

---

## 🌟 Executive Overview

Public sector estates and district hospitals generate gigabytes of telemetry across sub-meters, flow sensors, smart bins, and ambient air monitors. Yet facility directors and Medical Superintendents are left asking: **"What should my team fix today to stop wasting money and water?"**

**Dhara** bridges this gap. It continuously ingests campus telemetry, detects cross-domain anomalies, and translates them into **prioritized, plain-language work orders** with transparent financial savings (₹) and carbon abatement figures.

---

## 🏗️ Architecture & Modules

```
dhara/
  ├── config.yaml          # Campus zones, tariffs, CPCB thresholds, weights, emission factors
  ├── requirements.txt     # Python package dependencies
  ├── README.md            # Setup guide & 3-minute pitch demo script
  ├── server.py            # FastAPI Backend REST API Server (Port 8000)
  ├── backend/
  │   └── external_apis.py # Connectors to Open-Meteo Weather, Live AQI, and CEA Grid APIs
  ├── data/
  │   ├── generator.py     # 30-day realistic hospital data generator + 4 injected faults
  │   └── pipeline.py      # Telemetry loading, cleaning, rolling Z-score feature engineering
  ├── analytics/
  │   ├── energy.py        # 48h load forecast (Prophet / Fourier-Ridge) + Isolation Forest
  │   ├── water.py         # 2 AM - 4 AM night-flow check, leak severity & location isolation
  │   ├── waste.py         # Bin fill rates, overflow countdown, smart collection routing, BMW rules
  │   └── air.py           # CPCB air quality classification, multi-hour hotspot detection
  ├── engine/
  │   ├── recommend.py     # Cross-domain rules engine + transparent ₹ & CO₂ savings formulas
  │   ├── priority.py      # Composite ranking: Rank = Savings × Urgency × Ease of Fix
  │   └── scorecard.py     # Weighted facility rating (A-D) across Energy, Water, Waste & Air
  ├── genai/
  │   └── brief.py         # Grounded daily executive brief in English, Hindi, and Bengali
  └── app.py               # Interactive Streamlit dashboard with Folium map & simulator
```

---

## 🔌 Backend REST API & External Integrations

Dhara is powered by a high-performance **FastAPI** backend that exposes clean REST endpoints and connects to live external environmental telemetry:

### 1. External Public API Connectors:
- **Live Ambient Weather (Open-Meteo API):** Real-time temperature (°C), relative humidity (%), and barometric pressure for Ramnagar hospital coordinates (`26.8467°N, 80.9462°E`).
- **Live Ambient Air Quality (Open-Meteo AQI API):** Real-time ambient PM2.5 and PM10 telemetry.
- **National Carbon Intensity Benchmark:** Central Electricity Authority (CEA) Version 19 grid emissions baseline (`0.75 kg CO₂/kWh`).

### 2. Core REST Endpoints (`http://127.0.0.1:8000`):
- `GET /` — API root, heartbeat, and service directory.
- `GET /api/external/live` — Live real-time ambient weather & AQI telemetry.
- `GET /api/telemetry/live` — Campus-wide sub-meter power, water flow, and zone loads.
- `GET /api/energy/forecast` — 48-hour electrical demand forecast with 90% confidence bands.
- `GET /api/water/leak-audit` — 2 AM - 4 AM night flow leak audit with riser isolation location.
- `GET /api/waste/bins` — Telemetry fill levels, overflow countdown, and 48-hr BMW compliance.
- `GET /api/air/hotspots` — CPCB particulate hotspot warnings.
- `GET /api/recommendations` — Prioritized executive actions (Rank = Savings × Urgency × Ease).
- `POST /api/workorders/approve` — Signs and records physical work order dispatch receipts.
- `GET /api/scorecard` — Weighted A-D facility grade and radar pillar scores.
- `GET /api/brief?language=Hindi` — Grounded AI executive morning briefing.
- `POST /api/simulate` — What-if scenario simulation engine.

Interactive OpenAPI Documentation:
- **Swagger UI:** `http://127.0.0.1:8000/docs`
- **ReDoc:** `http://127.0.0.1:8000/redoc`

---

## 🚀 Setup & Quickstart

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.10 – 3.14)
- Git

### 2. Installation
```bash
# Clone or navigate to the directory
cd dhara

# Install required dependencies
pip install -r requirements.txt
```

### 3. Generate Initial Datasets & Run Anomaly Test
```bash
python -m data.generator
```

### 4. Launch Streamlit Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

*(Optional)* To enable live LLM rephrasing with Anthropic:
```bash
set ANTHROPIC_API_KEY=your_key_here   # Windows PowerShell: $env:ANTHROPIC_API_KEY="your_key_here"
```
*Note: If no API key is provided, Dhara automatically defaults to deterministic executive templates in English, Hindi, and Bengali.*

---

## 🎬 3-Minute Demo Script (Pitch & Walkthrough)

| Time | Phase | Spoken Narrative & Actions |
| :--- | :--- | :--- |
| **0:00 - 0:20** | **1. The Problem** | *"District hospitals run on tight public funds. In a 300-bed hospital like Ramnagar, HVAC runs in empty halls at night, water pipes leak unnoticed behind hostel walls, and bins overflow during lunch rushes. Administrators like Dr. Sharma don't have time to stare at SCADA charts—they need to know what to fix this morning."* |
| **0:20 - 1:00** | **2. Overview & Dr. Sharma Persona** | *(Point to the header & KPI cards on Administrator View)*<br>*"Meet Dhara. Notice the top banner: Dr. Sharma immediately sees that by clearing just 3 work orders today, he can recover ₹58,000 this month and avoid 4,800 kg of CO₂. The AI Briefing greets him directly in Hindi, Bengali, or English, outlining his morning priorities without technical jargon."* |
| **1:00 - 1:40** | **3. Caught Anomaly & Folium Map** | *(Scroll to the Priority Cards & Campus Spatial Map)*<br>*"Look at the spatial map: Red marker on Doctors' Hostel. Dhara's water analytics ran a 2 AM–4 AM night-flow audit and flagged a 7.5x surge on Pipeline 2 riser—wasting 690 L/hr. Right beside it, orange on Block C: Isolation Forest caught the OPD AC running at 11 PM with zero occupants. Each card shows the exact formula: Load × Hours × Tariff."* |
| **1:40 - 2:20** | **4. What-If Scenario Simulator** | *(Expand the Scenario Simulator at the bottom)*<br>*"What if Dr. Sharma decides to act? Let's toggle 'Fix Hostel Pipeline 2 leak' and slide 'Switch AC off 2 hours earlier'. Watch the numbers: Projected monthly recovery jumps instantly to ₹62,000+, and the hospital's Eco-Grade lifts from C to A."* |
| **2:20 - 2:40** | **5. AI Brief & Role Switching** | *(Switch sidebar role to 'Operations Team' and language to 'Hindi')*<br>*"When Dr. Sharma signs off, his Operations Team switches to their view: a live shift checklist with deadlines, 48h load forecast confidence bands, and a smart route for waste carts prioritizing Canteen Bins 4, 7, and 9 before 2 PM lunch overflow."* |
| **2:40 - 3:00** | **6. Impact & Vision** | *(Switch to Sustainability Officer view)*<br>*"Dhara provides transparent carbon accounting based on CEA baseline factors. From scattered sensors to immediate action: 'हर परिसर, एक हरित भविष्य'."* |

---

## ⚖️ Engineering Assumptions & Formulas

- **Grid Emission Factor:** `0.75 kg CO₂/kWh` (Reference: Central Electricity Authority CO₂ Baseline Database v19; verify latest CEA figure).
- **Electricity Tariff:** `₹8.50 / kWh` (Commercial/Institutional slab).
- **Water Tariff:** `₹25.00 / KL` (Municipal bulk water tariff) + `0.55 kWh/KL` pumping power tariff.
- **Biomedical Waste Standard:** `48 Hours` maximum storage compliance under Bio-Medical Waste Management Rules, 2016.
- **Decision-Support Disclaimer:** *Outputs are decision-support estimates to prioritize human action, not official accredited billing or legal measurements.*
