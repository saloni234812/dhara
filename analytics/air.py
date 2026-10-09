"""
Dhara - Air Quality Analytics Module
- CPCB (Central Pollution Control Board, India) threshold benchmarks
- Zone-level PM2.5 and PM10 hotspot detection by duration and severity
- Actionable emission mitigation advisories
"""

import numpy as np
import pandas as pd

class AirAnalytics:
    def __init__(self, config=None):
        self.config = config or {}
        self.standards = self.config.get("thresholds", {}).get("air", {}).get("cpcb_standards", {
            "pm25_good": 30,
            "pm25_moderate": 60,
            "pm25_poor": 90,
            "pm25_very_poor": 120,
            "pm10_moderate": 100,
            "pm10_poor": 250
        })
        self.min_consecutive_hours = self.config.get("thresholds", {}).get("air", {}).get("hotspot_min_consecutive_hours", 3)

    def analyze_air_quality(self, df_facility):
        """
        Detects persistent air quality hotspots and breaches:
        - Evaluates the recent 24-hour window for each zone.
        - Identifies duration of elevated PM2.5 (> 60 µg/m³ moderate, > 120 µg/m³ very poor).
        - Generates plain-language advisories.
        """
        df = df_facility.copy()
        
        # Recent 24 hours
        max_time = df["timestamp"].max()
        recent_df = df[df["timestamp"] >= (max_time - pd.Timedelta(hours=24))].sort_values(by="timestamp")
        
        hotspot_records = []
        zone_air_status = {}
        
        for zone_id, group in recent_df.groupby("zone_id"):
            group = group.sort_values(by="timestamp")
            latest_row = group.iloc[-1]
            latest_pm25 = latest_row["pm25"]
            latest_pm10 = latest_row["pm10"]
            
            # Check consecutive hours above moderate threshold (60 ug/m3)
            elevated_mask = (group["pm25"] >= self.standards["pm25_moderate"]).values
            
            # Count trailing consecutive elevated hours
            consecutive_elevated = 0
            for val in reversed(elevated_mask):
                if val:
                    consecutive_elevated += 1
                else:
                    break
                    
            # Determine CPCB category
            if latest_pm25 <= self.standards["pm25_good"]:
                category = "Good"
                color = "green"
            elif latest_pm25 <= self.standards["pm25_moderate"]:
                category = "Satisfactory"
                color = "lightgreen"
            elif latest_pm25 <= self.standards["pm25_poor"]:
                category = "Moderate"
                color = "orange"
            elif latest_pm25 <= self.standards["pm25_very_poor"]:
                category = "Poor"
                color = "darkorange"
            elif latest_pm25 <= 250:
                category = "Very Poor"
                color = "red"
            else:
                category = "Severe"
                color = "purple"
                
            is_hotspot = (consecutive_elevated >= self.min_consecutive_hours) and (latest_pm25 >= self.standards["pm25_moderate"])
            
            # Plain language advisory
            if is_hotspot:
                if zone_id == "Generator":
                    action = f"Generator area PM2.5 is {round(latest_pm25, 0)} µg/m³ ({category}), elevated for {consecutive_elevated} hours. Verify diesel particulate filter (DPF), service fuel injectors, and avoid non-essential idle running."
                elif zone_id == "Parking":
                    action = f"Parking & Ambulance bay PM2.5 elevated for {consecutive_elevated} hours ({round(latest_pm25, 0)} µg/m³). Send advisory to security guards to enforce strict 3-minute anti-idling policy for ambulances and visitor cabs."
                elif zone_id == "Canteen":
                    action = f"Canteen exhaust PM2.5 elevated for {consecutive_elevated} hours ({round(latest_pm25, 0)} µg/m³). Clean kitchen chimney electrostatic precipitators and verify kitchen ventilation."
                else:
                    action = f"{zone_id} PM2.5 elevated ({round(latest_pm25, 0)} µg/m³) for {consecutive_elevated} hours. Inspect air filter dampers and ensure windows remain shut."
            else:
                action = f"Air quality in {zone_id} is within safe CPCB baseline ({round(latest_pm25, 0)} µg/m³, {category})."

            rec = {
                "zone_id": zone_id,
                "latest_pm25": round(latest_pm25, 1),
                "latest_pm10": round(latest_pm10, 1),
                "category": category,
                "color": color,
                "consecutive_elevated_hours": consecutive_elevated,
                "is_hotspot": is_hotspot,
                "action_advisory": action
            }
            
            zone_air_status[zone_id] = rec
            if is_hotspot:
                hotspot_records.append(rec)
                
        return {
            "all_zones": zone_air_status,
            "hotspots": hotspot_records,
            "active_hotspot_count": len(hotspot_records),
            "campus_avg_pm25": round(recent_df[recent_df["timestamp"] == max_time]["pm25"].mean(), 1)
        }
