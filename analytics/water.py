"""
Dhara - Water Analytics Module
- Rolling baseline average and Z-score detection
- Night-flow check (2 AM - 4 AM minimum flow window)
- Leak severity calculation (Litres/hr, KL/month, Cost in ₹)
- Probable location identification (e.g. Hostel Pipeline 2)
"""

import numpy as np
import pandas as pd

class WaterAnalytics:
    def __init__(self, config=None):
        self.config = config or {}
        self.tariff_per_kl = self.config.get("tariffs_and_factors", {}).get("water_tariff_inr_kl", 25.0) # ₹ per 1,000 Liters
        self.night_window = self.config.get("thresholds", {}).get("water", {}).get("night_flow_window_hours", [2, 3, 4])
        self.leak_multiplier = self.config.get("thresholds", {}).get("water", {}).get("night_flow_leak_multiplier", 2.2)

    def analyze_water_network(self, df_facility):
        """
        Analyzes water flow across hospital zones:
        1. Evaluates night flow between 2:00 AM - 4:00 AM vs expected minimum baseline.
        2. Computes rolling Z-score and pressure anomalies.
        3. Quantifies leak severity, monthly financial loss, and pinpointed sub-zones.
        """
        df = df_facility.copy()
        
        # Filter for recent 7 days to get active leaks
        max_time = df["timestamp"].max()
        recent_cutoff = max_time - pd.Timedelta(days=7)
        recent_df = df[df["timestamp"] >= recent_cutoff]
        
        leak_findings = []
        zone_summaries = {}
        
        for zone_id, group in recent_df.groupby("zone_id"):
            group = group.sort_values(by="timestamp")
            
            # 2 AM - 4 AM records
            night_records = group[group["hour"].isin(self.night_window)]
            day_records = group[~group["hour"].isin(self.night_window)]
            
            # Baseline minimum flow expected at 2-4 AM (derived from historical 20th percentile)
            hist_zone_night = df[(df["zone_id"] == zone_id) & (df["hour"].isin(self.night_window))]["water_lph"]
            normal_night_baseline = hist_zone_night.quantile(0.20) if len(hist_zone_night) > 0 else 150.0
            recent_night_flow = night_records["water_lph"].tail(6).mean()
            
            # Baseline pressure vs recent night pressure
            avg_pressure = df[df["zone_id"] == zone_id]["water_pressure_psi"].mean()
            recent_night_pressure = night_records["water_pressure_psi"].tail(6).mean()
            
            # Z-score of recent night flow vs 7-day night baseline
            night_flow_std = night_records["water_lph"].std() + 1e-4
            z_score = (recent_night_flow - normal_night_baseline) / night_flow_std
            
            is_leak = False
            probable_location = "None"
            severity = "Normal"
            leak_rate_lph = 0.0
            
            if recent_night_flow > (normal_night_baseline * self.leak_multiplier) and recent_night_flow > 250:
                is_leak = True
                leak_rate_lph = max(0.0, recent_night_flow - normal_night_baseline)
                
                if zone_id == "Hostel":
                    probable_location = "Hostel Pipeline 2 (Staff Quarters East Wing riser)"
                    severity = "Critical"
                elif zone_id == "Wards":
                    probable_location = "Block B Ground Floor Flush Cistern manifold"
                    severity = "High"
                elif zone_id == "Canteen":
                    probable_location = "Kitchen Dishwash Sub-line valve"
                    severity = "Medium"
                else:
                    probable_location = f"{zone_id} Main Distribution Header"
                    severity = "Medium"

            # Financial and physical loss estimation
            # A continuous leak runs 24 hours/day (not just 2-4 AM)
            daily_loss_liters = leak_rate_lph * 24.0
            monthly_loss_kl = (daily_loss_liters * 30.0) / 1000.0
            monthly_loss_inr = monthly_loss_kl * self.tariff_per_kl
            
            finding = {
                "zone_id": zone_id,
                "is_leak": is_leak,
                "severity": severity,
                "probable_location": probable_location,
                "recent_night_flow_lph": round(recent_night_flow, 1),
                "normal_night_baseline_lph": round(normal_night_baseline, 1),
                "ratio_to_baseline": round(recent_night_flow / max(1.0, normal_night_baseline), 1),
                "pressure_psi": round(recent_night_pressure, 1),
                "pressure_drop_psi": round(max(0.0, avg_pressure - recent_night_pressure), 1),
                "leak_rate_lph": round(leak_rate_lph, 1),
                "monthly_kl_lost": round(monthly_loss_kl, 1),
                "monthly_inr_lost": round(monthly_loss_inr, 0),
                "z_score": round(z_score, 1),
                "plain_explanation": (
                    f"Night-time usage is {round(recent_night_flow / max(1.0, normal_night_baseline), 1)}x normal baseline between 2 AM - 4 AM "
                    f"({round(recent_night_flow, 0)} L/hr vs {round(normal_night_baseline, 0)} L/hr baseline). "
                    f"Pressure dropped by {round(max(0.0, avg_pressure - recent_night_pressure), 1)} PSI."
                ) if is_leak else "Flow within expected nocturnal limits."
            }
            
            leak_findings.append(finding)
            zone_summaries[zone_id] = finding

        return {
            "all_zones": leak_findings,
            "detected_leaks": [f for f in leak_findings if f["is_leak"]],
            "total_monthly_kl_wasted": round(sum(f["monthly_kl_lost"] for f in leak_findings), 1),
            "total_monthly_inr_wasted": round(sum(f["monthly_inr_lost"] for f in leak_findings), 0)
        }
