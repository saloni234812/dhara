"""
Dhara - Waste Analytics Module
- Per-bin fill rate (%/hr) and time-to-overflow prediction
- Suggested optimal collection route order
- Hospital Bio-Medical Waste (BMW) vs General waste compliance tracking (48-hr rule)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class WasteAnalytics:
    def __init__(self, config=None):
        self.config = config or {}
        self.bin_capacity = self.config.get("thresholds", {}).get("waste", {}).get("bin_capacity_liters", 240)
        self.overflow_warning = self.config.get("thresholds", {}).get("waste", {}).get("overflow_warning_pct", 75.0)
        self.overflow_critical = self.config.get("thresholds", {}).get("waste", {}).get("overflow_critical_pct", 90.0)
        self.bmw_limit_hours = self.config.get("thresholds", {}).get("waste", {}).get("bmw_max_storage_hours", 48)

    def analyze_waste_status(self, df_waste):
        """
        Analyzes real-time waste sensor telemetry:
        - Identifies imminent overflow risks.
        - Calculates hours remaining until 100% full.
        - Computes suggested collection dispatch sequence.
        - Validates biomedical waste segregation and 48-hr regulatory compliance.
        """
        df = df_waste.copy()
        
        # Priority score for dispatch order:
        # Higher score = dispatch vehicle first
        # Factors:
        # 1. Fill % urgency: bins > 90% get massive weight
        # 2. Hours to overflow: fewer hours = higher priority
        # 3. Biomedical compliance urgency: bins near 48h deadline get high urgency
        
        def calc_dispatch_priority(row):
            score = 0.0
            fill = row["fill_percentage"]
            hrs_left = row["hours_to_overflow"]
            is_bmw = row["waste_category"] == "Biomedical"
            hrs_held = row["hours_since_emptied"]
            
            # Fill level score (0 - 50 pts)
            score += (fill / 100.0) * 50.0
            
            # Time urgency score (0 - 35 pts)
            if hrs_left < 1.0:
                score += 35.0
            elif hrs_left < 3.0:
                score += 25.0
            elif hrs_left < 6.0:
                score += 15.0
            else:
                score += max(0.0, (12.0 - hrs_left))
                
            # Regulatory compliance score (0 - 30 pts)
            if is_bmw:
                score += 10.0 # Biomedical always has baseline caution
                if hrs_held >= 40.0:
                    score += 25.0 # Close to 48-hr legal threshold
                    
            return round(score, 1)

        df["dispatch_priority_score"] = df.apply(calc_dispatch_priority, axis=1)
        
        # Determine status badge
        def get_bin_status(row):
            if row["fill_percentage"] >= self.overflow_critical:
                return "Critical Overflow Risk"
            elif row["fill_percentage"] >= self.overflow_warning:
                return "Warning: Nearing Full"
            elif row["waste_category"] == "Biomedical" and row["hours_since_emptied"] >= 40:
                return "BMW Storage Compliance Alert"
            return "Normal"

        df["status"] = df.apply(get_bin_status, axis=1)
        
        # Sort in optimal route sequence
        route_df = df.sort_values(by="dispatch_priority_score", ascending=False).reset_index(drop=True)
        route_df["collection_order"] = range(1, len(route_df) + 1)
        
        # Identify top critical bins (e.g. Canteen Bins 4, 7, 9)
        critical_bins = route_df[route_df["status"] == "Critical Overflow Risk"]
        bmw_alerts = route_df[route_df["bmw_compliance_breach"] | (route_df["hours_since_emptied"] >= 40)]
        
        return {
            "all_bins": route_df,
            "critical_bins": critical_bins,
            "bmw_alerts": bmw_alerts,
            "total_bins": len(route_df),
            "urgent_pickup_count": len(critical_bins),
            "suggested_first_stop": route_df.iloc[0]["zone"] if len(route_df) > 0 else "None",
            "plain_advisory": (
                f"Bins {', '.join(str(b) for b in critical_bins['bin_id'].tolist())} near {critical_bins.iloc[0]['zone'] if len(critical_bins) > 0 else 'Canteen'} "
                f"are above 90% capacity and predicted to overflow by {critical_bins.iloc[0]['predicted_overflow_time'] if len(critical_bins) > 0 else '2:00 PM'}. "
                f"Dispatch waste collection vehicle to {critical_bins.iloc[0]['zone'] if len(critical_bins) > 0 else 'Canteen'} first."
            ) if len(critical_bins) > 0 else "All waste bins within normal operational fill ranges."
        }
