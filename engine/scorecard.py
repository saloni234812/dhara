"""
Dhara - Facility Sustainability Scorecard
Computes weighted composite score (0-100) and Letter Grade (A to D)
Evaluates Energy (35%), Water (25%), Waste (20%), Air & Compliance (20%).
"""

class ScorecardEngine:
    def __init__(self, config=None):
        self.config = config or {}
        score_weights = self.config.get("weights", {}).get("scorecard", {})
        self.w_energy = score_weights.get("energy", 0.35)
        self.w_water = score_weights.get("water", 0.25)
        self.w_waste = score_weights.get("waste", 0.20)
        self.w_air = score_weights.get("air", 0.20)

    def evaluate_facility(self, energy_analysis, water_analysis, waste_analysis, air_analysis):
        """
        Calculates domain efficiency scores and overall composite grade (A-D).
        """
        # 1. Energy Score (0 - 100)
        # Penalized for active night HVAC anomalies and excess unoptimized consumption
        energy_anomalies_count = len(energy_analysis.get("anomalies_df", []))
        opd_excess_kw = energy_analysis.get("opd_night_savings", {}).get("avg_excess_kw", 0.0)
        
        energy_score = 100.0
        if opd_excess_kw > 20:
            energy_score -= 28.0 # Heavy night waste
        elif opd_excess_kw > 5:
            energy_score -= 15.0
            
        if energy_anomalies_count > 10:
            energy_score -= 12.0
        energy_score = max(30.0, min(100.0, energy_score))

        # 2. Water Score (0 - 100)
        # Penalized for undetected/unrepaired leaks and high night-flow ratios
        detected_leaks = water_analysis.get("detected_leaks", [])
        water_score = 100.0
        if len(detected_leaks) > 0:
            for leak in detected_leaks:
                ratio = leak.get("ratio_to_baseline", 1.0)
                if ratio > 3.0:
                    water_score -= 35.0 # Critical leak (Hostel 4x)
                elif ratio > 1.8:
                    water_score -= 18.0
        water_score = max(25.0, min(100.0, water_score))

        # 3. Waste Score (0 - 100)
        # Penalized for overflowing bins and biomedical non-compliance
        critical_bins_count = waste_analysis.get("urgent_pickup_count", 0)
        bmw_alerts_count = len(waste_analysis.get("bmw_alerts", []))
        
        waste_score = 95.0
        waste_score -= (critical_bins_count * 12.0)
        waste_score -= (bmw_alerts_count * 20.0)
        waste_score = max(20.0, min(100.0, waste_score))

        # 4. Air Quality & Compliance Score (0 - 100)
        # Evaluates CPCB compliance and active particulate hotspots
        active_hotspots = air_analysis.get("active_hotspot_count", 0)
        campus_avg_pm25 = air_analysis.get("campus_avg_pm25", 50.0)
        
        air_score = 90.0
        if campus_avg_pm25 > 90:
            air_score -= 25.0
        elif campus_avg_pm25 > 60:
            air_score -= 12.0
            
        air_score -= (active_hotspots * 15.0)
        air_score = max(25.0, min(100.0, air_score))

        # Composite Weighted Score
        composite_score = (
            (energy_score * self.w_energy) +
            (water_score * self.w_water) +
            (waste_score * self.w_waste) +
            (air_score * self.w_air)
        )
        composite_score = round(composite_score, 1)

        # Grade Letter & Badge
        if composite_score >= 85.0:
            grade = "A"
            grade_title = "Exemplary Green Hospital"
            color = "#10B981" # Green
            badge_class = "success"
        elif composite_score >= 70.0:
            grade = "B"
            grade_title = "Good - Minor Operational Gaps"
            color = "#3B82F6" # Blue
            badge_class = "info"
        elif composite_score >= 50.0:
            grade = "C"
            grade_title = "Needs Attention - Active Resource Leaks"
            color = "#F59E0B" # Amber
            badge_class = "warning"
        else:
            grade = "D"
            grade_title = "Critical Action Required - Non-Compliant"
            color = "#EF4444" # Red
            badge_class = "danger"

        return {
            "overall_score": composite_score,
            "grade": grade,
            "grade_title": grade_title,
            "grade_color": color,
            "badge_class": badge_class,
            "domain_scores": {
                "energy": round(energy_score, 1),
                "water": round(water_score, 1),
                "waste": round(waste_score, 1),
                "air": round(air_score, 1)
            },
            "weights": {
                "energy": f"{int(self.w_energy * 100)}%",
                "water": f"{int(self.w_water * 100)}%",
                "waste": f"{int(self.w_waste * 100)}%",
                "air": f"{int(self.w_air * 100)}%"
            }
        }
