"""
Dhara - Recommendation Engine
Synthesizes cross-domain analytics (Energy, Water, Waste, Air) into actionable, plain-language tasks.
Includes transparent financial formulas (₹ savings) and carbon impact (CO2 avoided).
"""

from datetime import datetime, timedelta

class RecommendationEngine:
    def __init__(self, config=None):
        self.config = config or {}
        self.electricity_tariff = self.config.get("tariffs_and_factors", {}).get("electricity_tariff_inr_kwh", 8.50)
        self.water_tariff = self.config.get("tariffs_and_factors", {}).get("water_tariff_inr_kl", 25.00)
        self.emission_factor = self.config.get("tariffs_and_factors", {}).get("emission_factor_kg_co2_kwh", 0.75)

    def generate_recommendations(self, energy_analysis, water_analysis, waste_analysis, air_analysis):
        """
        Generates actionable recommendations with:
        - Plain language action
        - Zone
        - Deadline
        - Estimated savings in ₹
        - CO2 avoided in kg
        - Visible calculation formula
        - Raw factors for priority scoring: savings_score, urgency_score, ease_of_fix_score
        """
        recommendations = []
        now = datetime.now()

        # 1. Energy Recommendations: Block C / OPD Night HVAC
        # Check energy anomalies in OPD / Block C
        opd_night_savings = energy_analysis.get("opd_night_savings", {})
        if opd_night_savings and opd_night_savings.get("monthly_savings_inr", 0) > 0:
            monthly_inr = opd_night_savings["monthly_savings_inr"]
            monthly_co2 = opd_night_savings["monthly_co2_kg"]
            excess_kw = opd_night_savings["avg_excess_kw"]
            excess_hrs = opd_night_savings["excess_hours_per_night"]
            
            recommendations.append({
                "id": "REC-ENG-01",
                "domain": "Energy",
                "zone": "OPD & Registration Wing (Block C)",
                "action": "Block C AC runs at 11 PM with near-zero occupancy. Move to an automated timed schedule (auto shut-off at 8:00 PM).",
                "short_title": "Automate Block C Night AC Shut-off",
                "deadline": "Tonight before 10:30 PM",
                "deadline_datetime": (now.replace(hour=22, minute=30, second=0)),
                "estimated_monthly_savings_inr": monthly_inr,
                "estimated_monthly_co2_kg": monthly_co2,
                "formula_display": f"{excess_kw} kW excess × {excess_hrs} hrs/night × 30 days × ₹{self.electricity_tariff}/kWh = ₹{int(monthly_inr):,}/month | CO2 = {excess_kw * excess_hrs * 30:,.0f} kWh × {self.emission_factor} kg/kWh = {int(monthly_co2):,} kg CO2",
                "urgency_score": 9,      # High urgency (happens every night)
                "ease_of_fix_score": 8,  # Easy: BMS timer configuration or security guard checklist
                "financial_score": min(10, int(monthly_inr / 5000))
            })

        # 2. Water Recommendations: Hostel Pipeline 2 Leak
        detected_leaks = water_analysis.get("detected_leaks", [])
        for leak in detected_leaks:
            loc = leak["probable_location"]
            zone = leak["zone_id"]
            ratio = leak["ratio_to_baseline"]
            lph = leak["leak_rate_lph"]
            monthly_inr = leak["monthly_inr_lost"]
            monthly_kl = leak["monthly_kl_lost"]
            
            # Pumping energy factor for water: ~0.5 kWh per KL lifted to overhead hospital tank
            pumping_kwh_saved = monthly_kl * 0.55
            co2_avoided = pumping_kwh_saved * self.emission_factor

            recommendations.append({
                "id": f"REC-WAT-{zone[:3].upper()}",
                "domain": "Water",
                "zone": f"{zone} ({loc})",
                "action": f"Hostel night usage is {ratio}x normal between 2 AM - 4 AM ({round(leak['recent_night_flow_lph'], 0)} L/hr). Probable leak on Pipeline 2 riser. Dispatch plumber to isolate valve and replace worn seal.",
                "short_title": f"Isolate & Repair {loc}",
                "deadline": "Immediate (Within 4 hours)",
                "deadline_datetime": (now + timedelta(hours=4)),
                "estimated_monthly_savings_inr": monthly_inr + (pumping_kwh_saved * self.electricity_tariff),
                "estimated_monthly_co2_kg": round(co2_avoided, 1),
                "formula_display": f"{lph} L/hr leak × 24 hrs × 30 days = {monthly_kl:,.1f} KL lost/mo × ₹{self.water_tariff}/KL + pumping tariff saved = ₹{int(monthly_inr + pumping_kwh_saved * self.electricity_tariff):,}/mo",
                "urgency_score": 10,     # Critical: ongoing 24/7 loss of potable hospital water
                "ease_of_fix_score": 7,  # Moderate: plumber pipe isolation
                "financial_score": 8
            })

        # 3. Waste Recommendations: Canteen Bins Overflow Dispatch
        critical_bins = waste_analysis.get("critical_bins", None)
        if critical_bins is not None and len(critical_bins) > 0:
            bin_ids_str = ", ".join(str(b) for b in critical_bins["bin_id"].tolist())
            first_overflow = critical_bins.iloc[0]["predicted_overflow_time"]
            
            # Cost impact: avoids emergency cleanup penalty, pest infestation, and haulage surcharges
            estimated_waste_cost_saving = 6500.0
            co2_avoided_methane = 140.0 # Diverting organic waste before open rotting prevents methane
            
            recommendations.append({
                "id": "REC-WST-01",
                "domain": "Waste",
                "zone": "Canteen & Kitchen Compound",
                "action": f"Bins {bin_ids_str} overflow by {first_overflow} due to lunch rush. Send the waste collection vehicle to the canteen side first before proceeding to wards.",
                "short_title": f"Priority Dispatch: Bins {bin_ids_str} (Canteen)",
                "deadline": f"Today by {first_overflow}",
                "deadline_datetime": (now + timedelta(hours=2)),
                "estimated_monthly_savings_inr": estimated_waste_cost_saving,
                "estimated_monthly_co2_kg": co2_avoided_methane,
                "formula_display": f"Optimized routing prevents 2 emergency haulage runs + spillage penalties = ₹6,500/mo estimated avoidance",
                "urgency_score": 9,      # Overflow imminent
                "ease_of_fix_score": 9,  # Very high ease: re-route internal collection cart
                "financial_score": 6
            })

        # 4. Biomedical Waste Compliance Alert
        bmw_alerts = waste_analysis.get("bmw_alerts", None)
        if bmw_alerts is not None and len(bmw_alerts) > 0:
            bmw_row = bmw_alerts.iloc[0]
            recommendations.append({
                "id": "REC-BMW-02",
                "domain": "Biomedical Compliance",
                "zone": f"{bmw_row['zone']} (Bin #{bmw_row['bin_id']} - {bmw_row['bin_color']})",
                "action": f"Bio-Medical Waste Bin #{bmw_row['bin_id']} in {bmw_row['zone']} has been stored for {bmw_row['hours_since_emptied']} hours (approaching mandatory 48-hour regulatory deadline under BMW Rules 2016). Hand over to CPCB-authorized CBWTF vehicle immediately.",
                "short_title": f"CBWTF Handover for Bin #{bmw_row['bin_id']} ({bmw_row['zone']})",
                "deadline": "Today within 3 hours",
                "deadline_datetime": (now + timedelta(hours=3)),
                "estimated_monthly_savings_inr": 25000.0, # Statutory penalty avoidance
                "estimated_monthly_co2_kg": 45.0,
                "formula_display": "Avoids statutory non-compliance notice under BMW Management Rules 2016 (estimated compliance value ₹25,000)",
                "urgency_score": 10,
                "ease_of_fix_score": 8,
                "financial_score": 7
            })

        # 5. Air Quality Recommendations: Generator & Parking PM2.5 Surge
        hotspots = air_analysis.get("hotspots", [])
        for hs in hotspots:
            zid = hs["zone_id"]
            hrs = hs["consecutive_elevated_hours"]
            pm25_val = hs["latest_pm25"]
            
            if zid == "Generator":
                recommendations.append({
                    "id": "REC-AIR-GEN",
                    "domain": "Air Quality",
                    "zone": "Backup DG Generator & Utility Yard",
                    "action": f"Generator area PM2.5 elevated for {hrs} hours ({round(pm25_val, 0)} µg/m³). Service fuel injectors, inspect particulate scrubbers, and switch back to grid main as voltage is stable.",
                    "short_title": "DG Generator Fuel Injector & Scrubber Check",
                    "deadline": "Today within 2 hours",
                    "deadline_datetime": (now + timedelta(hours=2)),
                    "estimated_monthly_savings_inr": 18500.0, # Diesel fuel optimization vs efficient running
                    "estimated_monthly_co2_kg": 420.0,
                    "formula_display": "Eliminates ~12L diesel wastage/week from suboptimal combustion = ₹18,500/mo; 155 kg CO2 avoided",
                    "urgency_score": 8,
                    "ease_of_fix_score": 7,
                    "financial_score": 7
                })
            elif zid == "Parking":
                recommendations.append({
                    "id": "REC-AIR-PRK",
                    "domain": "Air Quality",
                    "zone": "Ambulance Bay & Public Parking",
                    "action": f"Parking PM2.5 elevated for {hrs} hours ({round(pm25_val, 0)} µg/m³). Send an advisory to security guards to restrict idling for ambulances and visitor cabs.",
                    "short_title": "Enforce Parking Anti-Idling Advisory",
                    "deadline": "Immediate (Within 1 hour)",
                    "deadline_datetime": (now + timedelta(hours=1)),
                    "estimated_monthly_savings_inr": 4500.0,
                    "estimated_monthly_co2_kg": 180.0,
                    "formula_display": "Reduces idle idling emissions near patient entrance; fuel & public health cost avoidance",
                    "urgency_score": 7,
                    "ease_of_fix_score": 10, # Very easy: notify guards via PA / walkie-talkie
                    "financial_score": 5
                })

        return recommendations
