"""
Dhara - Facility & Estate Intelligence System
Synthetic Data Generator with Fault Injection for 300-bed District Hospital
"""

import os
import yaml
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def load_config(config_path="config.yaml"):
    if not os.path.isabs(config_path):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def generate_facility_data(days=30, end_date=None, config=None, inject_faults=True):
    """
    Generate realistic hourly time series data for 30 days across hospital zones:
    - Energy (kW, kWh, Power Factor, Voltage)
    - Water (L/hr, Pressure PSI, Pipeline flow)
    - Air Quality (PM2.5, PM10, CO2 ppm, Temperature C, Humidity %)
    - Waste (Fill %, Bin ID, Type, Zone, Last emptied hours ago)
    """
    if config is None:
        config = load_config()

    if end_date is None:
        end_date = datetime.now().replace(minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=days)
    
    timestamps = pd.date_range(start=start_date, end=end_date, freq="h")
    n_hours = len(timestamps)
    
    zones = [z["id"] for z in config["zones"]]
    zone_lookup = {z["id"]: z for z in config["zones"]}
    
    records = []
    
    # We will generate data point for each zone and timestamp
    for t in timestamps:
        hour = t.hour
        day_of_week = t.weekday() # 0 = Monday, 6 = Sunday
        is_weekend = day_of_week in [5, 6]
        is_sunday = day_of_week == 6
        
        # Diurnal ambient temperature variation (India summer/monsoon ~26C to 38C)
        ambient_temp = 28.0 + 8.0 * np.sin(np.pi * (hour - 9) / 12) + np.random.normal(0, 0.5)
        cooling_demand_factor = max(0.0, (ambient_temp - 24.0) / 14.0)
        
        for zone_id in zones:
            z_meta = zone_lookup[zone_id]
            base_kw = z_meta["baseline_power_kw"]
            base_lph = z_meta["baseline_water_lph"]
            day_occ = z_meta["occupancy_day"]
            night_occ = z_meta["occupancy_night"]
            
            # 1. Occupancy Profile
            if zone_id == "OPD":
                # OPD runs 8:00 to 16:00, closed Sundays
                if is_sunday:
                    occ = int(night_occ + np.random.poisson(2))
                elif 8 <= hour <= 16:
                    peak_ratio = np.sin(np.pi * (hour - 8) / 8)
                    occ = int(night_occ + (day_occ - night_occ) * (0.4 + 0.6 * peak_ratio) + np.random.normal(0, 15))
                else:
                    occ = int(night_occ + np.random.poisson(3))
            elif zone_id in ["Wards", "ICU"]:
                # Inpatient 24x7 steady, slight visitor peak 16:00-19:00
                visitor_factor = 1.3 if (16 <= hour <= 19 and zone_id == "Wards") else 1.0
                occ = int(night_occ + (day_occ - night_occ) * 0.8 * visitor_factor + np.random.normal(0, 5))
            elif zone_id == "Hostel":
                # Morning (6-9 AM) and evening (18-23 PM) peak
                if (6 <= hour <= 9) or (18 <= hour <= 23):
                    occ = int(night_occ * 0.9 + np.random.normal(0, 8))
                else:
                    occ = int(day_occ + np.random.normal(0, 5))
            elif zone_id == "Canteen":
                # Meal times: 8-9 Breakfast, 12-14 Lunch, 19-21 Dinner
                if 12 <= hour <= 14:
                    occ = int(day_occ * 0.95 + np.random.normal(0, 10))
                elif (8 <= hour <= 9) or (19 <= hour <= 21):
                    occ = int(day_occ * 0.6 + np.random.normal(0, 8))
                else:
                    occ = int(night_occ + np.random.poisson(4))
            elif zone_id == "Parking":
                if 8 <= hour <= 17 and not is_sunday:
                    occ = int(day_occ * 0.8 + np.random.normal(0, 10))
                else:
                    occ = int(night_occ + np.random.normal(0, 3))
            elif zone_id == "Generator":
                occ = int(night_occ + np.random.poisson(1))
            else: # Labs
                if 8 <= hour <= 18 and not is_sunday:
                    occ = int(day_occ * 0.85 + np.random.normal(0, 8))
                else:
                    occ = int(night_occ + np.random.poisson(2))
            
            occ = max(1, occ)
            
            # 2. Energy Load (kW)
            # Base electrical load + AC load proportional to temperature & occupancy
            hvac_load = base_kw * 0.5 * cooling_demand_factor * (occ / max(1, day_occ))
            equipment_load = base_kw * 0.4 * (occ / max(1, day_occ))
            lighting_load = base_kw * 0.1 * (1.2 if (18 <= hour or hour <= 6) else 0.4)
            power_kw = (base_kw * 0.25) + hvac_load + equipment_load + lighting_load + np.random.normal(0, 1.2)
            power_kw = max(2.0, power_kw)
            
            # 3. Water Consumption (Liters/hour)
            # Baseline + per capita usage (sleeping occupants consume far less at 1-5 AM)
            active_water_mult = 0.12 if (1 <= hour <= 5) else 1.0
            water_lph = (base_lph * (0.05 if (1 <= hour <= 5) else 0.15)) + (occ * 8.5 * active_water_mult) + np.random.normal(0, 15.0)
            water_lph = max(10.0, water_lph)
            
            # 4. Air Quality (PM2.5, PM10)
            # Background regional AQI: morning and evening traffic spikes
            base_pm25 = 42.0 + 15.0 * np.sin(np.pi * (hour - 6) / 12)**2 + np.random.normal(0, 4.0)
            base_pm10 = base_pm25 * 1.8 + np.random.normal(0, 8.0)
            
            records.append({
                "timestamp": t,
                "zone_id": zone_id,
                "occupancy": occ,
                "ambient_temp_c": round(ambient_temp, 1),
                "power_kw": round(power_kw, 2),
                "energy_kwh": round(power_kw * 1.0, 2), # 1 hour interval
                "water_lph": round(water_lph, 1),
                "water_pressure_psi": round(45.0 - (water_lph / 100.0) * 0.4 + np.random.normal(0, 1.0), 1),
                "pm25": round(max(10.0, base_pm25), 1),
                "pm10": round(max(20.0, base_pm10), 1),
                "co2_ppm": round(410 + (occ * 1.5) + np.random.normal(0, 10), 0)
            })

    df = pd.DataFrame(records)
    
    # 5. Fault Injection (Known real-world faults requested by prompt)
    if inject_faults:
        # FAULT 1: Night-time water leak in Hostel Pipeline 2 (last 4 days, 2 AM - 4 AM and persistent overnight)
        last_4_days_mask = (df["timestamp"] >= (end_date - timedelta(days=4))) & (df["zone_id"] == "Hostel")
        night_hours_mask = df["timestamp"].dt.hour.isin([1, 2, 3, 4, 5])
        hostel_leak_mask = last_4_days_mask & night_hours_mask
        # Baseline night usage in Hostel is ~150-250 L/hr. Add 650 L/hr leak! (4x normal)
        df.loc[hostel_leak_mask, "water_lph"] += 680.0
        df.loc[hostel_leak_mask, "water_pressure_psi"] -= 12.0 # pressure drop
        
        # FAULT 2: AC left on overnight in Block C (OPD) (11 PM onward, near-zero occupancy in last 3 days)
        last_3_days_mask = (df["timestamp"] >= (end_date - timedelta(days=3))) & (df["zone_id"] == "OPD")
        ac_night_hours = df["timestamp"].dt.hour.isin([23, 0, 1, 2, 3, 4, 5])
        opd_ac_fault_mask = last_3_days_mask & ac_night_hours
        # Regular night power is ~10-14 kW. Jump to 48-52 kW!
        df.loc[opd_ac_fault_mask, "power_kw"] += 38.5
        df.loc[opd_ac_fault_mask, "energy_kwh"] += 38.5
        
        # FAULT 4: PM2.5 surge near the generator and parking (last 4-6 hours)
        recent_hours_mask = (df["timestamp"] >= (end_date - timedelta(hours=5)))
        gen_surge_mask = recent_hours_mask & (df["zone_id"] == "Generator")
        park_surge_mask = recent_hours_mask & (df["zone_id"] == "Parking")
        df.loc[gen_surge_mask, "pm25"] += 115.0 # Elevate to ~160-180 ug/m3
        df.loc[gen_surge_mask, "pm10"] += 190.0
        df.loc[park_surge_mask, "pm25"] += 75.0
        df.loc[park_surge_mask, "pm10"] += 120.0

    return df

def generate_waste_bins_data(end_date=None, config=None, inject_faults=True):
    """
    Generate waste tracking data for hospital bins:
    Tracks General vs Bio-Medical Waste (Yellow, Red, Blue, White),
    current fill level %, fill rate (%/hr), hours since last cleared.
    """
    if config is None:
        config = load_config()
    if end_date is None:
        end_date = datetime.now()

    bins_config = config.get("waste_bins", [])
    waste_records = []
    
    # Bins 4, 7, 9 are near canteen (injected fault: rapid fill, near overflow by 2 PM)
    for b in bins_config:
        b_id = b["id"]
        zone = b["zone"]
        w_type = b["type"]
        color = b["color"]
        
        # Default baseline fill rates
        if zone == "Canteen":
            base_fill_rate = 6.5 # % per hour
            hours_since_clear = 8.5
            current_fill = min(100.0, base_fill_rate * hours_since_clear)
        elif zone in ["Wards", "ICU"]:
            base_fill_rate = 3.2
            hours_since_clear = 14.0
            current_fill = min(100.0, base_fill_rate * hours_since_clear)
        elif zone == "Labs":
            base_fill_rate = 2.4
            hours_since_clear = 18.0
            current_fill = min(100.0, base_fill_rate * hours_since_clear)
        else:
            base_fill_rate = 2.0
            hours_since_clear = 16.0
            current_fill = min(100.0, base_fill_rate * hours_since_clear)
            
        # Injected Fault: Bins 4, 7, 9 near Canteen have massive lunch rush spike
        if inject_faults and b_id in [4, 7, 9]:
            base_fill_rate = 14.2 # very fast fill rate
            hours_since_clear = 6.5
            current_fill = 93.0 if b_id == 4 else (88.5 if b_id == 7 else 91.0)
            
        # BMW Compliance check: Bio-Medical waste must be cleared within 48 hours
        # Injected compliance test: Bin 8 in Labs was cleared 42 hours ago (nearing 48h deadline)
        if b_id == 8:
            hours_since_clear = 44.0
            current_fill = 68.0

        waste_records.append({
            "bin_id": b_id,
            "zone": zone,
            "waste_category": w_type,
            "bin_color": color,
            "fill_percentage": round(current_fill, 1),
            "fill_rate_pct_per_hr": round(base_fill_rate, 2),
            "hours_since_emptied": round(hours_since_clear, 1),
            "capacity_liters": 240,
            "weight_kg_est": round(current_fill * 240 * (0.35 if w_type == 'General' else 0.45) / 100, 1),
            "timestamp": end_date
        })
        
    return pd.DataFrame(waste_records)

def save_synthetic_dataset(out_dir="data"):
    """Generate and save standard CSV files for offline testing and pipeline loading."""
    os.makedirs(out_dir, exist_ok=True)
    df_facility = generate_facility_data(days=30)
    facility_path = os.path.join(out_dir, "facility_hourly.csv")
    df_facility.to_csv(facility_path, index=False)
    
    df_waste = generate_waste_bins_data()
    waste_path = os.path.join(out_dir, "waste_bins_status.csv")
    df_waste.to_csv(waste_path, index=False)
    
    print(f"Dataset generated successfully:\n - {facility_path} ({len(df_facility)} rows)\n - {waste_path} ({len(df_waste)} bins)")
    return facility_path, waste_path

if __name__ == "__main__":
    save_synthetic_dataset()
