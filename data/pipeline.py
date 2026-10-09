"""
Dhara - Data Ingestion and Feature Engineering Pipeline
Handles CSV/API ingestion, cleaning, resampling, rolling baselines, and domain features.
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

class DharaDataPipeline:
    def __init__(self, config=None):
        self.config = config or load_config()
        self.zones = [z["id"] for z in self.config.get("zones", [])]
        self.zone_map = {z["id"]: z for z in self.config.get("zones", [])}

    def load_facility_data(self, source=None):
        """
        Load facility time series data from:
        - Uploaded CSV path or file buffer
        - Default CSV in data/facility_hourly.csv
        - On-the-fly synthetic generator if file doesn't exist
        """
        df = None
        if source is not None:
            if isinstance(source, (str, os.PathLike)) and os.path.exists(source):
                df = pd.read_csv(source)
            elif hasattr(source, "read"): # Streamlit UploadedFile or buffer
                df = pd.read_csv(source)
        
        default_csv = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "facility_hourly.csv")
        if df is None and os.path.exists(default_csv):
            df = pd.read_csv(default_csv)
            
        if df is None:
            from data.generator import generate_facility_data
            df = generate_facility_data(days=30, config=self.config)

        # Standardize columns and datatypes
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        else:
            raise ValueError("Dataframe must contain 'timestamp' column.")

        # Sort chronologically
        df = df.sort_values(by=["timestamp", "zone_id"]).reset_index(drop=True)
        return self._engineer_facility_features(df)

    def _engineer_facility_features(self, df):
        """Add temporal, statistical, and domain features for energy, water, and air."""
        df = df.copy()
        
        # 1. Temporal features
        df["hour"] = df["timestamp"].dt.hour
        df["day_name"] = df["timestamp"].dt.day_name()
        df["dayofweek"] = df["timestamp"].dt.dayofweek
        df["is_weekend"] = df["dayofweek"].isin([5, 6])
        df["is_night"] = df["hour"].isin([22, 23, 0, 1, 2, 3, 4, 5])
        df["is_night_flow_window"] = df["hour"].isin(self.config["thresholds"]["water"]["night_flow_window_hours"])

        # 2. Zone-specific rolling baselines and Z-scores
        enhanced_dfs = []
        for zone_id, group in df.groupby("zone_id"):
            group = group.sort_values(by="timestamp").copy()
            
            # Energy rolling statistics (24h and 7d)
            group["power_roll_mean_24h"] = group["power_kw"].rolling(window=24, min_periods=4).mean()
            group["power_roll_std_24h"] = group["power_kw"].rolling(window=24, min_periods=4).std().fillna(1.0)
            group["power_zscore_24h"] = (group["power_kw"] - group["power_roll_mean_24h"]) / (group["power_roll_std_24h"] + 1e-4)

            # Water rolling statistics
            group["water_roll_mean_24h"] = group["water_lph"].rolling(window=24, min_periods=4).mean()
            group["water_roll_std_24h"] = group["water_lph"].rolling(window=24, min_periods=4).std().fillna(5.0)
            group["water_zscore_24h"] = (group["water_lph"] - group["water_roll_mean_24h"]) / (group["water_roll_std_24h"] + 1e-4)
            
            # Specific night baseline for water (avg flow between 2 AM - 4 AM)
            night_flow_avg = group[group["is_night_flow_window"]]["water_lph"].mean()
            group["water_night_baseline"] = night_flow_avg if not np.isnan(night_flow_avg) else group["water_lph"].quantile(0.1)

            # Air quality categories (CPCB guidelines)
            def classify_pm25(v):
                if v <= 30: return "Good"
                if v <= 60: return "Satisfactory"
                if v <= 90: return "Moderate"
                if v <= 120: return "Poor"
                if v <= 250: return "Very Poor"
                return "Severe"
            
            group["pm25_category"] = group["pm25"].apply(classify_pm25)
            
            # Air hotspot feature: rolling 3-hour mean PM2.5
            group["pm25_roll_mean_3h"] = group["pm25"].rolling(window=3, min_periods=1).mean()
            
            enhanced_dfs.append(group)
            
        full_df = pd.concat(enhanced_dfs, axis=0).sort_values(by=["timestamp", "zone_id"]).reset_index(drop=True)
        return full_df

    def load_waste_data(self, source=None):
        """
        Load waste bin telemetry and compute:
        - fill rate
        - hours to overflow (100% capacity)
        - predicted overflow timestamp
        - compliance flag (Bio-medical waste > 48h)
        """
        df = None
        if source is not None:
            if isinstance(source, (str, os.PathLike)) and os.path.exists(source):
                df = pd.read_csv(source)
            elif hasattr(source, "read"):
                df = pd.read_csv(source)
                
        default_csv = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "waste_bins_status.csv")
        if df is None and os.path.exists(default_csv):
            df = pd.read_csv(default_csv)
            
        if df is None:
            from data.generator import generate_waste_bins_data
            df = generate_waste_bins_data(config=self.config)

        df = df.copy()
        now = datetime.now()
        
        # Calculate time-to-overflow
        # Hours remaining = (100 - current_fill) / fill_rate_pct_per_hr
        def calc_hours_to_overflow(row):
            remaining_cap = max(0.0, 100.0 - row["fill_percentage"])
            rate = row["fill_rate_pct_per_hr"]
            if rate <= 0.05:
                return 999.0
            return round(remaining_cap / rate, 1)

        df["hours_to_overflow"] = df.apply(calc_hours_to_overflow, axis=1)
        df["predicted_overflow_time"] = df["hours_to_overflow"].apply(lambda h: (now + timedelta(hours=min(h, 72))).strftime("%I:%M %p"))

        # Bio-Medical Waste (BMW) Compliance
        bmw_limit = self.config["thresholds"]["waste"]["bmw_max_storage_hours"]
        df["bmw_compliance_breach"] = (df["waste_category"] == "Biomedical") & (df["hours_since_emptied"] >= (bmw_limit - 4))
        df["is_critical_overflow"] = df["fill_percentage"] >= self.config["thresholds"]["waste"]["overflow_critical_pct"]

        return df

    def get_latest_snapshot(self, df_facility):
        """Return the most recent single hourly slice across all zones."""
        latest_time = df_facility["timestamp"].max()
        return df_facility[df_facility["timestamp"] == latest_time].copy()
