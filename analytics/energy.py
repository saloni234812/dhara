"""
Dhara - Energy Analytics Module
- 24-48h Load Forecasting with Confidence Bands (Prophet with graceful fallback to Linear/Ridge regression)
- Isolation Forest Anomaly Detection
- Plain-language explanation for every flagged anomaly (e.g., "3.2 sigma above normal night baseline")
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import Ridge
from datetime import timedelta

class EnergyAnalytics:
    def __init__(self, config=None):
        self.config = config or {}
        self.tariff = self.config.get("tariffs_and_factors", {}).get("electricity_tariff_inr_kwh", 8.50)
        self.emission_factor = self.config.get("tariffs_and_factors", {}).get("emission_factor_kg_co2_kwh", 0.75)

    def detect_anomalies(self, df_facility, zone_id=None):
        """
        Runs Isolation Forest anomaly detection on power consumption.
        Generates statistical Z-score and human-readable explanation for each flagged instance.
        """
        df = df_facility.copy()
        if zone_id:
            df = df[df["zone_id"] == zone_id].copy()

        results = []
        for zid, group in df.groupby("zone_id"):
            group = group.sort_values(by="timestamp").copy()
            if len(group) < 24:
                continue

            # Feature matrix for Isolation Forest
            features = pd.DataFrame({
                "power_kw": group["power_kw"],
                "hour": group["hour"],
                "is_weekend": group["is_weekend"].astype(int),
                "ambient_temp_c": group["ambient_temp_c"],
                "occupancy": group["occupancy"]
            }).bfill().fillna(0)

            # Fit Isolation Forest
            iso = IsolationForest(
                contamination=self.config.get("thresholds", {}).get("energy", {}).get("anomaly_contamination", 0.05),
                random_state=42
            )
            group["iso_score"] = iso.fit_predict(features) # -1 is anomaly, 1 is normal
            group["anomaly_score"] = iso.decision_function(features)

            # Calculate normal hourly expected baseline (median power by hour of day and weekend)
            baseline_map = group.groupby(["hour", "is_weekend"])["power_kw"].median().to_dict()
            group["expected_power_kw"] = group.apply(lambda r: baseline_map.get((r["hour"], r["is_weekend"]), r["power_kw"]), axis=1)
            group["excess_power_kw"] = np.maximum(0.0, group["power_kw"] - group["expected_power_kw"])

            # Explain each anomaly in plain language
            def explain_row(r):
                if r["iso_score"] == -1:
                    excess = r["power_kw"] - r["expected_power_kw"]
                    zscore = r.get("power_zscore_24h", 0.0)
                    if r["is_night"] and r["occupancy"] <= 15 and excess > 15:
                        return f"AC / heavy equipment running during unoccupied night hours ({round(r['power_kw'], 1)} kW vs {round(r['expected_power_kw'], 1)} kW baseline, {round(zscore, 1)}σ elevated)."
                    elif excess > 0:
                        return f"Unusual power spike of {round(excess, 1)} kW ({round(zscore, 1)}σ above expected hourly profile)."
                    else:
                        return f"Abnormal load drop ({round(abs(excess), 1)} kW below typical operation)."
                return "Normal"

            group["anomaly_explanation"] = group.apply(explain_row, axis=1)
            group["is_anomaly"] = group["iso_score"] == -1
            results.append(group)

        if not results:
            return df
        return pd.concat(results, axis=0).sort_values(by=["timestamp", "zone_id"]).reset_index(drop=True)

    def forecast_energy(self, df_facility, zone_id=None, horizon_hours=48):
        """
        Generates 24-48h forecast with upper and lower confidence intervals.
        Attempts to use Prophet; if unavailable, uses high-fidelity Fourier-Ridge regression.
        """
        df = df_facility.copy()
        if zone_id:
            df = df[df["zone_id"] == zone_id].copy()
            
        # Aggregate to campus or zone level
        ts_df = df.groupby("timestamp")["power_kw"].sum().reset_index()
        ts_df = ts_df.sort_values(by="timestamp").reset_index(drop=True)
        
        last_time = ts_df["timestamp"].max()
        future_times = [last_time + timedelta(hours=i) for i in range(1, horizon_hours + 1)]
        
        # Try Prophet
        prophet_success = False
        forecast_df = None
        try:
            from prophet import Prophet
            m = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                interval_width=0.90
            )
            prophet_input = ts_df.rename(columns={"timestamp": "ds", "power_kw": "y"})
            m.fit(prophet_input)
            future = m.make_future_dataframe(periods=horizon_hours, freq="h")
            forecast = m.predict(future)
            
            future_forecast = forecast[forecast["ds"] > last_time].copy()
            forecast_df = pd.DataFrame({
                "timestamp": future_forecast["ds"],
                "yhat": future_forecast["yhat"].clip(lower=0),
                "yhat_lower": future_forecast["yhat_lower"].clip(lower=0),
                "yhat_upper": future_forecast["yhat_upper"].clip(lower=0),
                "model_used": "Prophet"
            })
            prophet_success = True
        except Exception:
            prophet_success = False

        if not prophet_success:
            # High-accuracy fallback: Ridge regression with Fourier terms & calendar features
            ts_df["hour"] = ts_df["timestamp"].dt.hour
            ts_df["dayofweek"] = ts_df["timestamp"].dt.dayofweek
            
            # Fourier harmonics for 24h diurnal cycle
            ts_df["sin_24"] = np.sin(2 * np.pi * ts_df["hour"] / 24)
            ts_df["cos_24"] = np.cos(2 * np.pi * ts_df["hour"] / 24)
            ts_df["sin_168"] = np.sin(2 * np.pi * (ts_df["dayofweek"] * 24 + ts_df["hour"]) / 168)
            ts_df["cos_168"] = np.cos(2 * np.pi * (ts_df["dayofweek"] * 24 + ts_df["hour"]) / 168)
            
            X_cols = ["hour", "dayofweek", "sin_24", "cos_24", "sin_168", "cos_168"]
            X = ts_df[X_cols]
            y = ts_df["power_kw"]
            
            model = Ridge(alpha=1.0)
            model.fit(X, y)
            
            # Residual standard deviation for 90% confidence bands (1.645 * std)
            residuals = y - model.predict(X)
            sigma = np.std(residuals)
            
            # Generate future features
            future_df = pd.DataFrame({"timestamp": future_times})
            future_df["hour"] = future_df["timestamp"].dt.hour
            future_df["dayofweek"] = future_df["timestamp"].dt.dayofweek
            future_df["sin_24"] = np.sin(2 * np.pi * future_df["hour"] / 24)
            future_df["cos_24"] = np.cos(2 * np.pi * future_df["hour"] / 24)
            future_df["sin_168"] = np.sin(2 * np.pi * (future_df["dayofweek"] * 24 + future_df["hour"]) / 168)
            future_df["cos_168"] = np.cos(2 * np.pi * (future_df["dayofweek"] * 24 + future_df["hour"]) / 168)
            
            preds = model.predict(future_df[X_cols])
            forecast_df = pd.DataFrame({
                "timestamp": future_times,
                "yhat": np.maximum(5.0, preds),
                "yhat_lower": np.maximum(0.0, preds - 1.645 * sigma),
                "yhat_upper": preds + 1.645 * sigma,
                "model_used": "Fourier-Ridge Regression (Prophet Fallback)"
            })

        return forecast_df

    def calculate_night_waste_savings(self, df_with_anomalies, zone_id="OPD"):
        """
        Calculates monthly potential savings from unoptimized night cooling / equipment.
        Visible Formula: load kW x excess hours x 30 days x tariff = ₹ Savings.
        """
        zone_df = df_with_anomalies[df_with_anomalies["zone_id"] == zone_id]
        night_anomalies = zone_df[(zone_df["is_night"]) & (zone_df["is_anomaly"])]
        
        if len(night_anomalies) == 0:
            avg_excess_kw = 0.0
            excess_hours_per_night = 0.0
        else:
            avg_excess_kw = night_anomalies["excess_power_kw"].mean()
            # Estimate hours per night
            excess_hours_per_night = min(7.0, len(night_anomalies) / max(1, len(zone_df["timestamp"].dt.date.unique())))
            
        monthly_kwh_wasted = avg_excess_kw * excess_hours_per_night * 30.0
        monthly_savings_inr = monthly_kwh_wasted * self.tariff
        monthly_co2_avoided_kg = monthly_kwh_wasted * self.emission_factor
        
        return {
            "avg_excess_kw": round(avg_excess_kw, 1),
            "excess_hours_per_night": round(excess_hours_per_night, 1),
            "monthly_kwh_saved": round(monthly_kwh_wasted, 1),
            "monthly_savings_inr": round(monthly_savings_inr, 0),
            "monthly_co2_kg": round(monthly_co2_avoided_kg, 1),
            "formula_str": f"{round(avg_excess_kw, 1)} kW × {round(excess_hours_per_night, 1)} hrs/night × 30 days × ₹{self.tariff}/kWh"
        }
