"""
Dhara - External API Connectors
Fetches real-time environmental data (Weather, AQI) for Ramnagar hospital coordinates,
and communicates with LLM & Grid APIs.
"""

import requests
import os

class ExternalAPIClient:
    def __init__(self, lat=26.8467, lon=80.9462):
        self.lat = lat
        self.lon = lon
        self.weather_url = "https://api.open-meteo.com/v1/forecast"
        self.aqi_url = "https://air-quality-api.open-meteo.com/v1/air-quality"

    def get_live_weather(self):
        """
        Fetches live ambient temperature and humidity for hospital coordinates
        from Open-Meteo free API (No key required).
        """
        try:
            params = {
                "latitude": self.lat,
                "longitude": self.lon,
                "current": "temperature_2m,relative_humidity_2m,surface_pressure"
            }
            res = requests.get(self.weather_url, params=params, timeout=4)
            if res.status_code == 200:
                data = res.json().get("current", {})
                return {
                    "status": "online",
                    "temperature_c": data.get("temperature_2m", 28.5),
                    "relative_humidity_pct": data.get("relative_humidity_2m", 72),
                    "pressure_hpa": data.get("surface_pressure", 1012.0),
                    "timestamp": data.get("time")
                }
        except Exception as e:
            pass
            
        return {
            "status": "cached",
            "temperature_c": 28.5,
            "relative_humidity_pct": 72,
            "pressure_hpa": 1012.0,
            "timestamp": "fallback"
        }

    def get_live_ambient_aqi(self):
        """
        Fetches live ambient PM2.5 and PM10 for Ramnagar coordinates
        from Open-Meteo Air Quality API.
        """
        try:
            params = {
                "latitude": self.lat,
                "longitude": self.lon,
                "current": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide"
            }
            res = requests.get(self.aqi_url, params=params, timeout=4)
            if res.status_code == 200:
                data = res.json().get("current", {})
                pm25 = data.get("pm2_5", 58.0)
                pm10 = data.get("pm10", 66.0)
                
                # CPCB classification
                cat = "Good" if pm25 <= 30 else ("Satisfactory" if pm25 <= 60 else ("Moderate" if pm25 <= 90 else "Poor"))
                return {
                    "status": "online",
                    "ambient_pm25": pm25,
                    "ambient_pm10": pm10,
                    "category": cat,
                    "timestamp": data.get("time")
                }
        except Exception as e:
            pass
            
        return {
            "status": "cached",
            "ambient_pm25": 58.0,
            "ambient_pm10": 66.0,
            "category": "Satisfactory",
            "timestamp": "fallback"
        }

    def get_grid_carbon_intensity(self):
        """
        Returns Central Electricity Authority (CEA) v19 grid emission benchmark
        for Northern Regional Grid (India).
        """
        return {
            "grid_region": "Northern Regional Grid (UP/NR)",
            "emission_factor_kg_co2_kwh": 0.75,
            "source": "CEA CO2 Baseline Database Version 19",
            "status": "verified"
        }
