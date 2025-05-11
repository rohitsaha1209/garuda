import requests
import time
from typing import Optional, Dict, Any
from config import Config
class RotatingGeocoder:
    def __init__(self):
        self.call_count = 0

    def geocode_with_google(self, location_str: str) -> Optional[Dict[str, Any]]:
        url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {
            "address": location_str,
            "key": Config.GOOGLE_MAPS_API_KEY
        }
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("results"):
                    result = data["results"][0]
                    location = result["geometry"]["location"]
                    return {
                        "lat": location["lat"],
                        "lon": location["lng"],
                        "display_name": result["formatted_address"],
                        "source": "google"
                    }
        except Exception as e:
            print(f"[Google Error] {e}")
        return None

    def geocode_with_nominatim(self, location_str: str) -> Optional[Dict[str, Any]]:
        url = f"https://nominatim.openstreetmap.org/search?format=json&q={requests.utils.quote(location_str)}"
        headers = {"User-Agent": "BidRankingSystem/1.0"}
        try:
            time.sleep(1)  # Required for Nominatim
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data:
                    return {
                        "lat": float(data[0]["lat"]),
                        "lon": float(data[0]["lon"]),
                        "display_name": data[0]["display_name"],
                        "source": "nominatim"
                    }
        except Exception as e:
            print(f"[Nominatim Error] {e}")
        return None

    def geocode(self, location_str: str) -> Optional[Dict[str, Any]]:
        # Rotate: Google, Nominatim, Google, Google, Nominatim, ...
        api_index = self.call_count % 5
        self.call_count += 1

        if api_index in [0, 2, 3]:  # Google for calls 0, 2, 3
            print(f"Using Google Maps (call #{self.call_count})")
            return self.geocode_with_google(location_str)
        else:  # Nominatim for calls 1, 4
            print(f"Using Nominatim (call #{self.call_count})")
            return self.geocode_with_nominatim(location_str)