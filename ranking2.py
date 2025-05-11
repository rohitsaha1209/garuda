from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Union, List, Dict, Any, Optional
from math import radians, cos, sin, asin, sqrt
import requests
import time
import numpy as np

class BidRankingSystem:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.geocode_cache = {}

    def _preprocess_trade(self, trade: Union[str, List[str]]) -> str:
        return " ".join(trade).lower() if isinstance(trade, list) else trade.lower()

    def score_trade_match(self, bid_trade: Union[str, List[str]], preferred_trade: Union[str, List[str]]) -> float:
        bid_text = self._preprocess_trade(bid_trade)
        preferred_text = self._preprocess_trade(preferred_trade)
        if bid_text == preferred_text:
            return 1.0
        embeddings = self.model.encode([bid_text, preferred_text])
        return float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        dlon, dlat = lon2 - lon1, lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        return 6371 * c  # km

    def geocode_location(self, location: str) -> Optional[Dict[str, float]]:
        if location in self.geocode_cache:
            return self.geocode_cache[location]

        time.sleep(1)  # Respect API rate limits
        url = f"https://nominatim.openstreetmap.org/search?format=json&q={requests.utils.quote(location)}"
        headers = {"User-Agent": "BidRankingSystem/1.0"}
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            data = response.json()
            if data:
                coords = {"lat": float(data[0]["lat"]), "lon": float(data[0]["lon"])}
                self.geocode_cache[location] = coords
                return coords

        return None

    def get_driving_distance(self, start: Dict[str, float], end: Dict[str, float]) -> float:
        url = f"http://router.project-osrm.org/route/v1/driving/{start['lon']},{start['lat']};{end['lon']},{end['lat']}?overview=false"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if data.get("code") == "Ok" and data["routes"]:
                return data["routes"][0]["distance"] / 1000
        return self.haversine_distance(start["lat"], start["lon"], end["lat"], end["lon"])

    def calculate_location_score(self, user_loc: str, bid_loc: str, max_dist=500, use_driving=False) -> float:
        user_coords, bid_coords = self.geocode_location(user_loc), self.geocode_location(bid_loc)
        if not user_coords or not bid_coords:
            return 0.1
        try:
            dist = self.get_driving_distance(user_coords, bid_coords) if use_driving else \
                   self.haversine_distance(user_coords["lat"], user_coords["lon"], bid_coords["lat"], bid_coords["lon"])
        except Exception:
            dist = self.haversine_distance(user_coords["lat"], user_coords["lon"], bid_coords["lat"], bid_coords["lon"])
        return max(0, 1 - (dist / max_dist))

    def preprocess_bids_with_location(self, bids: List[Dict[str, Any]], user_loc: str, use_driving=False):
        for bid in bids:
            bid['location_score'] = self.calculate_location_score(user_loc, bid.get('location', ''), use_driving=use_driving)
        return bids

    def calculate_weighted_scores(self, bids: List[Dict[str, Any]], weights: Dict[str, float], preferred_trades=None, user_loc=None):
        if user_loc and weights.get('location', 0) > 0:
            self.preprocess_bids_with_location(bids, user_loc)

        for bid in bids:
            score_sum = 0
            bid['parameter_scores'] = {}
            bid['weighted_parameter_scores'] = {}

            if 'trades' in bid and preferred_trades and 'trades' in weights:
                trade_score = self.score_trade_match(bid['trades'], preferred_trades)
                bid['parameter_scores']['trades'] = trade_score
                bid['weighted_parameter_scores']['trades'] = trade_score * weights['trades']
                score_sum += trade_score * weights['trades']

            if 'location_score' in bid and 'location' in weights:
                loc_score = bid['location_score']
                bid['parameter_scores']['location'] = loc_score
                bid['weighted_parameter_scores']['location'] = loc_score * weights['location']
                score_sum += loc_score * weights['location']

            for param, weight in weights.items():
                if param in ['trades', 'location']: continue
                if param in bid and bid[param] in ['trades', 'location']:
                    score = bid[param]
                    bid['parameter_scores'][param] = score
                    bid['weighted_parameter_scores'][param] = score * weight
                    score_sum += score * weight

            bid['total_weighted_score'] = score_sum

        bids.sort(key=lambda b: b['total_weighted_score'], reverse=True)
        for i, bid in enumerate(bids):
            bid['rank'] = i + 1

        return bids

def demo(outputs, weights, filters):
    ranker = BidRankingSystem()
    preferred_trades = filters['trades']
    user_loc = filters['property_address']
    ranked_bids = ranker.calculate_weighted_scores(outputs, weights, preferred_trades=preferred_trades, user_loc=user_loc)
    return ranked_bids

