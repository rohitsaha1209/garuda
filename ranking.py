from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Union, List, Dict, Any, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
import requests
import time
from math import radians, cos, sin, asin, sqrt
import pandas as pd

class BidRankingSystem:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the bid ranking system with a vector embedding model.
        
        Args:
            model_name: The name of the SentenceTransformer model to use
        """
        self.model = SentenceTransformer(model_name)
    
    def score_trade_match(self, bid_trade: Union[str, List[str]], 
                         preferred_trade: Union[str, List[str]]) -> float:
        """
        Calculate similarity score between bid trades and preferred trades.
        
        Args:
            bid_trade: Trade(s) from a bid (string or list of strings)
            preferred_trade: User's preferred trade(s) (string or list of strings)
            
        Returns:
            Similarity score between 0 and 1, where 1 is a perfect match
        """
        # Preprocess both inputs
        bid_text = self._preprocess_trade(bid_trade)
        preferred_text = self._preprocess_trade(preferred_trade)
        
        # Handle exact match case for efficiency
        if bid_text == preferred_text:
            return 1.0
            
        # Get embeddings for both texts
        bid_embedding = self.model.encode([bid_text])[0]
        preferred_embedding = self.model.encode([preferred_text])[0]
        
        # Calculate cosine similarity
        similarity = cosine_similarity(
            bid_embedding.reshape(1, -1),
            preferred_embedding.reshape(1, -1)
        )[0][0]
        
        return float(similarity)
    
    def _preprocess_trade(self, trade: Union[str, List[str]]) -> str:
        """
        Preprocess a trade or list of trades into a format suitable for embedding.
        
        Args:
            trade: Either a single trade string or a list of trade strings
            
        Returns:
            A preprocessed string representation
        """
        if isinstance(trade, list):
            # Join the list of trades into a single string
            return " ".join(trade).lower()
        else:
            # Just lowercase the single trade
            return str(trade).lower()
    
    # Location scoring methods based on provided code
    def haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate the great circle distance between two points 
        on the earth (specified in decimal degrees)
        """
        # Convert decimal degrees to radians
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

        # Haversine formula
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a)) 
        r = 6371  # Radius of earth in kilometers
        return c * r
    
    def geocode_location(self, location_str: str) -> Optional[Dict[str, Any]]:
        """
        Convert a location string to latitude and longitude using OpenStreetMap's Nominatim API
        """
        encoded_location = requests.utils.quote(location_str)
        url = f"https://nominatim.openstreetmap.org/search?format=json&q={encoded_location}"
        
        headers = {
            "User-Agent": "BidRankingSystem/1.0"  # Required by Nominatim
        }
        
        # Respect rate limits (1 request per second)
        time.sleep(1)
        
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            if data and len(data) > 0:
                return {
                    "lat": float(data[0]["lat"]),
                    "lon": float(data[0]["lon"]),
                    "display_name": data[0]["display_name"]
                }
        
        return None
    
    def get_driving_distance(self, start_coords: Dict[str, float], end_coords: Dict[str, float]) -> float:
        """
        Get driving distance between two coordinates using OSRM API
        """
        url = f"http://router.project-osrm.org/route/v1/driving/{start_coords['lon']},{start_coords['lat']};{end_coords['lon']},{end_coords['lat']}?overview=false"
        
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            if data["code"] == "Ok" and data["routes"] and len(data["routes"]) > 0:
                # Return distance in kilometers
                return data["routes"][0]["distance"] / 1000
        
        # If OSRM fails, fall back to haversine
        return self.haversine_distance(
            start_coords["lat"], start_coords["lon"],
            end_coords["lat"], end_coords["lon"]
        )
    
    def calculate_location_score(self, 
                               user_location: str, 
                               bid_location: str, 
                               max_distance: float = 500,
                               use_driving_distance: bool = False) -> float:
        """
        Calculate a location score (0-1) based on proximity.
        
        Args:
            user_location: String of user's location
            bid_location: String of bid's location
            max_distance: Maximum distance in km (scores will be 0 beyond this)
            use_driving_distance: Whether to use driving distance vs straight-line
            
        Returns:
            A score between 0 and 1, where 1 means same location and 
            0 means distance >= max_distance
        """
        # Geocode both locations
        user_coords = self.geocode_location(user_location)
        bid_coords = self.geocode_location(bid_location)
        
        # If geocoding fails, return a low score
        if not user_coords or not bid_coords:
            return 0.1  # Return a small non-zero score as fallback
        
        # Calculate distance
        if use_driving_distance:
            try:
                distance = self.get_driving_distance(user_coords, bid_coords)
            except Exception:
                # Fall back to straight-line distance
                distance = self.haversine_distance(
                    user_coords["lat"], user_coords["lon"],
                    bid_coords["lat"], bid_coords["lon"]
                )
        else:
            distance = self.haversine_distance(
                user_coords["lat"], user_coords["lon"],
                bid_coords["lat"], bid_coords["lon"]
            )
        
        # Calculate score: 1 for distance=0, 0 for distance>=max_distance
        # Linear interpolation for distances in between
        score = max(0, 1 - (distance / max_distance))
        
        return float(score)
    
    def preprocess_bids_with_location(self, 
                                     bids: List[Dict[str, Any]], 
                                     user_location: str,
                                     use_driving_distance: bool = False) -> List[Dict[str, Any]]:
        """
        Preprocess bids to add location scores based on user's location
        
        Args:
            bids: List of bid dictionaries, each containing a 'location' field
            user_location: String of user's location
            use_driving_distance: Whether to use driving distance
            
        Returns:
            Bids with added location_score field
        """
        processed_bids = []
        
        for bid in bids:
            processed_bid = bid.copy()
            
            # Calculate location score if location is specified
            if 'location' in bid:
                processed_bid['location_score'] = self.calculate_location_score(
                    user_location, 
                    bid['location'],
                    use_driving_distance=use_driving_distance
                )
            else:
                # If location not specified, use a default low score
                processed_bid['location_score'] = 0.1
                
            processed_bids.append(processed_bid)
            
        return processed_bids
    
    def calculate_weighted_scores(self, 
                                 bids: List[Dict[str, Any]], 
                                 parameter_weights: Dict[str, float],
                                 preferred_trades: Union[str, List[str]] = None,
                                 user_location: str = None) -> List[Dict[str, Any]]:
        """
        Calculate weighted scores for each bid based on parameter weights.
        
        Args:
            bids: List of bid dictionaries, each containing parameter scores
                  and a 'trades' field if trade matching is needed
            parameter_weights: Dictionary mapping parameter names to weights (0 to 1)
            preferred_trades: Optional user's preferred trades for trade matching
            user_location: Optional user's location for location scoring
            
        Returns:
            List of bid dictionaries with added weighted scores and ranking
        """
        # Validate weights are in the correct range
        for param, weight in parameter_weights.items():
            if not 0 <= weight <= 1:
                raise ValueError(f"Weight for parameter '{param}' must be between 0 and 1")
        
        # Preprocess bids with location scores if needed
        if user_location and 'location' in parameter_weights and parameter_weights['location'] > 0:
            bids = self.preprocess_bids_with_location(bids, user_location)
        
        # Create a copy of bids with additional scoring info
        scored_bids = []
        
        for bid in bids:
            scored_bid = bid.copy()
            parameter_scores = {}
            weighted_parameter_scores = {}
            
            # Process trade matching if needed
            if 'trades' in bid and preferred_trades and 'trades' in parameter_weights:
                trade_score = self.score_trade_match(bid['trades'], preferred_trades)
                parameter_scores['trades'] = trade_score
                
                # Apply weight for trades parameter
                trade_weight = parameter_weights.get('trades', 0)
                weighted_parameter_scores['trades'] = trade_score * trade_weight
            
            # Process location if it has already been calculated
            if 'location_score' in bid and 'location' in parameter_weights:
                score = bid['location_score']
                parameter_scores['location'] = score
                weighted_parameter_scores['location'] = score * parameter_weights['location']
            
            # Process other parameters (assuming scores are already in the bid)
            for param, weight in parameter_weights.items():
                if param == 'trades' and preferred_trades:
                    continue  # Already handled above
                if param == 'location' and 'location_score' in bid:
                    continue  # Already handled above
                
                if param in bid:
                    score = bid[param]
                    parameter_scores[param] = score
                    weighted_parameter_scores[param] = score * weight
            
            # Calculate total weighted score
            total_weighted_score = sum(weighted_parameter_scores.values())
            
            # Add scores to the bid data
            scored_bid['parameter_scores'] = parameter_scores
            scored_bid['weighted_parameter_scores'] = weighted_parameter_scores
            scored_bid['total_weighted_score'] = total_weighted_score
            
            scored_bids.append(scored_bid)
        
        # Sort bids by total weighted score (descending)
        scored_bids.sort(key=lambda x: x['total_weighted_score'], reverse=True)
        
        # Add ranking
        for i, bid in enumerate(scored_bids, 1):
            bid['rank'] = i
        
        return scored_bids


# Example usage
def demo(outputs, weights):
    # Initialize the ranking system
    ranker = BidRankingSystem()
    
    # Example bids with trades, location, and other parameter scores
    bids = outputs
    
    # User's preferred trades
    preferred_trades = ['Electrical', 'Plumbing', 'HVAC']    
    # User's location
    user_location = 'Minneapolis, MN'
    
    # User's parameter weights
    parameter_weights = {
        'trades': 0.3,       # Trade match is important
        'location': 0.3,     # Location is equally important
    }
    
    # Rank the bids
    ranked_bids = ranker.calculate_weighted_scores(
        bids, 
        parameter_weights, 
        preferred_trades,
        user_location
    )
    
    # Display results
    print("\n=== Ranked Bids ===")
    # for bid in ranked_bids:
    #     print(f"Rank {bid['rank']}: {bid['title']} (ID: {bid['id']})")
    #     print(f"  Location: {bid['location']}")
    #     print(f"  Trades: {', '.join(bid['trades'])}")
    #     print(f"  Parameter Scores:")
    #     for param, score in bid['parameter_scores'].items():
    #         weight = parameter_weights.get(param, 0)
    #         weighted_score = bid['weighted_parameter_scores'].get(param, 0)
    #         print(f"    - {param}: {score:.2f} × weight {weight:.2f} = {weighted_score:.2f}")
    #     print(f"  Total Weighted Score: {bid['total_weighted_score']:.4f}")
    #     print()
    return ranked_bids