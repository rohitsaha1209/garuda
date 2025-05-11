from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Union, List, Dict, Any, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
import requests
import time
from math import radians, cos, sin, asin, sqrt, exp
import pandas as pd
import re

import time
import logging

# Set up logging (you can configure this as needed)
logging.basicConfig(level=logging.INFO)


def time_tracker(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        logging.info(f"Function {func.__name__} took {duration:.4f} seconds")
        return result
    return wrapper

def geocode_location(location_str: str) -> Optional[Dict[str, Any]]:
        """
        Convert a location string to latitude and longitude using OpenStreetMap's Nominatim API
        """
        encoded_location = requests.utils.quote(location_str)
        url = f"https://nominatim.openstreetmap.org/search?format=json&q={encoded_location}"
        
        headers = {
            "User-Agent": "BidRankingSystem/1.0"  # Required by Nominatim
        }
        
        # Respect rate limits (1 request per second)
        time.sleep(0.2)
        
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



class BidRankingSystem:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the bid ranking system with a vector embedding model.
        
        Args:
            model_name: The name of the SentenceTransformer model to use
        """
        self.model = SentenceTransformer(model_name)
        self.user_geocoded_location = None
    
    #
    # TRADE MATCHING METHODS
    #
    @time_tracker
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
    
    @time_tracker
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
    
    #
    # LOCATION SCORING METHODS
    #
    @time_tracker
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
    
    @time_tracker
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
        time.sleep(0.2)
        
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
    
    @time_tracker
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
    
    @time_tracker
    def calculate_location_score(self, 
                               user_location: str, 
                               bid_location: str, 
                               bid_geocode,
                               use_driving_distance: bool = False) -> float:
        """
        Calculate a location score (0-1) based on proximity.
        Uses an inverse proportion that gradually decreases as distance increases.
        
        Args:
            user_location: String of user's location
            bid_location: String of bid's location
            use_driving_distance: Whether to use driving distance vs straight-line
            
        Returns:
            A score between 0 and 1, where 1 means same location and 
            score approaches 0 as distance increases
        """
        # Check for exact string match first (faster)
        if user_location.lower() == bid_location.lower():
            return 1.0
            
        # Geocode both locations
        if self.user_geocoded_location is None:
            user_coords = self.geocode_location(user_location)
            self.user_geocoded_location = user_coords
        else:
            user_coords = self.user_geocoded_location
        
        bid_coords = bid_geocode
        
        # If geocoding fails, use string similarity as fallback
        if not user_coords or not bid_coords:
            # Simple string similarity as fallback
            user_parts = user_location.lower().split(',')
            bid_parts = bid_location.lower().split(',')
            
            # Check if state/country matches
            if len(user_parts) > 1 and len(bid_parts) > 1:
                if user_parts[-1].strip() == bid_parts[-1].strip():
                    return 0.7  # Same state/country
            
            return 0.3  # Different state/country or couldn't determine
        
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
        
        # Use an inverse proportion formula similar to budget scoring
        # The scaling_factor (100) determines how quickly score decreases with distance
        # At 100km, score will be 0.5
        # At 400km, score will be 0.2
        scaling_factor = 100  # in kilometers
        score = 1 / (1 + (distance / scaling_factor))
        
        # Ensure score is between 0 and 1
        return max(0, min(1, float(score)))
    
    #
    # PROJECT BUDGET SCORING
    #
    @time_tracker
    def score_budget_match(self, 
                         bid_budget: float, 
                         user_budget: float) -> float:
        """
        Calculate a score based on how close the bid budget is to the user's preferred budget.
        Uses an inverse proportion that gradually decreases as the difference increases.
        
        Args:
            bid_budget: The budget amount of the bid
            user_budget: The user's preferred budget
            
        Returns:
            A score between 0 and 1, where 1 means exact match and 
            score approaches 0 as the difference increases
        """
        if bid_budget == user_budget:
            return 1.0
        
        # Calculate absolute percentage difference
        percentage_diff = abs(bid_budget - user_budget) / user_budget
        
        # Use an inverse function to calculate score
        # This gives a smooth curve from 1.0 (no difference) approaching 0 (large difference)
        # The multiplier 5 in the denominator controls how quickly the score drops
        score = 1 / (1 + (percentage_diff * 5))
        
        return float(score)
    
    #
    # PROJECT SIZE SCORING
    #
    @time_tracker
    def score_project_size(self, 
                         bid_size: str, 
                         preferred_size: str) -> float:
        """
        Score the match between bid project size and preferred project size.
        
        Args:
            bid_size: The size of the project in the bid (small, medium, large)
            preferred_size: The preferred size (small, medium, large)
            
        Returns:
            1.0 for exact match, 0.5 for one size category difference, 0.0 for two size categories difference
        """
        # Normalize input strings
        bid_size = bid_size.lower().strip()
        preferred_size = preferred_size.lower().strip()
        
        # Define size categories and their relative positions
        size_order = {"small": 0, "medium": 1, "large": 2}
        
        # If either size is not in the standard categories, return a low score
        if bid_size not in size_order or preferred_size not in size_order:
            return 0.1  # Return a small non-zero score as fallback
        
        # Calculate the distance between size categories
        size_distance = abs(size_order[bid_size] - size_order[preferred_size])
        
        # Score based on size distance
        if size_distance == 0:
            return 1.0  # Exact match
        elif size_distance == 1:
            return 0.5  # One category off
        else:
            return 0.0  # Two categories off
    
    #
    # PAST RELATIONSHIP SCORING
    #
    @time_tracker
    def extract_name_from_email(self, email: str) -> str:
        """
        Extract name part from an email address
        
        Args:
            email: An email address string
            
        Returns:
            The name part of the email (before @)
        """
        try:
            return email.split('@')[0].lower()
        except:
            return ''
    
    @time_tracker
    def score_past_relationship(self, 
                              bid_related_emails: List[str], 
                              user_contacts: List[Union[str, Dict[str, str]]]) -> float:
        """
        Score the bid based on past relationships with contacts.
        
        Args:
            bid_related_emails: List of email addresses associated with the bid
            user_contacts: List of user's contacts, either as email strings or 
                          dictionaries with 'name' and/or 'email' keys
            
        Returns:
            A score between 0 and 1 based on matches found
        """
        if not bid_related_emails or not user_contacts:
            return 0.0
        
        # Normalize bid emails
        bid_emails_lower = [email.lower() for email in bid_related_emails]
        bid_names_lower = [self.extract_name_from_email(email) for email in bid_emails_lower]
        
        # Extract user contact information
        user_emails = []
        user_names = []
        
        for contact in user_contacts:
            if isinstance(contact, str):
                # If it's a string, treat it as an email
                user_emails.append(contact.lower())
                user_names.append(self.extract_name_from_email(contact))
            elif isinstance(contact, dict):
                # If it's a dictionary, extract email and name
                if 'email' in contact:
                    user_emails.append(contact['email'].lower())
                if 'name' in contact:
                    user_names.append(contact['name'].lower())
        
        # Count matches
        email_matches = sum(1 for email in user_emails if email in bid_emails_lower)
        
        # Look for name matches in email usernames
        name_matches = 0
        for name in user_names:
            if name and any(name in bid_name for bid_name in bid_names_lower):
                name_matches += 1
        
        # Calculate score based on the number of matches
        total_contacts = len(user_contacts)
        match_score = min(1.0, (email_matches + 0.5 * name_matches) / total_contacts)
        
        return float(match_score)
    
    #
    # COMBINED SCORING AND RANKING
    #   
    @time_tracker
    def preprocess_bids_with_parameters(self, 
                                      bids: List[Dict[str, Any]],
                                      user_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Preprocess bids to add scores for all parameters
        
        Args:
            bids: List of bid dictionaries
            user_params: Dictionary of user parameters for scoring
            
        Returns:
            Bids with added parameter scores
        """
        processed_bids = []
        
        for bid in bids:
            processed_bid = bid.copy()
            
            # Score location if needed
            if 'location' in bid and 'location' in user_params:
                processed_bid['location_score'] = self.calculate_location_score(
                    user_params['location'], 
                    bid['location'],
                    bid['geocode'],
                    use_driving_distance=user_params.get('use_driving_distance', False)
                )
            
            # Score budget if needed
            if 'project_cost' in bid and 'budget' in user_params:
                processed_bid['budget_score'] = self.score_budget_match(
                    float(bid['project_cost']) if bid['project_cost'] != "" else 0,
                    user_params['budget']
                )
            
            # Score project size if needed
            if 'project_size' in bid and 'project_size' in user_params:
                processed_bid['project_size_score'] = self.score_project_size(
                    bid['project_size'],
                    user_params['project_size']
                )
            
            # Score past relationships if needed
            if 'related_emails' in bid and 'contacts' in user_params:
                processed_bid['relationship_score'] = self.score_past_relationship(
                    bid['related_emails'],
                    user_params['contacts']
                )
                
            processed_bids.append(processed_bid)
            
        return processed_bids
    
    @time_tracker
    def calculate_weighted_scores(self, 
                                 bids: List[Dict[str, Any]], 
                                 parameter_weights: Dict[str, float],
                                 user_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Calculate weighted scores for each bid based on parameter weights and user preferences.
        
        Args:
            bids: List of bid dictionaries with various fields
            parameter_weights: Dictionary mapping parameter names to weights (0 to 1)
            user_params: Dictionary containing user preferences for scoring:
                - trades: preferred trades (string or list)
                - location: user's location
                - budget: user's budget
                - project_size: preferred project size
                - contacts: user's contacts for relationship scoring
                - use_driving_distance: whether to use driving distance (optional)
                - budget_tolerance: tolerance percentage for budget (optional)
            
        Returns:
            List of bid dictionaries with added weighted scores and ranking
        """
        # Validate weights are in the correct range
        for param, weight in parameter_weights.items():
            if not 0 <= weight <= 1:
                raise ValueError(f"Weight for parameter '{param}' must be between 0 and 1")
        
        # Preprocess bids with all parameter scores
        bids = self.preprocess_bids_with_parameters(bids, user_params)
        
        # Create a copy of bids with additional scoring info
        scored_bids = []
        
        # Parameter to score field mapping
        param_score_fields = {
            'trades': 'trade_score',
            'location': 'location_score',
            'budget': 'budget_score',
            'projectSize': 'project_size_score',
            'pastRelationship': 'relationship_score'
        }
        
        for bid in bids:
            scored_bid = bid.copy()
            parameter_scores = {}
            weighted_parameter_scores = {}
            
            # Process trades if needed
            if 'trades' in bid and 'trades' in user_params and 'trades' in parameter_weights:
                trade_score = self.score_trade_match(bid['trades'], user_params['trades'])
                parameter_scores['trades'] = trade_score
                weighted_parameter_scores['trades'] = trade_score * parameter_weights['trades']
            
            # Process other parameters that have already been calculated
            for param, score_field in param_score_fields.items():
                if param == 'trades':  # Already handled above
                    continue
                    
                if score_field in bid and param in parameter_weights:
                    score = bid[score_field]
                    parameter_scores[param] = score
                    weighted_parameter_scores[param] = score * parameter_weights[param]
            
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
def demo(outputs, weights, filters):
    # Initialize the ranking system
    ranker = BidRankingSystem()
    
    # Example bids with all 5 parameters
    bids = outputs
    
    # User parameters for scoring
    user_params = {
        'trades': filters['trades'],  # Preferred trades
        'location': filters['property_address'],                      # User's location
        'budget': filters['project_budget'],                              # Target budget
        'project_size': filters['project_size'],                       # Preferred project size
        'contacts': filters['past_relationships']
    }
    
    # Rank the bids
    ranked_bids = ranker.calculate_weighted_scores(
        bids, 
        weights, 
        user_params
    )
    
    return ranked_bids