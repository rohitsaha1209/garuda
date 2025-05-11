from typing import List, Dict, Any
from geopy.distance import geodesic
from sentence_transformers import SentenceTransformer, util

class BidRankingSystem:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def geocode_address(self, address: str) -> tuple:
        """Mock geocoding function. Replace with real API in production."""
        return (0.0, 0.0)  # Dummy coordinates

    def calculate_distance(self, coord1: tuple, coord2: tuple) -> float:
        return geodesic(coord1, coord2).miles

    def score_location(self, bid_location: str, user_location: str) -> float:
        bid_coords = self.geocode_address(bid_location)
        user_coords = self.geocode_address(user_location)
        distance = self.calculate_distance(bid_coords, user_coords)
        if distance <= 10:
            return 1.0
        elif distance <= 50:
            return 0.8
        elif distance <= 100:
            return 0.6
        elif distance <= 200:
            return 0.4
        elif distance <= 300:
            return 0.2
        else:
            return 0.0

    def score_trade_match(self, bid_trades: List[str], user_trades: List[str]) -> float:
        if not bid_trades or not user_trades:
            return 0.0
        bid_embedding = self.model.encode(" ".join(bid_trades), convert_to_tensor=True)
        user_embedding = self.model.encode(" ".join(user_trades), convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(bid_embedding, user_embedding).item()
        return max(0.0, min(similarity, 1.0))

    def score_budget(self, bid_budget: float, user_budget: float) -> float:
        if bid_budget <= user_budget:
            return 1.0
        return max(0.0, 1 - ((bid_budget - user_budget) / user_budget))

    def score_project_size(self, bid_project_size: float, user_project_size: float) -> float:
        if bid_project_size <= user_project_size:
            return 1.0
        return max(0.0, 1 - ((bid_project_size - user_project_size) / user_project_size))

    def score_past_relationship(self, past_worked: bool) -> float:
        return 1.0 if past_worked else 0.0

    def preprocess_bids_with_parameters(self, bids: List[Dict[str, Any]], user_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        processed_bids = []
        for bid in bids:
            processed_bid = bid.copy()
            if 'location' in bid and 'location' in user_params:
                processed_bid['location_score'] = self.score_location(bid['location'], user_params['location'])
            if 'project_cost' in bid and 'budget' in user_params:
                processed_bid['budget_score'] = self.score_budget(float(bid['project_cost']), user_params['budget'])
            if 'project_size' in bid and 'projectSize' in user_params:
                processed_bid['project_size_score'] = self.score_project_size(bid['project_size'], user_params['projectSize'])
            if 'pastRelationship' in bid:
                processed_bid['relationship_score'] = self.score_past_relationship(bid['pastRelationship'])
            processed_bids.append(processed_bid)
        return processed_bids

    def calculate_weighted_scores(self, bids: List[Dict[str, Any]], parameter_weights: Dict[str, float], user_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        for param, weight in parameter_weights.items():
            if not isinstance(weight, (int, float)) or not 0 <= weight <= 1:
                raise ValueError(f"Invalid weight for '{param}': must be a float between 0 and 1")

        scored_bids = []
        for bid in self.preprocess_bids_with_parameters(bids, user_params):
            scored_bid = bid.copy()
            parameter_scores = {}
            weighted_scores = {}

            if 'trades' in parameter_weights and 'trades' in bid and 'trades' in user_params:
                trade_score = self.score_trade_match(bid['trades'], user_params['trades'])
                parameter_scores['trades'] = trade_score
                weighted_scores['trades'] = trade_score * parameter_weights['trades']

            param_to_field = {
                'location': 'location_score',
                'budget': 'budget_score',
                'projectSize': 'project_size_score',
                'pastRelationship': 'relationship_score'
            }

            for param, field in param_to_field.items():
                if param in parameter_weights and field in bid:
                    score = bid[field]
                    if isinstance(score, (int, float)):
                        parameter_scores[param] = score
                        weighted_scores[param] = score * parameter_weights[param]

            total_score = sum(weighted_scores.values())
            scored_bid.update({
                'parameter_scores': parameter_scores,
                'weighted_parameter_scores': weighted_scores,
                'total_weighted_score': total_score
            })
            scored_bids.append(scored_bid)

        scored_bids.sort(key=lambda b: b['total_weighted_score'], reverse=True)
        for idx, bid in enumerate(scored_bids, 1):
            bid['rank'] = idx

        return scored_bids

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