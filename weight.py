import pandas as pd
import numpy as np
import json
import re
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
import time
from models import Output

def normalize_weights(weights):
    """
    Normalize weights to ensure they sum to 1
    
    Args:
        weights (dict): Dictionary of parameter weights
        
    Returns:
        dict: Normalized weights
    """
    total = sum(weights.values())
    return {k: v/total for k, v in weights.items()}

def geocode_location(location_str):
    """
    Convert a text location to coordinates using geocoding
    
    Args:
        location_str (str): Location as a string (address, city, etc.)
        
    Returns:
        tuple: (latitude, longitude) or None if geocoding fails
    """
    if not location_str or not isinstance(location_str, str):
        return None
        
    try:
        # Use geopy's Nominatim geocoder with a custom user-agent
        geolocator = Nominatim(user_agent="construction_bid_ranker")
        location = geolocator.geocode(location_str)
        
        if location:
            return (location.latitude, location.longitude)
            
        # If geocoding fails, try a simpler version of the address
        simplified_location = re.sub(r'^[\d-]+\s+', '', location_str)  # Remove street number
        simplified_location = re.sub(r',.*$', '', simplified_location)  # Keep only the first part
        
        location = geolocator.geocode(simplified_location)
        if location:
            return (location.latitude, location.longitude)
            
        return None
    except Exception as e:
        print(f"Geocoding error for {location_str}: {e}")
        return None

def calculate_distance(loc1, loc2):
    """
    Calculate distance between two locations 
    
    Args:
        loc1 (str/tuple): Location of first point as string address or (lat, lng) tuple
        loc2 (str/tuple): Location of second point as string address or (lat, lng) tuple
        
    Returns:
        float: Distance in kilometers or default value if calculation fails
    """
    # Handle different location formats
    def get_coordinates(location):
        if isinstance(location, tuple) and len(location) == 2:
            return location
        elif isinstance(location, dict) and 'lat' in location and 'lng' in location:
            return (location['lat'], location['lng'])
        elif isinstance(location, dict) and 'latitude' in location and 'longitude' in location:
            return (location['latitude'], location['longitude'])
        elif isinstance(location, str):
            return geocode_location(location)
        else:
            return None
    
    # Add caching for geocoding to avoid repeated API calls
    if not hasattr(calculate_distance, 'geocode_cache'):
        calculate_distance.geocode_cache = {}
    
    # Check cache first for string locations
    if isinstance(loc1, str) and loc1 in calculate_distance.geocode_cache:
        loc1_coords = calculate_distance.geocode_cache[loc1]
    else:
        loc1_coords = get_coordinates(loc1)
        if isinstance(loc1, str):
            calculate_distance.geocode_cache[loc1] = loc1_coords
            
    if isinstance(loc2, str) and loc2 in calculate_distance.geocode_cache:
        loc2_coords = calculate_distance.geocode_cache[loc2]
    else:
        loc2_coords = get_coordinates(loc2)
        if isinstance(loc2, str):
            calculate_distance.geocode_cache[loc2] = loc2_coords
    
    if loc1_coords and loc2_coords:
        try:
            return geodesic(loc1_coords, loc2_coords).kilometers
        except Exception as e:
            print(f"Distance calculation error: {e}")
            return 100  # default high distance
    else:
        return 100  # default high distance when coordinates can't be determined

def extract_emails_text(email_list):
    """
    Extract text from an email list to create a single document for embedding
    
    Args:
        email_list (list): List of email addresses or email contents
        
    Returns:
        str: Combined text of emails
    """
    if not email_list:
        return ""
        
    if isinstance(email_list, list):
        return " ".join(email_list)
    elif isinstance(email_list, str):
        return email_list
    else:
        return ""

def calculate_embedding(text):
    """
    Create a simple vector embedding from text
    In production, use a pre-trained model like BERT, Word2Vec, etc.
    
    Args:
        text (str): Input text
        
    Returns:
        dict: Word frequency vector
    """
    if not isinstance(text, str):
        return {}
        
    words = ''.join(c.lower() if c.isalnum() else ' ' for c in text).split()
    vector = {}
    
    for word in words:
        if word:
            vector[word] = vector.get(word, 0) + 1
            
    return vector

def cosine_similarity(vec1, vec2):
    """
    Calculate cosine similarity between two vectors
    
    Args:
        vec1 (dict): First vector
        vec2 (dict): Second vector
        
    Returns:
        float: Similarity score between 0-1
    """
    all_keys = set(list(vec1.keys()) + list(vec2.keys()))
    
    dot_product = sum(vec1.get(key, 0) * vec2.get(key, 0) for key in all_keys)
    mag1 = sum(vec1.get(key, 0) ** 2 for key in all_keys) ** 0.5
    mag2 = sum(vec2.get(key, 0) ** 2 for key in all_keys) ** 0.5
    
    if mag1 == 0 or mag2 == 0:
        return 0
        
    return dot_product / (mag1 * mag2)

def parse_square_footage(sq_ft_str):
    """
    Parse square footage from various string formats
    
    Args:
        sq_ft_str (str): Square footage as a string (e.g., "5,000", "Less than 5000")
        
    Returns:
        float: Square footage as a number or default 0
    """
    if not sq_ft_str or not isinstance(sq_ft_str, str):
        return 0.0
    
    # Remove commas and convert to lowercase
    sq_ft_str = sq_ft_str.replace(',', '').lower()
    
    # Try to extract any numbers
    numbers = re.findall(r'\d+\.?\d*', sq_ft_str)
    if numbers:
        return float(numbers[0])
    
    # Handle special cases
    if "less than" in sq_ft_str:
        # Get the first number after "less than"
        match = re.search(r'less than\s+(\d+)', sq_ft_str)
        if match:
            return float(match.group(1)) * 0.75  # Assume ~75% of the upper bound
    
    return 0.0  # Default

def get_size_range(size_category):
    """
    Convert size category to a representative square footage
    
    Args:
        size_category (str): Size category ("small", "medium", "large", "extra large")
        
    Returns:
        tuple: (min_size, max_size, mid_point) representative range and middle value
    """
    size_category = str(size_category).lower().strip()
    
    if size_category == "small":
        return (0, 5000, 2500)
    elif size_category == "medium":
        return (5000, 20000, 12500)
    elif size_category == "large":
        return (20000, 50000, 35000)
    elif size_category == "extra large" or size_category == "extralarge":
        return (50000, 100000, 75000)  # Assuming upper bound of 100k for extra large
    else:
        # Try to parse as a number if it's not a category
        try:
            value = parse_square_footage(size_category)
            return (value * 0.8, value * 1.2, value)  # Create a range around the value
        except:
            return (0, 10000, 5000)  # Default range if parsing fails

def parse_project_cost(cost_str):
    """
    Parse project cost from string format
    
    Args:
        cost_str (str): Project cost as a string
        
    Returns:
        float: Project cost as a number or default 0
    """
    if not cost_str:
        return 0.0
        
    if isinstance(cost_str, (int, float)):
        return float(cost_str)
        
    # Remove non-numeric characters except decimal points
    clean_cost = re.sub(r'[^\d.]', '', str(cost_str))
    if clean_cost:
        try:
            return float(clean_cost)
        except ValueError:
            pass
            
    return 0.0  # Default

def score_past_relationship(query, email_list):
    """
    Calculate similarity between user query and related emails
    
    Args:
        query (str): User's query about past relationship
        email_list (list): List of email addresses or email texts
        
    Returns:
        float: Similarity score between 0-1
    """
    if not query or not email_list:
        return 0
        
    # Extract text from emails
    email_text = extract_emails_text(email_list)
    
    # Convert query to embedding
    query_embedding = calculate_embedding(query)
    
    # Convert email text to embedding
    email_embedding = calculate_embedding(email_text)
    
    # Calculate similarity
    return cosine_similarity(query_embedding, email_embedding)

def score_trade_match(bid_trade, preferred_trade):
    """Score for trade match (1 for exact match, partial for related trades)"""
    if not bid_trade or not preferred_trade:
        return 0.0
        
    bid_trade = str(bid_trade).lower()
    preferred_trade = str(preferred_trade).lower()
    
    if bid_trade == preferred_trade:
        return 1.0
    
    # Look for partial matches (e.g., "HVAC" in "HVAC Repair")
    if bid_trade in preferred_trade or preferred_trade in bid_trade:
        return 0.7
    
    # Calculate word similarity to handle related trades
    bid_words = set(bid_trade.split())
    preferred_words = set(preferred_trade.split())
    
    # If there are any common words, give partial score
    common_words = bid_words.intersection(preferred_words)
    if common_words:
        return 0.4 * len(common_words) / max(len(bid_words), len(preferred_words))
    
    return 0.0

def score_location(bid_location, user_location, max_distance=200):
    """Score for location (closer is better)"""
    distance = calculate_distance(bid_location, user_location)
    return max(0, 1 - (distance / max_distance))

def score_project_size(bid_size_str, ideal_size_category):
    """
    Score for project size based on categorical ranges
    
    Args:
        bid_size_str (str): Bid's square footage as a string
        ideal_size_category (str): User's preferred size category (small, medium, large, extra large)
        
    Returns:
        float: Score between 0-1, higher for better matches
    """
    bid_size = parse_square_footage(bid_size_str)
    
    # If bid size is missing, return neutral score
    if bid_size == 0:
        return 0.5
    
    # Get the size range for the ideal category
    min_size, max_size, mid_point = get_size_range(ideal_size_category)
    
    # If the bid size is within the preferred range, give high score
    if min_size <= bid_size <= max_size:
        # Give highest score (1.0) when exactly at midpoint, and 0.8 at range boundaries
        position_in_range = 1.0 - 0.2 * abs(bid_size - mid_point) / (max_size - min_size) * 2
        return max(0.8, position_in_range)
    
    # If outside the range, score based on distance from nearest boundary
    if bid_size < min_size:
        distance = min_size - bid_size
        reference = min_size
    else:  # bid_size > max_size
        distance = bid_size - max_size
        reference = max_size
    
    # Partial score for being close to the preferred range
    return max(0.1, 0.8 - 0.6 * (distance / reference))

def score_budget(bid_budget_str, user_budget_str):
    """Score for budget (under budget is better)"""
    bid_budget = parse_project_cost(bid_budget_str)
    user_budget = parse_project_cost(user_budget_str)
    
    if bid_budget == 0 or user_budget == 0:
        return 0.5  # Neutral score for missing data
    
    if bid_budget <= user_budget:
        # Under budget - higher score for closer to budget (optimizes value)
        return 0.5 + 0.5 * (bid_budget / user_budget)
    else:
        # Over budget - penalize proportionally
        over_budget_ratio = bid_budget / user_budget
        return max(0, 1 - (over_budget_ratio - 1))

def rank_bids(bids_data, weights, user_data):
    """
    Rank bids based on weighted parameters
    
    Args:
        bids_data (list/DataFrame): List of bid dictionaries or DataFrame containing bid data
        weights (dict): Dictionary of weights for each parameter
        user_data (dict): User preferences for comparison
        
    Returns:
        list/DataFrame: Sorted bids with calculated scores
    """
    # Convert to DataFrame if input is a list of dictionaries
    if isinstance(bids_data, list):
        df = pd.DataFrame(bids_data)
    else:
        df = bids_data.copy()
    
    # Normalize weights
    norm_weights = normalize_weights(weights)
    
    # Calculate individual scores
    scores = pd.DataFrame(index=df.index)
    
    # Past relationship score based on email addresses
    if 'related_emails' in df.columns and 'pastRelationship' in norm_weights and 'pastRelationshipQuery' in user_data:
        scores['past_relationship_score'] = df['related_emails'].apply(
            lambda emails: score_past_relationship(user_data['pastRelationshipQuery'], emails) * norm_weights['pastRelationship']
        )
    
    # Trade score
    if 'trade' in df.columns and 'trade' in norm_weights and 'preferredTrade' in user_data:
        scores['trade_score'] = df['trade'].apply(
            lambda x: score_trade_match(x, user_data['preferredTrade']) * norm_weights['trade']
        )
    
    # Location score
    if 'location' in df.columns and 'location' in norm_weights and 'location' in user_data:
        scores['location_score'] = df['location'].apply(
            lambda x: score_location(x, user_data['location']) * norm_weights['location']
        )
    
    # Project size score
    if 'project_size' in df.columns and 'projectSize' in norm_weights and 'idealProjectSize' in user_data:
        scores['size_score'] = df['project_size'].apply(
            lambda x: score_project_size(x, user_data['idealProjectSize']) * norm_weights['projectSize']
        )
    
    # Budget score
    if 'project_cost' in df.columns and 'budget' in norm_weights and 'budget' in user_data:
        scores['budget_score'] = df['project_cost'].apply(
            lambda x: score_budget(x, user_data['budget']) * norm_weights['budget']
        )
    
    # Calculate total score
    df['total_score'] = scores.sum(axis=1)
    
    # Sort by total score (descending)
    result_df = df.sort_values('total_score', ascending=False)
    
    # If input was a list, return as list of dictionaries
    if isinstance(bids_data, list):
        return result_df.to_dict('records')
    else:
        return result_df

def process_json_data(json_data, weights, user_data):
    """
    Process JSON data and rank bids
    
    Args:
        json_data (str or list): JSON string or already parsed JSON data
        weights (dict): Dictionary of weights for each parameter
        user_data (dict): User preferences for comparison
        
    Returns:
        list: Sorted list of bid dictionaries with scores
    """
    # Parse JSON if it's a string
    if isinstance(json_data, str):
        try:
            bids = json.loads(json_data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")
    else:
        bids = json_data
    
    # Ensure bids is a list
    if not isinstance(bids, list):
        if isinstance(bids, dict) and any(isinstance(bids.get(k), list) for k in bids):
            # Find the first list in the JSON object
            for key, value in bids.items():
                if isinstance(value, list):
                    bids = value
                    break
        else:
            raise ValueError("JSON data must contain a list of bids")
    
    # Field mappings for possible alternative field names
    field_mappings = {
        'id': ['id', 'bid_id', 'bidId'],
        'company': ['company', 'contractor', 'contractor_name', 'contractorName'],
        'related_emails': ['related_emails', 'relatedEmails', 'emails', 'communications'],
        'trade': ['trade', 'trade_type', 'tradeType'],
        'location': ['location', 'coordinates', 'position'],
        'project_size': ['project_size', 'projectSize', 'size', 'square_footage_of_work'],
        'project_cost': ['project_cost', 'cost', 'budget', 'price']
    }
    
    # Standardize field names in bids
    standardized_bids = []
    
    for bid in bids:
        standardized_bid = {}
        
        # Find matching fields using mappings
        for standard_field, possible_fields in field_mappings.items():
            for field in possible_fields:
                if field in bid:
                    standardized_bid[standard_field] = bid[field]
                    break
            
            # If no match found, keep original fields too
            if standard_field not in standardized_bid:
                standardized_bid[standard_field] = None
        
        # Copy any other fields not in our mapping
        for key, value in bid.items():
            if key not in [item for sublist in field_mappings.values() for item in sublist]:
                standardized_bid[key] = value
        
        standardized_bids.append(standardized_bid)
    
    # Rank bids
    return rank_bids(standardized_bids, weights, user_data)

def main_json():
    """
    Main function to demonstrate JSON-based bid ranking with the new structure
    """
    # Read JSON data from a file
    try:
        with open('construction_bids.json', 'r') as f:
            json_data = f.read()
    except FileNotFoundError:
        # Example JSON data if file not found (using the updated structure)
        json_data = Output.query.all().keys(id, company, related_emails, trade, location, project_size, project_cost)
        print(json_data)
        return 0
        """
        [
            {
                "id": 1,
                "company": "Jorgenson Construction Inc.",
                "related_emails": [
                    "karan@attentive.ai",
                    "rohitsaha@attentive.ai",
                    "kevin@truemechmn.com",
                    "jacob@jorgensonconstruction.com"
                ],
                "trade": "HVAC",
                "location": "600 Robert Street North, Saint Paul, MN 55101, United States of America",
                "project_size": "Less than 5000",
                "project_cost": "0"
            },
            {
                "id": 2,
                "company": "Benton County",
                "related_emails": [
                    "info@ironpeakmech.com",
                    "472616@message.planhub.com"
                ],
                "trade": "Government / Public",
                "location": "Foley, Minnesota",
                "project_size": "33502.00",
                "project_cost": "0"
            },
            {
                "id": 3,
                "company": "PlanHub, Inc.",
                "related_emails": [
                    "781148@message.planhub.com",
                    "info@ironpeakmech.com"
                ],
                "trade": "Commercial",
                "location": "Saint Paul, Minnesota",
                "project_size": "6,600.00",
                "project_cost": "0"
            }
        ]
        """
    
    # User-provided weights (0-1 for each parameter)
    weights = {
        'pastRelationship': 0.8,  # High importance on past relationship
        'trade': 0.5,             # Medium importance on trade match
        'location': 1.0,          # Highest importance on location
        'projectSize': 0.3,       # Lower importance on project size
        'budget': 0.7             # High importance on budget
    }
    
    # User data for comparison - now with categorical project size
    user_data = {
        'pastRelationshipQuery': "Worked with HVAC contractors in Minnesota",
        'preferredTrade': 'HVAC',
        'location': "Saint Paul, Minnesota",
        'idealProjectSize': "small",  # Now using category: small, medium, large, extra large
        'budget': "50000"
    }
    
    # Process and rank bids
    ranked_bids = process_json_data(json_data, weights, user_data)
    
    # Print results
    print("Ranked Bids:")
    for i, bid in enumerate(ranked_bids):
        print(f"{i+1}. {bid.get('id', 'N/A')} - {bid.get('company', 'N/A')} - Score: {bid.get('total_score', 0):.2f}")
    
    # Get top bid
    if ranked_bids:
        top_bid = ranked_bids[0]
        print("\nTop Recommended Bid:")
        print(f"ID: {top_bid.get('id', 'N/A')}")
        print(f"Company: {top_bid.get('company', 'N/A')}")
        print(f"Trade: {top_bid.get('trade', 'N/A')}")
        print(f"Location: {top_bid.get('location', 'N/A')}")
        print(f"Project Size: {top_bid.get('project_size', 'N/A')}")
        print(f"Score: {top_bid.get('total_score', 0):.2f}")
        
        # Print a structured output for the top bid
        size_value = parse_square_footage(top_bid.get('project_size', '0'))
        size_category = "Unknown"
        if 0 < size_value <= 5000:
            size_category = "Small"
        elif 5000 < size_value <= 20000:
            size_category = "Medium"
        elif 20000 < size_value <= 50000:
            size_category = "Large"
        elif size_value > 50000:
            size_category = "Extra Large"
        
        structured_output = {
            "id": top_bid.get('id', 'N/A'),
            "company": top_bid.get('company', 'N/A'),
            "trade": top_bid.get('trade', 'N/A'),
            "location": top_bid.get('location', 'N/A'),
            "project_size": {
                "raw_value": top_bid.get('project_size', 'N/A'),
                "numeric_value": size_value,
                "category": size_category
            },
            "total_score": float(f"{top_bid.get('total_score', 0):.2f}")
        }
        
        print("\nStructured Output:")
        print(json.dumps(structured_output, indent=2))

def read_bids_from_json_file(file_path):
    """
    Read construction bids from a JSON file
    
    Args:
        file_path (str): Path to JSON file
        
    Returns:
        list: List of bid dictionaries
    """
    with open(file_path, 'r') as f:
        json_data = f.read()
    
    # Parse JSON
    try:
        return json.loads(json_data)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in {file_path}: {e}")

def calculate_weight(output_data, filter_data):
    """
    Calculate weight based on matching criteria between output and filter
    """
    weight = 0
    
    # Check trades match
    output_trades = set(output_data['trade'].split(', '))
    filter_trades = set(filter_data['trades'])
    trade_matches = output_trades.intersection(filter_trades)
    if trade_matches:
        weight += len(trade_matches) * 10  # 10 points per matching trade
    
    # Check scope of work match
    if output_data['scope_of_work'] in filter_data['scope_of_work']:
        weight += 20  # 20 points for matching scope
    
    # Check building type match
    if output_data['type_of_building'].lower() == filter_data['building_type'].lower():
        weight += 15  # 15 points for matching building type
    
    # Check job type match
    if output_data['type_of_job'].lower() == filter_data['job_type'].lower():
        weight += 15  # 15 points for matching job type
    
    # Check project budget range
    try:
        output_cost = float(output_data['project_cost'].replace(',', ''))
        if output_cost <= filter_data['project_budget']:
            weight += 20  # 20 points if within budget
    except (ValueError, TypeError):
        pass
    
    # Check if company is blacklisted
    if output_data['company'] in filter_data['blacklisted_companies']:
        weight = 0  # Zero weight if company is blacklisted
    
    return weight

def get_weighted_outputs(outputs, filter_data):
    """
    Get outputs with their calculated weights
    """
    weighted_outputs = []
    for output in outputs:
        weight = calculate_weight(output.to_dict(), filter_data)
        output_dict = output.to_dict()
        output_dict['weight'] = weight
        weighted_outputs.append(output_dict)
    
    # Sort by weight in descending order
    weighted_outputs.sort(key=lambda x: x['weight'], reverse=True)
    return weighted_outputs

if __name__ == "__main__":
    main_json()