from flask import Flask, jsonify, request
from models import db, User, Filter, Output, ProjectSize, Output, Status
from config import Config
from bs4 import BeautifulSoup
import requests
from flask_cors import CORS
import json
from datetime import datetime
from bs4 import BeautifulSoup
from ranking3 import demo, geocode_location
from stateandfederal import login_and_save_state, use_logged_in_session

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.config.from_object(Config)
db.init_app(app)

# Create database tables
with app.app_context():
    db.create_all()

@app.route('/')
def index():
    return jsonify({"message": "Welcome to Flask SQLAlchemy API"})

@app.route('/get_links', methods=['POST'])
def get_links():
    # return jsonify(output={"bid_details_link": {"Bid link": "https://www.google.com"}})
    # Assuming the HTML is stored under the key 'html_content'
    data = request.get_json()
    html_content = data.get('html_content', '')
    # html_content = json.loads(html_content)

    # Parse the HTML using BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find all anchor tags with an href attribute
    links = soup.find_all('a', href=True)

    # Extract and filter links containing the specific API method
    kv_links = {}

    for link in links:
        href = link['href']
        text = link.get_text(strip=True)  # Clean up extra spaces/newlines
        kv_links[text] = href

    filter_out_keys = []
    for key, val in kv_links.items():
        if "@" in key :
            filter_out_keys.append(key)
        if "tel" in val:
            filter_out_keys.append(key)

    for key in filter_out_keys:
        del kv_links[key]

    print("Total matching links:", len(kv_links))

    return jsonify({"output":{ "bid_details_link": kv_links}})


@app.route('/users', methods=['GET'])
def get_users():
    users = User.query.all()
    return jsonify([{
        'id': user.id,
        'username': user.username,
        'email': user.email,
        'created_at': user.created_at.isoformat()
    } for user in users])

@app.route('/users', methods=['POST'])
def create_user():
    data = request.get_json()
    
    if not data or not data.get('username') or not data.get('email'):
        return jsonify({"error": "Username and email are required"}), 400
    
    try:
        user = User(
            username=data['username'],
            email=data['email']
        )
        db.session.add(user)
        db.session.commit()
        return jsonify({
            'id': user.id,
            'username': user.username,
            'email': user.email,
            'created_at': user.created_at.isoformat()
        }), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 400

@app.route('/start_workflow', methods=['POST'])
def create_filter():
    data = request.get_json()
    import requests
    import json

    url = "http://34.100.131.110:5100/start_workflow"

    payload = json.dumps(data)
    headers = {
    'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)
    # hit workflow
    url = "http://localhost:5678/webhook/e57db4a5-048f-47fa-a395-4dfe98f6aa7a"

    payload = json.dumps({
        "filter_id": ""
        })
    headers = {
        'Content-Type': 'application/json'
        }
    response = requests.request("POST", url, headers=headers, data=payload)

    print("Response from filter webhook: ", response.text)
    return jsonify({"message": response.text})

@app.route('/get_latest_filter', methods=['GET'])
def get_latest_filter():
    filter_obj = Filter.query.order_by(Filter.created_at.desc()).first()
    if filter_obj:
        return jsonify(filter_obj.to_dict()), 200
    return jsonify({"error": "No filters found"}), 404

@app.route('/filters', methods=['GET'])
def get_filters():
    filters = Filter.query.all()
    return jsonify([filter_obj.to_dict() for filter_obj in filters])

@app.route('/filters/<int:filter_id>', methods=['GET'])
def get_filter(filter_id):
    filter_obj = Filter.query.get_or_404(filter_id)
    return jsonify(filter_obj.to_dict())

@app.route('/public-bid-links', methods=['POST'])
def get_links_for_public_bids():
    data = request.get_json()
    html_content = data.get('html_content')
    if not html_content:
        return jsonify({"error": "html_content is required in the request body"}), 400

    soup = BeautifulSoup(html_content, 'html.parser')
    links = soup.find_all('a', href=True)
    filtered_links = []
    MATCHING_API_METHOD = "ShowBid"
    for link in links:
        if MATCHING_API_METHOD in link["href"]:
            filtered_links.append(link["href"])
    return jsonify({"filtered_links": filtered_links, "count": len(filtered_links)})

@app.route('/output', methods=['POST'])
def create_output():
    data = request.get_json()
    
    if not data or not isinstance(data, list) or len(data) == 0:
        return jsonify({"error": "Request must contain a non-empty array of output data"}), 400
    
    output_data = data[0]  # Get the first record
    if not output_data.get('output'):
        return jsonify({"error": "Each record must contain 'output' fields"}), 400
    
    try:
        output = output_data['output']
        print("This is size:", output['project_size'])
        geocode = geocode_location(output['location'])
        
        # Create new output record
        output_obj = Output(
            location=output['location'],
            project_name=output['project_name'],
            geocode=geocode,
            project_description=output['project_description'],
            company=output['company'],
            bid_due_date=datetime.strptime(output['bid_due_date'], '%d-%m-%Y').date() if output['bid_due_date']!="" else None,
            project_start_date=datetime.strptime(output['project_start_date'], '%d-%m-%Y').date() if output['project_start_date']!="" else None,
            project_end_date=datetime.strptime(output['project_end_date'], '%d-%m-%Y').date() if output['project_end_date']!="" else None,
            project_cost=output['project_cost'],
            trades=output['trades'],
            scope_of_work=output['scope_of_work'],
            complexity_of_the_project=output.get('complexity_of_the_project'),
            area_of_expertise=output.get('area_of_expertise'),
            square_footage_of_work=output.get('square_footage_of_work'),
            project_size=output['project_size'],
            type_of_building=output['type_of_building'],
            type_of_job=output['type_of_job'],
            is_public_work=output['is_public_work'],
            is_private_work=output['is_private_work'],
            bid_details_link=output.get('bid_details_link', []),
            related_emails=output.get('related_emails', [])
        )
        
        db.session.add(output_obj)
        db.session.commit()
        
        return jsonify(output_obj.to_dict()), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 400

@app.route('/outputs', methods=['POST'])
def get_outputs():
    outputs = Output.query.all()
    return jsonify([output.to_dict() for output in outputs])

@app.route('/output/<int:output_id>', methods=['GET'])
def get_output(output_id):
    output = Output.query.get_or_404(output_id)
    return jsonify(output.to_dict())

@app.route('/status', methods=['GET'])
def get_status():
    status = Status.query.first()
    return jsonify({"status": status.status})


@app.route('/end_workflow', methods=['POST'])
def end_workflow():
    Status.query.delete()
    status = Status(status="ended")
    db.session.add(status)
    db.session.commit()
    return jsonify({"message": "Workflow ended"})


@app.route('/get_weighted_outputs', methods=['POST'])
def get_weighted_outputs():
    weights = request.get_json()
    outputs = Output.query.all()
    outputs = [output.to_dict() for output in outputs]
    filters = Filter.query.first()
    response = demo(outputs, weights, filters.to_dict())

    return jsonify(response)


# STATE and Federal APIS

@app.route('/login_state_and_federal', methods=['POST'])
def login_state_and_federal():
    data = request.get_json()
    username = app.config['STATE_AND_FEDERAL_USERNAME']
    password = app.config['STATE_AND_FEDERAL_PASSWORD']
    url = app.config['STATE_AND_FEDERAL_LOGIN_URL']
    try:
        login_and_save_state(url, username, password)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"message": "Login successful"})


@app.route('/get_state_and_federal_bids', methods=['POST'])
def get_state_and_federal_bids():
    data = request.get_json()
    url = data.get('url')
    html_content = use_logged_in_session(url)
    return jsonify({"html_content": html_content})


if __name__ == '__main__':
    app.run(port=5000,debug=True) 