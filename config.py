import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-please-change-in-production'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///app.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False 
    STATE_AND_FEDERAL_LOGIN_URL = os.environ.get('STATE_AND_FEDERAL_URL') or 'https://www.stateandfederalbids.com/bids/myAccount'
    STATE_AND_FEDERAL_USERNAME = os.environ.get('STATE_AND_FEDERAL_USERNAME') or 'your_email@example.com'
    STATE_AND_FEDERAL_PASSWORD = os.environ.get('STATE_AND_FEDERAL_PASSWORD') or 'your_password'
    GOOGLE_MAPS_API_KEY = os.environ.get('GOOGLE_MAPS_API_KEY') or 'your_google_maps_api_key'