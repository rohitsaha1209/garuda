from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
import enum

db = SQLAlchemy()

class ProjectSize(enum.Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<User {self.username}>'

class Status(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    status = db.Column(db.String(20), nullable=False)


class Filter(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    trades = db.Column(db.JSON, nullable=False)  # Array of trades
    blacklisted_companies = db.Column(db.JSON, nullable=False)  # Array of blacklisted companies
    project_size = db.Column(db.Enum(ProjectSize), nullable=False)
    scope_of_work = db.Column(db.JSON, nullable=False)  # Array of scope of work
    project_budget = db.Column(db.Float, nullable=False)
    job_type = db.Column(db.String(50), nullable=False)
    building_type = db.Column(db.String(50), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f'<Filter {self.id}>'

    def to_dict(self):
        return {
            'id': self.id,
            'trades': self.trades,
            'blacklisted_companies': self.blacklisted_companies,
            'project_size': self.project_size.value,
            'scope_of_work': self.scope_of_work,
            'project_budget': self.project_budget,
            'job_type': self.job_type,
            'building_type': self.building_type,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

class Output(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    location = db.Column(db.String(100), nullable=False)
    project_name = db.Column(db.String(200), nullable=False)
    project_description = db.Column(db.Text, nullable=False)
    company = db.Column(db.String(200), nullable=False)
    bid_due_date = db.Column(db.Date, nullable=True)
    project_start_date = db.Column(db.Date, nullable=True)
    project_end_date = db.Column(db.Date, nullable=True)
    project_cost = db.Column(db.String(50), nullable=False)
    trade = db.Column(db.String(200), nullable=False)
    scope_of_work = db.Column(db.String(100), nullable=False)
    complexity_of_the_project = db.Column(db.Text, nullable=True)
    area_of_expertise = db.Column(db.String(100), nullable=True)
    square_footage_of_work = db.Column(db.String(50), nullable=True)
    type_of_building = db.Column(db.String(100), nullable=False)
    type_of_job = db.Column(db.String(50), nullable=False)
    is_public_work = db.Column(db.Boolean, default=False)
    is_private_work = db.Column(db.Boolean, default=False)
    bid_details_link = db.Column(db.JSON, nullable=True)  # Array of URLs
    related_emails = db.Column(db.JSON, nullable=True)  # Array of emails
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f'<Output {self.id}>'

    def to_dict(self):
        return {
            'id': self.id,
            'location': self.location,
            'project_name': self.project_name,
            'project_description': self.project_description,
            'company': self.company,
            'bid_due_date': self.bid_due_date.strftime('%d-%m-%Y') if self.bid_due_date else None,
            'project_start_date': self.project_start_date.strftime('%d-%m-%Y') if self.project_start_date else None,
            'project_end_date': self.project_end_date.strftime('%d-%m-%Y') if self.project_end_date else None,
            'project_cost': self.project_cost,
            'trade': self.trade,
            'scope_of_work': self.scope_of_work,
            'complexity_of_the_project': self.complexity_of_the_project,
            'area_of_expertise': self.area_of_expertise,
            'square_footage_of_work': self.square_footage_of_work,
            'type_of_building': self.type_of_building,
            'type_of_job': self.type_of_job,
            'is_public_work': self.is_public_work,
            'is_private_work': self.is_private_work,
            'bid_details_link': self.bid_details_link,
            'related_emails': self.related_emails,
            'created_at': self.created_at.strftime('%d-%m-%Y %H:%M:%S') if self.created_at else None,
            'updated_at': self.updated_at.strftime('%d-%m-%Y %H:%M:%S') if self.updated_at else None
        } 