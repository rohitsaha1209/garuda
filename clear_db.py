from app import app, db
from models import User, Filter, Output

def clear_database():
    with app.app_context():
        # Delete all data from all tables
        Output.query.delete()
        Filter.query.delete()
        User.query.delete()
        
        # Commit the changes
        db.session.commit()
        print("All data has been cleared from the database.")

if __name__ == "__main__":
    clear_database() 