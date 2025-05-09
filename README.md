# Flask SQLAlchemy API

A basic Flask application with SQLAlchemy ORM for database operations.

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

The application will be available at `http://localhost:5000`

## API Endpoints

- `GET /`: Welcome message
- `GET /users`: List all users
- `POST /users`: Create a new user
  - Required JSON body: `{"username": "string", "email": "string"}`

## Database

The application uses SQLite by default. The database file will be created as `app.db` in the project root directory.

To use a different database, set the `DATABASE_URL` environment variable with your database connection string. 