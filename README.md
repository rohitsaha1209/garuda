
---

## Running the App

### Development

```bash
python app.py
```

### Production (with Gunicorn)

```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app --timeout 120
```

### Docker

```bash
docker build -t bid-app .
docker run -p 5000:5000 --env-file .env bid-app
```

---

## API Endpoints

- `GET /users` — List users
- `POST /users` — Create user
- `GET /filters` — List filters
- `POST /filters` — Create filter
- `GET /output` — List all outputs (optionally weighted)
- `POST /output` — Add output record
- `GET /public-bid-links` — Extract filtered links from HTML
- `POST /get_state_and_federal_bids` — (Custom) Fetch authenticated content via Playwright

See the code for full details and request/response formats.

---

## Bid Ranking System

- Uses NLP (Sentence Transformers) for semantic trade matching
- Geocodes and scores locations
- Scores budget, project size, and past relationships
- Customizable parameter weights
- Returns ranked list of bids

See `ranking3.py` for details.

---

## Automated Web Login & Scraping

- `stateandfederal.py` provides functions to log in and fetch authenticated pages using Playwright.
- Session state is saved and reused for efficient scraping.

---

## Development Notes

- All configuration is via environment variables or `.env` file.
- Database is auto-created on first run (SQLite by default).
- For schema changes, delete `app.db` and restart the app (or use migrations if enabled).

---

## License

MIT license

---

## Acknowledgements

- [Flask](https://flask.palletsprojects.com/)
- [SQLAlchemy](https://www.sqlalchemy.org/)
- [Playwright](https://playwright.dev/python/)
- [Sentence Transformers](https://www.sbert.net/)