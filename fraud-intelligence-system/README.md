# Fraud Intelligence System

## API
- Start the API:
  `uvicorn api.chatgpt_codex_api:app --reload`
- Health check:
  `GET /health`
- Generate a sample transaction:
  `GET /transactions/generate`
- Score a transaction:
  `POST /transactions/score`

## Notes
- Activate the venv first:
  `./venv/Scripts/Activate.ps1`
