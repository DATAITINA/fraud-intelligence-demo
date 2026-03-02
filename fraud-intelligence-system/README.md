diff --git a/C:\Users\DELL\Desktop\AI Engineering\Projects\fraud-intelligence-system\README.md b/C:\Users\DELL\Desktop\AI Engineering\Projects\fraud-intelligence-system\README.md
--- a/C:\Users\DELL\Desktop\AI Engineering\Projects\fraud-intelligence-system\README.md
+++ b/C:\Users\DELL\Desktop\AI Engineering\Projects\fraud-intelligence-system\README.md
@@ -2,14 +2,35 @@
 
-## API
-- Start the API:
-  `uvicorn api.chatgpt_codex_api:app --reload`
-- Health check:
-  `GET /health`
-- Generate a sample transaction:
-  `GET /transactions/generate`
-- Score a transaction:
-  `POST /transactions/score`
+Synthetic fraud detection workflow with a FastAPI scoring service, synthetic data generation, and a lightweight PyTorch model.
+
+## Quick Start
+```powershell
+python -m venv .venv
+.\.venv\Scripts\Activate.ps1
+pip install -r requirements.txt
+python model/train.py
+uvicorn api.fraud_api:app --reload
+```
 
+## API Endpoints
+- `GET /health` → service status
+- `GET /transactions/generate` → generate a synthetic transaction payload
+- `POST /transactions/score` → score a transaction payload
+
+Example request (PowerShell):
+```powershell
+Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/transactions/score -ContentType "application/json" -Body (@{
+  amount = 120.50
+  currency = "USD"
+  category = "Electronics"
+  merchant = "Amazon"
+} | ConvertTo-Json)
+```
+
+## Project Structure
+- `api/` FastAPI service
+- `data/` synthetic transaction generation + investigation helpers
+- `model/` training, inference, and model artifacts
+- `docs/` architecture notes
+
 ## Notes
-- Activate the venv first:
-  `./venv/Scripts/Activate.ps1`
+- If you see a 503 error when scoring, run `python model/train.py` to generate model artifacts.
