# Decision Intelligence Copilot

FastAPI service for decision intelligence workflows. The application ingests structured business data, trains predictive models, explains model drivers, and uses controlled OpenAI prompts to produce structured business recommendations.

## Problem Statement

Business teams often keep sales, customer, and operations data in spreadsheets or CSV exports. This service provides a repeatable API workflow for converting that data into model-backed analysis:

- Machine-readable JSON analysis
- Classification risk signals, such as churn risk
- Regression forecasts, such as next-month revenue
- Interpretable feature importance
- Structured insight, risk explanation, and recommendation payloads

## Architecture

```text
app/
  api/          FastAPI routes
  models/       Pydantic request and response schemas
  services/     Data ingestion, ML, LLM, prompts, reports
  utils/        Logging helpers
data/           Example dataset and uploaded CSV files
reports/        Runtime analysis artifacts, excluded from version control
tests/          Focused service tests
```

Flow:

1. `/upload-data` accepts a CSV file and validates basic shape.
2. `/run-analysis` preprocesses data with pandas, trains scikit-learn classification and regression models, selects features, evaluates accuracy/RMSE, extracts feature importance, and generates OpenAI-powered insights.
3. `/get-report/{analysis_id}` returns the stored analysis artifact and generated report body for the requested run.

If `OPENAI_API_KEY` is not configured, the system returns deterministic local fallback insights. This keeps ingestion, preprocessing, model training, and report persistence available in offline environments.

## Prompt Engineering

The prompt layer includes three structured templates:

- Insight Generation Prompt
- Risk Explanation Prompt
- Recommendation Prompt

Each template includes context, data summary or model outputs, and explicit instructions to use only supplied data, avoid unsupported claims, and return structured JSON in a professional business tone.

## How To Run

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Set environment variables:

```powershell
Copy-Item .env.example .env
# Add OPENAI_API_KEY to .env to enable OpenAI-backed recommendations.
```

Start the API:

```powershell
uvicorn app.main:app --reload
```

Open:

- API docs: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`

## Example API Usage

Upload the included dataset:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/upload-data" `
  -F "file=@data/sample_business_data.csv"
```

Run analysis:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/run-analysis" `
  -H "Content-Type: application/json" `
  -d "{\"dataset_id\":\"<DATASET_ID>\",\"classification_target\":\"churn_risk\",\"regression_target\":\"next_month_revenue\"}"
```

Fetch stored analysis:

```powershell
curl.exe "http://127.0.0.1:8000/get-report/<ANALYSIS_ID>"
```

Or use:

```powershell
python example_api_usage.py
```

## Sample Output

Example JSON fields:

```json
{
  "metrics": {
    "classification_accuracy": 0.9,
    "regression_rmse": 288.52
  },
  "feature_importance": {
    "method": "Random forest feature importance after preprocessing"
  },
  "llm_insights": {
    "insight_generation": {},
    "risk_explanation": {},
    "recommendations": {}
  }
}
```

## Business Value

The service supports retention analysis, revenue planning, operating reviews, and prioritization workflows. It keeps model outputs explainable and stores each run as an auditable analysis artifact.

## Docker

```powershell
docker build -t decision-intelligence-copilot .
docker run --env-file .env -p 8000:8000 decision-intelligence-copilot
```

## Testing

```powershell
pytest
```
