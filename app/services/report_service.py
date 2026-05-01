"""Report persistence and formatting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import HTTPException, status

from app.config import Settings


class ReportService:
    """Create and retrieve decision reports."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.settings.reports_dir.mkdir(parents=True, exist_ok=True)

    def save(self, analysis_id: str, result: dict[str, Any]) -> str:
        """Persist the structured result and generated report body."""

        json_path = self._json_path(analysis_id)
        report_path = self._report_path(analysis_id)
        json_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        report_path.write_text(self.render_markdown(result), encoding="utf-8")
        return str(report_path)

    def get(self, analysis_id: str) -> tuple[str, dict[str, Any]]:
        """Load a stored report and its machine-readable result."""

        json_path = self._json_path(analysis_id)
        report_path = self._report_path(analysis_id)
        if not json_path.exists() or not report_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Report '{analysis_id}' was not found.",
            )
        return report_path.read_text(encoding="utf-8"), json.loads(json_path.read_text(encoding="utf-8"))

    @staticmethod
    def render_markdown(result: dict[str, Any]) -> str:
        """Render a concise decision report for business readers."""

        llm = result["llm_insights"]
        insight = llm.get("insight_generation", {})
        risk = llm.get("risk_explanation", {})
        recs = llm.get("recommendations", {})

        key_insights = "\n".join(f"- {item}" for item in insight.get("key_insights", []))
        drivers = "\n".join(f"- {item}" for item in risk.get("drivers", []))
        actions = "\n".join(f"- {item}" for item in recs.get("recommended_actions", []))

        return f"""# Decision Intelligence Report

## Executive Summary
{insight.get("executive_summary", "No executive summary returned.")}

## Model Performance
- Classification accuracy: {result["metrics"]["classification_accuracy"]}
- Regression RMSE: {result["metrics"]["regression_rmse"]}

## Key Insights
{key_insights or "- No key insights returned."}

## Risk Explanation
- Risk level: {risk.get("risk_level", "not specified")}

### Main Drivers
{drivers or "- No risk drivers returned."}

### Limitations
{risk.get("limitations", "Model outputs should support human judgment and not replace it.")}

## Recommendations
{actions or "- No recommendations returned."}

## Expected Business Value
{recs.get("expected_business_value", "Not specified.")}

## Measurement Plan
{recs.get("measurement_plan", "Not specified.")}
"""

    def _json_path(self, analysis_id: str) -> Path:
        return self.settings.reports_dir / f"{analysis_id}.json"

    def _report_path(self, analysis_id: str) -> Path:
        return self.settings.reports_dir / f"{analysis_id}.md"
