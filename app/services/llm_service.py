"""OpenAI-powered business insight generation."""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import OpenAI

from app.config import Settings
from app.services.prompt_templates import (
    insight_generation_prompt,
    recommendation_prompt,
    risk_explanation_prompt,
)

logger = logging.getLogger(__name__)


class LLMService:
    """Generate structured business insights from model outputs."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None

    def generate_insights(
        self,
        data_summary: dict[str, Any],
        metrics: dict[str, Any],
        model_outputs: dict[str, Any],
        feature_importance: dict[str, Any],
    ) -> dict[str, Any]:
        """Run three prompt templates and return structured insights."""

        prompts = {
            "insight_generation": insight_generation_prompt(data_summary, model_outputs),
            "risk_explanation": risk_explanation_prompt(model_outputs, feature_importance),
            "recommendations": recommendation_prompt(data_summary, metrics, model_outputs),
        }

        if not self.client:
            logger.warning("OPENAI_API_KEY not configured; returning deterministic local fallback insights.")
            return self._fallback_insights(metrics, model_outputs, feature_importance)

        results = {}
        for name, prompt in prompts.items():
            results[name] = self._call_openai(prompt)
        return results

    def _call_openai(self, prompt: str) -> dict[str, Any]:
        response = self.client.responses.create(
            model=self.settings.openai_model,
            input=[
                {
                    "role": "system",
                    "content": "You are a precise business analyst. Return valid JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        text = response.output_text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("OpenAI response was not valid JSON; wrapping raw text.")
            return {"raw_output": text}

    @staticmethod
    def _fallback_insights(
        metrics: dict[str, Any],
        model_outputs: dict[str, Any],
        feature_importance: dict[str, Any],
    ) -> dict[str, Any]:
        class_importance = list(feature_importance.get("classification", {}).keys())[:3]
        reg_importance = list(feature_importance.get("regression", {}).keys())[:3]
        risk_score = model_outputs.get("classification", {}).get("average_risk_score", 0)

        return {
            "insight_generation": {
                "executive_summary": "The analysis identified measurable risk and performance signals from the uploaded dataset.",
                "key_insights": [
                    f"Classification accuracy was {metrics.get('classification_accuracy')}.",
                    f"Regression RMSE was {metrics.get('regression_rmse')}.",
                    f"Average modeled risk score was {risk_score}.",
                ],
                "confidence_notes": "Fallback report generated without OpenAI because OPENAI_API_KEY is not configured.",
            },
            "risk_explanation": {
                "risk_level": "elevated" if risk_score >= 0.5 else "moderate",
                "drivers": class_importance or ["No dominant classification drivers were available."],
                "limitations": "Feature importance indicates model signal strength, not causality.",
            },
            "recommendations": {
                "recommended_actions": [
                    "Review the highest-risk segment first and compare it with recent operational events.",
                    "Create a monitoring view for the top classification and regression drivers.",
                    "Collect additional labeled outcomes before expanding automated decisions.",
                ],
                "expected_business_value": "Better prioritization of intervention effort and clearer visibility into demand or revenue drivers.",
                "measurement_plan": "Track risk score movement, actual outcomes, and forecast error over the next operating cycle.",
            },
        }

