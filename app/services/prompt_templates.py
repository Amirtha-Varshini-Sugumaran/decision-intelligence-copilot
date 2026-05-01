"""Prompt templates for deterministic business reporting."""

from __future__ import annotations

import json
from typing import Any


def _json_block(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, default=str)


def insight_generation_prompt(data_summary: dict[str, Any], model_outputs: dict[str, Any]) -> str:
    """Create the insight generation prompt."""

    return f"""
Context:
You are a Decision Intelligence Copilot for business leaders. You convert validated data summaries and model outputs into concise operational insights.

Data Summary:
{_json_block(data_summary)}

Model Outputs:
{_json_block(model_outputs)}

Instructions:
- Use only the provided data and model outputs.
- Do not invent external benchmarks, causes, or financial values.
- Write in a professional business tone.
- Return JSON with keys: executive_summary, key_insights, confidence_notes.
- key_insights must be a list of short, specific observations.
""".strip()


def risk_explanation_prompt(model_outputs: dict[str, Any], feature_importance: dict[str, Any]) -> str:
    """Create the risk explanation prompt."""

    return f"""
Context:
You explain model-driven risk signals to non-technical business stakeholders.

Model Outputs:
{_json_block(model_outputs)}

Feature Importance:
{_json_block(feature_importance)}

Instructions:
- Use only the provided model outputs and feature importance.
- Explain what appears to influence risk in plain business language.
- Avoid claiming causality; describe associations and model signals.
- Return JSON with keys: risk_level, drivers, limitations.
- drivers must connect important features to likely business interpretation.
""".strip()


def recommendation_prompt(data_summary: dict[str, Any], metrics: dict[str, Any], model_outputs: dict[str, Any]) -> str:
    """Create the recommendation prompt."""

    return f"""
Context:
You recommend practical next actions based on business data and model performance.

Data Summary:
{_json_block(data_summary)}

Model Metrics:
{_json_block(metrics)}

Model Outputs:
{_json_block(model_outputs)}

Instructions:
- Use only the supplied data, metrics, and outputs.
- Provide recommendations that a single business team could act on in 30 days.
- Include measurement ideas for follow-up.
- Keep tone direct, controlled, and professional.
- Return JSON with keys: recommended_actions, expected_business_value, measurement_plan.
""".strip()

