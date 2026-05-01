"""Analysis orchestration service."""

from __future__ import annotations

import uuid
from typing import Any

from app.config import Settings
from app.services.data_service import DataService
from app.services.llm_service import LLMService
from app.services.ml_service import MLService
from app.services.report_service import ReportService


class AnalysisService:
    """Coordinate data, ML, LLM, and reporting layers."""

    def __init__(self, settings: Settings) -> None:
        self.data_service = DataService(settings)
        self.ml_service = MLService()
        self.llm_service = LLMService(settings)
        self.report_service = ReportService(settings)

    def run(self, dataset_id: str, classification_target: str, regression_target: str) -> dict[str, Any]:
        """Run a complete decision intelligence analysis."""

        raw_df = self.data_service.load_dataset(dataset_id)
        df = self.data_service.preprocess(raw_df)
        data_summary = self.data_service.summarize(df)
        ml_result = self.ml_service.run_analysis(df, classification_target, regression_target)
        llm_insights = self.llm_service.generate_insights(
            data_summary=data_summary,
            metrics=ml_result.metrics,
            model_outputs=ml_result.model_outputs,
            feature_importance=ml_result.feature_importance,
        )

        analysis_id = uuid.uuid4().hex
        result = {
            "analysis_id": analysis_id,
            "dataset_id": dataset_id,
            "metrics": ml_result.metrics,
            "feature_importance": ml_result.feature_importance,
            "data_summary": data_summary,
            "model_outputs": ml_result.model_outputs,
            "llm_insights": llm_insights,
            "report_path": "",
        }
        result["report_path"] = self.report_service.save(analysis_id, result)
        return result

