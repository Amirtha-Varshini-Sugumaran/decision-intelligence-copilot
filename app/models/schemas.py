"""API and service schemas."""

from typing import Any

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    """Response returned after a successful CSV upload."""

    dataset_id: str
    rows: int
    columns: list[str]
    message: str


class AnalysisRequest(BaseModel):
    """Request body for running a business analysis."""

    dataset_id: str = Field(..., description="Uploaded dataset identifier.")
    classification_target: str = Field(..., description="Binary or categorical target column.")
    regression_target: str = Field(..., description="Numeric target column.")


class ModelMetrics(BaseModel):
    """Model evaluation metrics."""

    classification_accuracy: float
    regression_rmse: float


class FeatureImportance(BaseModel):
    """Feature importance values for a trained model."""

    classification: dict[str, float]
    regression: dict[str, float]
    method: str


class AnalysisResponse(BaseModel):
    """Machine-readable analysis response."""

    analysis_id: str
    dataset_id: str
    metrics: ModelMetrics
    feature_importance: FeatureImportance
    data_summary: dict[str, Any]
    model_outputs: dict[str, Any]
    llm_insights: dict[str, Any]
    report_path: str


class ReportResponse(BaseModel):
    """Stored report response."""

    analysis_id: str
    report: str
    json_result: dict[str, Any]

