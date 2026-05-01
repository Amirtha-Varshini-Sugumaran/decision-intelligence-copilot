"""Focused service tests for the Decision Intelligence Copilot."""

from app.config import Settings
from app.services.analysis_service import AnalysisService
from app.services.data_service import DataService


def test_preprocess_normalizes_columns_and_fills_missing() -> None:
    settings = Settings(data_dir="data", reports_dir="reports")
    service = DataService(settings)
    df = service.load_dataset("sample_business_data")
    df.loc[0, "monthly_spend"] = None

    cleaned = service.preprocess(df)

    assert "monthly_spend" in cleaned.columns
    assert int(cleaned["monthly_spend"].isna().sum()) == 0


def test_full_analysis_generates_report() -> None:
    settings = Settings(data_dir="data", reports_dir="reports")

    result = AnalysisService(settings).run("sample_business_data", "churn_risk", "next_month_revenue")

    assert result["metrics"]["classification_accuracy"] >= 0
    assert result["metrics"]["regression_rmse"] >= 0
    assert result["report_path"].endswith(".md")
