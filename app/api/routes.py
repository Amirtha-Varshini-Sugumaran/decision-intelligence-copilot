"""FastAPI route definitions."""

from fastapi import APIRouter, Depends, UploadFile

from app.config import Settings, get_settings
from app.models.schemas import AnalysisRequest, AnalysisResponse, ReportResponse, UploadResponse
from app.services.analysis_service import AnalysisService
from app.services.data_service import DataService
from app.services.report_service import ReportService

router = APIRouter()


def get_data_service(settings: Settings = Depends(get_settings)) -> DataService:
    """Create data service dependency."""

    return DataService(settings)


def get_analysis_service(settings: Settings = Depends(get_settings)) -> AnalysisService:
    """Create analysis service dependency."""

    return AnalysisService(settings)


def get_report_service(settings: Settings = Depends(get_settings)) -> ReportService:
    """Create report service dependency."""

    return ReportService(settings)


@router.post("/upload-data", response_model=UploadResponse)
async def upload_data(file: UploadFile, service: DataService = Depends(get_data_service)) -> UploadResponse:
    """Upload and validate business CSV data."""

    dataset_id, df = await service.save_upload(file)
    return UploadResponse(
        dataset_id=dataset_id,
        rows=len(df),
        columns=df.columns.tolist(),
        message="Dataset uploaded and validated successfully.",
    )


@router.post("/run-analysis", response_model=AnalysisResponse)
def run_analysis(
    payload: AnalysisRequest,
    service: AnalysisService = Depends(get_analysis_service),
) -> AnalysisResponse:
    """Train models, generate insights, and save a decision report."""

    result = service.run(
        dataset_id=payload.dataset_id,
        classification_target=payload.classification_target,
        regression_target=payload.regression_target,
    )
    return AnalysisResponse(**result)


@router.get("/get-report/{analysis_id}", response_model=ReportResponse)
def get_report(
    analysis_id: str,
    service: ReportService = Depends(get_report_service),
) -> ReportResponse:
    """Return a saved report and JSON result."""

    report, result = service.get(analysis_id)
    return ReportResponse(analysis_id=analysis_id, report=report, json_result=result)

