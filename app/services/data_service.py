"""CSV ingestion, validation, and preprocessing."""

from __future__ import annotations

import uuid
from pathlib import Path

import pandas as pd
from fastapi import HTTPException, UploadFile, status

from app.config import Settings


class DataService:
    """Manage uploaded business datasets."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)

    async def save_upload(self, file: UploadFile) -> tuple[str, pd.DataFrame]:
        """Validate and persist an uploaded CSV file."""

        if not file.filename or not file.filename.lower().endswith(".csv"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only CSV uploads are supported.",
            )

        content = await file.read()
        max_bytes = self.settings.max_upload_mb * 1024 * 1024
        if len(content) > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File exceeds {self.settings.max_upload_mb} MB upload limit.",
            )

        dataset_id = uuid.uuid4().hex
        path = self._dataset_path(dataset_id)
        path.write_bytes(content)

        return dataset_id, self.load_dataset(dataset_id)

    def load_dataset(self, dataset_id: str) -> pd.DataFrame:
        """Load a saved dataset by identifier."""

        path = self._dataset_path(dataset_id)
        if not path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset '{dataset_id}' was not found.",
            )

        try:
            df = pd.read_csv(path)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"CSV could not be parsed: {exc}",
            ) from exc

        self._validate_dataframe(df)
        return df

    def load_dataset_from_path(self, path: Path) -> pd.DataFrame:
        """Load and validate a CSV from a local path."""

        try:
            df = pd.read_csv(path)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"CSV could not be parsed: {exc}",
            ) from exc
        self._validate_dataframe(df)
        return df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean missing values and normalize column names."""

        cleaned = df.copy()
        cleaned.columns = [str(col).strip().lower().replace(" ", "_") for col in cleaned.columns]

        for column in cleaned.columns:
            if cleaned[column].isna().all():
                cleaned = cleaned.drop(columns=[column])
                continue
            if pd.api.types.is_numeric_dtype(cleaned[column]):
                cleaned[column] = cleaned[column].fillna(cleaned[column].median())
            else:
                mode = cleaned[column].mode(dropna=True)
                cleaned[column] = cleaned[column].fillna(mode.iloc[0] if not mode.empty else "unknown")

        return cleaned

    def summarize(self, df: pd.DataFrame) -> dict:
        """Build a compact data summary for prompts and API responses."""

        numeric_columns = df.select_dtypes(include="number").columns.tolist()
        categorical_columns = [col for col in df.columns if col not in numeric_columns]

        return {
            "rows": int(len(df)),
            "columns": df.columns.tolist(),
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "missing_values": {col: int(count) for col, count in df.isna().sum().items()},
            "numeric_profile": df[numeric_columns].describe().round(2).to_dict() if numeric_columns else {},
        }

    def _dataset_path(self, dataset_id: str) -> Path:
        safe_id = "".join(ch for ch in dataset_id if ch.isalnum() or ch in {"-", "_"})
        return self.settings.data_dir / f"{safe_id}.csv"

    @staticmethod
    def _validate_dataframe(df: pd.DataFrame) -> None:
        if df.empty:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="CSV is empty.")
        if len(df.columns) < 3:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="CSV must contain at least three columns for meaningful analysis.",
            )
