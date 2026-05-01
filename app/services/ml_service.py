"""Machine learning analysis service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from fastapi import HTTPException, status
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class MLAnalysisResult:
    """Trained model outputs and evaluation results."""

    metrics: dict[str, float]
    feature_importance: dict[str, Any]
    model_outputs: dict[str, Any]


class MLService:
    """Train classification and regression models on business data."""

    def run_analysis(
        self,
        df: pd.DataFrame,
        classification_target: str,
        regression_target: str,
    ) -> MLAnalysisResult:
        """Run classification, regression, feature selection, and evaluation."""

        classification_target = self._normalize_column(classification_target)
        regression_target = self._normalize_column(regression_target)
        self._validate_targets(df, classification_target, regression_target)

        feature_df = df.drop(columns=[classification_target, regression_target])
        if feature_df.empty:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one feature column is required after removing target columns.",
            )

        class_result = self._train_classifier(feature_df, df[classification_target])
        reg_result = self._train_regressor(feature_df, df[regression_target])

        return MLAnalysisResult(
            metrics={
                "classification_accuracy": round(class_result["accuracy"], 4),
                "regression_rmse": round(reg_result["rmse"], 4),
            },
            feature_importance={
                "classification": class_result["feature_importance"],
                "regression": reg_result["feature_importance"],
                "method": "Random forest feature importance after preprocessing",
            },
            model_outputs={
                "classification": class_result["outputs"],
                "regression": reg_result["outputs"],
                "selected_features": {
                    "classification": class_result["selected_features"],
                    "regression": reg_result["selected_features"],
                },
            },
        )

    def _train_classifier(self, x: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
        if y.nunique(dropna=True) < 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Classification target must contain at least two classes.",
            )

        stratify = y if y.value_counts().min() >= 2 else None
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.25, random_state=42, stratify=stratify
        )
        preprocessor = self._build_preprocessor(x)
        encoded_train = preprocessor.fit_transform(x_train)
        encoded_test = preprocessor.transform(x_test)
        names = self._feature_names(preprocessor)

        selected_train, selected_test, selected_names = self._select_features(
            encoded_train, encoded_test, y_train, names, task="classification"
        )

        model = RandomForestClassifier(n_estimators=160, random_state=42, class_weight="balanced")
        model.fit(selected_train, y_train)
        predictions = model.predict(selected_test)
        probabilities = self._positive_probabilities(model, selected_test)

        return {
            "accuracy": accuracy_score(y_test, predictions),
            "feature_importance": self._top_importances(selected_names, model.feature_importances_),
            "selected_features": selected_names,
            "outputs": {
                "target_classes": [str(item) for item in model.classes_.tolist()],
                "sample_predictions": [str(item) for item in predictions[:10].tolist()],
                "average_risk_score": round(float(np.mean(probabilities)), 4),
                "high_risk_count": int(np.sum(probabilities >= 0.7)),
            },
        }

    def _train_regressor(self, x: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
        if not pd.api.types.is_numeric_dtype(y):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Regression target must be numeric.",
            )

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
        preprocessor = self._build_preprocessor(x)
        encoded_train = preprocessor.fit_transform(x_train)
        encoded_test = preprocessor.transform(x_test)
        names = self._feature_names(preprocessor)

        selected_train, selected_test, selected_names = self._select_features(
            encoded_train, encoded_test, y_train, names, task="regression"
        )

        model = RandomForestRegressor(n_estimators=160, random_state=42)
        model.fit(selected_train, y_train)
        predictions = model.predict(selected_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))

        return {
            "rmse": rmse,
            "feature_importance": self._top_importances(selected_names, model.feature_importances_),
            "selected_features": selected_names,
            "outputs": {
                "prediction_mean": round(float(np.mean(predictions)), 2),
                "prediction_min": round(float(np.min(predictions)), 2),
                "prediction_max": round(float(np.max(predictions)), 2),
                "sample_predictions": [round(float(item), 2) for item in predictions[:10]],
            },
        }

    @staticmethod
    def _build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
        numeric_features = x.select_dtypes(include="number").columns.tolist()
        categorical_features = [col for col in x.columns if col not in numeric_features]

        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )

        return ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, numeric_features),
                ("cat", categorical_pipeline, categorical_features),
            ],
            remainder="drop",
        )

    @staticmethod
    def _feature_names(preprocessor: ColumnTransformer) -> list[str]:
        try:
            return preprocessor.get_feature_names_out().tolist()
        except Exception:
            return [f"feature_{index}" for index in range(preprocessor.transformers_[0][2])]

    @staticmethod
    def _select_features(
        x_train: np.ndarray,
        x_test: np.ndarray,
        y_train: pd.Series,
        names: list[str],
        task: str,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        k = min(8, x_train.shape[1])
        score_func = f_classif if task == "classification" else f_regression
        selector = SelectKBest(score_func=score_func, k=k)
        selected_train = selector.fit_transform(x_train, y_train)
        selected_test = selector.transform(x_test)
        selected_names = [name for name, keep in zip(names, selector.get_support()) if keep]
        return selected_train, selected_test, selected_names

    @staticmethod
    def _positive_probabilities(model: RandomForestClassifier, x_test: np.ndarray) -> np.ndarray:
        probabilities = model.predict_proba(x_test)
        if probabilities.shape[1] == 1:
            return probabilities[:, 0]
        return probabilities[:, -1]

    @staticmethod
    def _top_importances(names: list[str], importances: np.ndarray) -> dict[str, float]:
        pairs = sorted(zip(names, importances), key=lambda item: item[1], reverse=True)[:8]
        return {name: round(float(value), 4) for name, value in pairs}

    @staticmethod
    def _normalize_column(column: str) -> str:
        return column.strip().lower().replace(" ", "_")

    @staticmethod
    def _validate_targets(df: pd.DataFrame, classification_target: str, regression_target: str) -> None:
        missing = [target for target in [classification_target, regression_target] if target not in df.columns]
        if missing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Target columns not found after preprocessing: {missing}",
            )
        if classification_target == regression_target:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Classification and regression targets must be different columns.",
            )
