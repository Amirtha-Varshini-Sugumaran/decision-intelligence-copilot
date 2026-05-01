"""Example client for the Decision Intelligence Copilot API."""

import requests


BASE_URL = "http://127.0.0.1:8000"


def main() -> None:
    """Upload the sample dataset, run analysis, and fetch the report."""

    with open("data/sample_business_data.csv", "rb") as csv_file:
        upload = requests.post(
            f"{BASE_URL}/upload-data",
            files={"file": ("sample_business_data.csv", csv_file, "text/csv")},
            timeout=30,
        )
    upload.raise_for_status()
    dataset_id = upload.json()["dataset_id"]
    print("Uploaded dataset:", dataset_id)

    analysis = requests.post(
        f"{BASE_URL}/run-analysis",
        json={
            "dataset_id": dataset_id,
            "classification_target": "churn_risk",
            "regression_target": "next_month_revenue",
        },
        timeout=120,
    )
    analysis.raise_for_status()
    analysis_id = analysis.json()["analysis_id"]
    print("Analysis:", analysis_id)
    print("Metrics:", analysis.json()["metrics"])

    report = requests.get(f"{BASE_URL}/get-report/{analysis_id}", timeout=30)
    report.raise_for_status()
    print(report.json()["report"])


if __name__ == "__main__":
    main()

