"""Data loading and validation for the car insurance dataset."""

from pathlib import Path

import pandas as pd


RAW_DATA_PATH = Path(__file__).parent.parent / "data" / "raw" / "car_insurance.csv"

TARGET_COL = "outcome"
DROP_COLS = ["id", "postal_code"]

EXPECTED_COLUMNS = [
    "id", "age", "gender", "driving_experience", "education", "income",
    "credit_score", "vehicle_ownership", "vehicle_year", "married",
    "children", "postal_code", "annual_mileage", "vehicle_type",
    "speeding_violations", "duis", "past_accidents", "outcome",
]

# credit_score and annual_mileage have missing values in the raw CSV
MISSING_COLS = ["credit_score", "annual_mileage"]


def load_data(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV and return a validated DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {path}\n"
            "Expected location: data/raw/car_insurance.csv"
        )
    df = pd.read_csv(path)
    validate_schema(df)
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Raise ValueError if the DataFrame is missing required columns or has bad target values."""
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")

    unexpected_target = set(df[TARGET_COL].dropna().unique()) - {0, 1, 0.0, 1.0}
    if unexpected_target:
        raise ValueError(
            f"Target column '{TARGET_COL}' contains unexpected values: {unexpected_target}"
        )


def get_feature_target_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) — drops identifier columns and separates the target."""
    cols_to_drop = [TARGET_COL] + [c for c in DROP_COLS if c in df.columns]
    X = df.drop(columns=cols_to_drop)
    y = df[TARGET_COL].astype(int)
    return X, y


def report_class_balance(y: pd.Series) -> dict[str, float]:
    """Return claim counts and overall claim rate for the target series."""
    n_total = len(y)
    n_claim = int((y == 1).sum())
    n_no_claim = int((y == 0).sum())
    return {
        "n_total": n_total,
        "n_claim": n_claim,
        "n_no_claim": n_no_claim,
        "claim_rate": round(n_claim / n_total, 4) if n_total > 0 else 0.0,
    }
