from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_MODELS = PROJECT_ROOT / "artifacts" / "models"
ARTIFACTS_REPORTS = PROJECT_ROOT / "artifacts" / "reports"

STOCK_CODE = "600519"
START_DATE = "20180101"
END_DATE = "20251231"
