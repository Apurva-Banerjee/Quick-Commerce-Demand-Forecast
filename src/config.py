"""Project-wide configuration values."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

RANDOM_STATE = 42
SKUS = ["MILK_1L", "BREAD_WHITE", "BANANA_1KG", "APPLE_1KG", "EGGS_12"]
START_DATE = "2025-01-01"
END_DATE = "2025-03-31 23:00:00"
