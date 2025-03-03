from pathlib import Path

PARENT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PARENT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRANSFORMED_DATA_DIR = DATA_DIR / "transformed"

for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, TRANSFORMED_DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
