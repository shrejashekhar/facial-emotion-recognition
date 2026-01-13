from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "artifacts" / "models"
LOG_DIR = BASE_DIR / "artifacts" / "logs"

for directory in [DATA_DIR, MODEL_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (48, 48)
NUM_CHANNELS = 1
NUM_CLASSES = 7

DEBUG = True
