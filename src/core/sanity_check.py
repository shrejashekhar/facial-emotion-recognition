from src.config.settings import BASE_DIR, MODEL_DIR
from src.core.constants import EMOTIONS
from src.utils.logger import get_logger

logger = get_logger(__name__)

def run():
    logger.info(f"Base directory: {BASE_DIR}")
    logger.info(f"Model directory: {MODEL_DIR}")
    logger.info(f"Emotions: {EMOTIONS}")

if __name__ == "__main__":
    run()
