import cv2
import numpy as np

from src.config.settings import IMAGE_SIZE, NUM_CHANNELS
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FacePreprocessor:
    def preprocess(self, face: np.ndarray) -> np.ndarray:
        """
        Input:
            face: BGR face image
        Output:
            Preprocessed face tensor (H, W, C)
        """
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, IMAGE_SIZE)

        normalized = resized.astype("float32") / 255.0

        if NUM_CHANNELS == 1:
            normalized = np.expand_dims(normalized, axis=-1)

        logger.info(f"Preprocessed face shape: {normalized.shape}")
        return normalized
