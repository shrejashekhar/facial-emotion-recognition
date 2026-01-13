import numpy as np
from tensorflow import keras

from src.config.settings import MODEL_DIR
from src.core.constants import EMOTIONS
from src.services.preprocessor import FacePreprocessor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EmotionClassifier:
    def __init__(self):
        model_path = MODEL_DIR / "emotion_cnn"
        self.model = keras.models.load_model(model_path)
        self.preprocessor = FacePreprocessor()
        logger.info("EmotionClassifier initialized")

    def predict(self, face: np.ndarray) -> dict:
        """
        Input:
            face: BGR face image
        Output:
            {
              "emotion": str,
              "confidence": float,
              "probabilities": {emotion: prob}
            }
        """
        tensor = self.preprocessor.preprocess(face)
        tensor = np.expand_dims(tensor, axis=0)

        probs = self.model.predict(tensor, verbose=0)[0]

        idx = int(np.argmax(probs))
        emotion = EMOTIONS[idx]
        confidence = float(probs[idx])

        return {
            "emotion": emotion,
            "confidence": confidence,
            "probabilities": {
                EMOTIONS[i]: float(probs[i]) for i in range(len(EMOTIONS))
            }
        }
