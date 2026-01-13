import cv2
import numpy as np
from mtcnn import MTCNN

from src.utils.logger import get_logger

logger = get_logger(__name__)


class FaceDetector:
    def __init__(self, min_confidence: float = 0.90):
        self.detector = MTCNN()
        self.min_confidence = min_confidence
        logger.info("MTCNN FaceDetector initialized")

    def detect_faces(self, image: np.ndarray) -> list[np.ndarray]:
        """
        Input:
            image: BGR image (OpenCV)
        Output:
            List of cropped face images (BGR)
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detections = self.detector.detect_faces(rgb_image)

        faces = []

        for det in detections:
            confidence = det.get("confidence", 0)
            if confidence < self.min_confidence:
                continue

            x, y, w, h = det["box"]
            x, y = max(0, x), max(0, y)

            face = image[y:y + h, x:x + w]

            if face.size == 0:
                continue

            faces.append(face)

        logger.info(f"Detected {len(faces)} face(s)")
        return faces
