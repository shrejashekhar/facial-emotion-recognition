import cv2
import numpy as np

from src.services.face_detector import FaceDetector
from src.services.emotion_classifier import EmotionClassifier
from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_tracker():
    # CSRT = high accuracy, still fast enough
    return cv2.TrackerCSRT_create()


def main():
    face_detector = FaceDetector()
    emotion_classifier = EmotionClassifier()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    logger.info("Webcam started. Press 'q' to quit.")

    tracker = None
    current_emotion = None
    current_confidence = None

    frame_count = 0
    DETECT_EVERY = 15        # MTCNN frequency
    EMOTION_EVERY = 10       # Emotion inference frequency

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Step 1: Detect face periodically OR if tracker lost
        if tracker is None or frame_count % DETECT_EVERY == 0:
            detections = face_detector.detector.detect_faces(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )

            if detections:
                det = max(detections, key=lambda d: d["confidence"])
                if det["confidence"] > 0.9:
                    x, y, w, h = det["box"]
                    x, y = max(0, x), max(0, y)

                    tracker = create_tracker()
                    tracker.init(frame, (x, y, w, h))

                    current_emotion = None
                    current_confidence = None

        # Step 2: Track face
        if tracker is not None:
            success, box = tracker.update(frame)

            if success:
                x, y, w, h = map(int, box)
                face = frame[y:y + h, x:x + w]

                # Step 3: Emotion inference periodically
                if frame_count % EMOTION_EVERY == 0 and face.size != 0:
                    prediction = emotion_classifier.predict(face)
                    current_emotion = prediction["emotion"]
                    current_confidence = prediction["confidence"]

                # Draw
                label = (
                    f"{current_emotion} ({current_confidence:.2f})"
                    if current_emotion else "Detecting..."
                )

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )
            else:
                tracker = None

        cv2.imshow("Live Facial Emotion Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Webcam stopped.")


if __name__ == "__main__":
    main()
