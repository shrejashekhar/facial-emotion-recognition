from pathlib import Path

from src.services.face_detector import FaceDetector
from src.services.emotion_classifier import EmotionClassifier
from src.utils.image_loader import load_image

TEST_IMAGE = Path("tests") / "test_face.jpg"

def main():
    detector = FaceDetector()
    classifier = EmotionClassifier()

    image = load_image(TEST_IMAGE)
    faces = detector.detect_faces(image)

    for i, face in enumerate(faces):
        result = classifier.predict(face)
        print(f"Face {i+1}: {result}")

if __name__ == "__main__":
    main()
