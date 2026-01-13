from pathlib import Path

from src.services.face_detector import FaceDetector
from src.services.preprocessor import FacePreprocessor
from src.utils.image_loader import load_image

TEST_IMAGE = Path("tests") / "test_face.jpg"

def main():
    detector = FaceDetector()
    preprocessor = FacePreprocessor()

    image = load_image(TEST_IMAGE)
    faces = detector.detect_faces(image)

    for idx, face in enumerate(faces):
        tensor = preprocessor.preprocess(face)
        print(f"Face {idx+1} tensor shape: {tensor.shape}")
        print(f"Min: {tensor.min()}, Max: {tensor.max()}")

if __name__ == "__main__":
    main()
