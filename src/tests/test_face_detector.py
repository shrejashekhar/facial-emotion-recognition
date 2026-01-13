from pathlib import Path

from src.services.face_detector import FaceDetector
from src.utils.image_loader import load_image

TEST_IMAGE = Path("tests") / "test_face.jpg"

def main():
    detector = FaceDetector()
    image = load_image(TEST_IMAGE)
    faces = detector.detect_faces(image)

    print(f"Faces detected: {len(faces)}")

    for idx, face in enumerate(faces):
        print(f"Face {idx+1}: shape={face.shape}")

if __name__ == "__main__":
    main()
