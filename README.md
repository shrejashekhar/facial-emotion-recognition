# Facial Emotion Recognition (Real-Time, Industry-Grade)

A **production-quality Facial Emotion Recognition system** built from scratch using Python 3.10, TensorFlow, MTCNN, and OpenCV.
The system supports **real-time webcam emotion recognition** with high accuracy and high FPS on CPU using an optimized **detect–track–classify** pipeline.

---

## 🚀 Key Features

* **Real-time emotion recognition via webcam**
* **High-accuracy face detection** using MTCNN
* **High-FPS tracking** using OpenCV CSRT tracker
* **CNN-based emotion classifier** trained on FER-2013
* **Optimized CPU performance** (no GPU required)
* **Clean, modular, industry-grade architecture**
* Windows-compatible, Python 3.10, pinned dependencies

---

## 🧠 System Architecture

```
Camera Frame
   │
   ▼
MTCNN Face Detection (periodic)
   │
   ▼
CSRT Face Tracking (per frame)
   │
   ▼
Face Preprocessing (48×48, grayscale, normalized)
   │
   ▼
CNN Emotion Classifier
   │
   ▼
Emotion + Confidence (Live Overlay)
```

**Key optimization**: Face detection and emotion inference are rate-limited, while tracking runs every frame to achieve real-time performance.

---

## 📁 Project Structure

```
facial-emotion-recognition/
│
├── src/
│   ├── api/                  # (Optional) FastAPI layer
│   ├── services/             # Core ML services
│   │   ├── face_detector.py
│   │   ├── preprocessor.py
│   │   ├── emotion_classifier.py
│   │
│   ├── models/               # CNN model definition
│   │   └── emotion_cnn.py
│   │
│   ├── utils/                # Utilities (logging, loaders)
│   ├── core/                 # Constants, sanity checks
│   └── app_live_camera.py    # Real-time webcam app
│
├── data/                     # FER-2013 dataset (not committed)
├── artifacts/                # Trained models, logs
├── requirements.txt
├── requirements.lock.txt
├── README.md
└── venv/
```

---

## 🧪 Dataset

* **FER-2013 (Facial Expression Recognition)**
* 7 emotion classes:

  * Angry
  * Disgust
  * Fear
  * Happy
  * Sad
  * Surprise
  * Neutral

Dataset directory structure (not included in repo):

```
data/fer2013/
├── train/
│   ├── angry/
│   ├── happy/
│   └── ...
└── val/
    ├── angry/
    ├── happy/
    └── ...
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/shrejashekhar/facial-emotion-recognition.git
cd facial-emotion-recognition
```

---

### 2️⃣ Create Virtual Environment (Python 3.10)

```bash
python -m venv venv
venv\Scripts\activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

All dependencies are **version-pinned** for Windows stability.

---

## 🎥 Run Real-Time Emotion Recognition

```bash
python -m src.app_live_camera
```

### Controls

* Webcam opens automatically
* Emotion + confidence displayed on detected face
* Press **`q`** to quit

---

## 📊 Performance Characteristics

* **Face Detection**: MTCNN (high accuracy)
* **Tracking**: CSRT (high stability)
* **Inference Rate**: Emotion predicted every N frames
* **FPS**: ~20–30 FPS on CPU (machine dependent)

---

## 🧩 Tech Stack

**Programming Language**

* Python 3.10

**Deep Learning**

* TensorFlow / Keras

**Computer Vision**

* OpenCV (contrib)
* MTCNN

**Model**

* Custom CNN (FER-2013 style)

**Tools**

* FastAPI (optional deployment)
* NumPy, Pillow

---

## 📌 Resume-Ready Highlights

* Built a **real-time facial emotion recognition system** with CNNs and MTCNN, achieving high accuracy and real-time performance on CPU.
* Designed a **detect–track–classify pipeline** using CSRT tracking to reduce inference latency and improve FPS by ~3×.
* Implemented a **modular, production-grade ML architecture** suitable for deployment via REST APIs.

---

## 🔮 Future Improvements

* Multi-face tracking with unique IDs
* MobileNet-based emotion classifier
* Model quantization for faster inference
* FastAPI + Docker deployment
* Web or mobile frontend

---

## 📄 License

This project is for **educational and portfolio purposes**.

---

## 🙌 Acknowledgements

* FER-2013 Dataset
* TensorFlow & OpenCV communities
