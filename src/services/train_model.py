from pathlib import Path

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.config.settings import (
    IMAGE_SIZE,
    NUM_CHANNELS,
    NUM_CLASSES,
    MODEL_DIR,
)
from src.models.emotion_cnn import build_emotion_cnn
from src.utils.logger import get_logger

logger = get_logger(__name__)


def train():
    data_dir = Path("data/fer2013")

    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = datagen.flow_from_directory(
        train_dir,
        target_size=IMAGE_SIZE,
        color_mode="grayscale",
        batch_size=64,
        class_mode="categorical",
    )

    val_gen = datagen.flow_from_directory(
        val_dir,
        target_size=IMAGE_SIZE,
        color_mode="grayscale",
        batch_size=64,
        class_mode="categorical",
    )

    model = build_emotion_cnn(
        input_shape=(*IMAGE_SIZE, NUM_CHANNELS),
        num_classes=NUM_CLASSES,
    )

    model.summary(print_fn=logger.info)

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=30,
    )

    save_path = MODEL_DIR / "emotion_cnn"
    model.save(save_path)

    logger.info(f"Model saved to {save_path}")


if __name__ == "__main__":
    train()
