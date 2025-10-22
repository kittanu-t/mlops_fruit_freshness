# 02_data_preprocessing.py
import os, json
from pathlib import Path
import mlflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator

TRAIN_DIR = "data_splits/train"
VAL_DIR   = "data_splits/val"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EXPERIMENT = "Fruits Freshness - Preprocessing"

def run():
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="Preprocessing"):
        mlflow.set_tag("ml.step", "preprocessing")
        mlflow.log_param("image_size", IMAGE_SIZE)
        mlflow.log_param("batch_size", BATCH_SIZE)

        train_gen = ImageDataGenerator(rescale=1./255)
        val_gen   = ImageDataGenerator(rescale=1./255)

        tg = train_gen.flow_from_directory(TRAIN_DIR, target_size=IMAGE_SIZE,
                                           batch_size=BATCH_SIZE, class_mode="categorical", shuffle=True)
        vg = val_gen.flow_from_directory(VAL_DIR, target_size=IMAGE_SIZE,
                                         batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False)

        mlflow.log_metric("train_samples", tg.samples)
        mlflow.log_metric("val_samples", vg.samples)
        mlflow.log_param("num_classes", tg.num_classes)

        os.makedirs("processed_metadata", exist_ok=True)
        with open("processed_metadata/class_indices.json","w") as f:
            json.dump(tg.class_indices, f, indent=2)
        mlflow.log_artifacts("processed_metadata", artifact_path="processed_metadata")
        print("Saved class_indices.json")

if __name__ == "__main__":
    run()
