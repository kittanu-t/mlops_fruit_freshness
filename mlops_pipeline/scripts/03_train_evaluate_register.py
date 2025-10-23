#!/usr/bin/env python3
"""
CPU-friendly version: Transfer Learning (MobileNetV2)
- Feature extraction + Fine-tuning (2-phase)
- EarlyStopping + ReduceLROnPlateau
- Logs + Registers to MLflow
"""

import numpy as np
import mlflow
import mlflow.tensorflow
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras import Model
import tensorflow as tf

# ======== CONFIG ======== #
TRAIN_DIR = "data_splits/train"
VAL_DIR = "data_splits/val"
IMAGE_SIZE = (160, 160)      # ลดขนาดลงเพื่อให้ CPU เร็วขึ้น
BATCH_SIZE = 16
EPOCHS_STAGE1 = 10
EPOCHS_STAGE2 = 3
THRESH_ACC = 0.85
THRESH_F1W = 0.85
EXPERIMENT = "Fruits Freshness - Model Training (CPU)"
MODEL_NAME = "fruits-freshness-classifier"
# ======================== #


def build_model(input_shape, num_classes):
    """MobileNetV2 transfer learning setup"""
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model, base_model


def evaluate_model(model, vg):
    """ประเมิน accuracy และ f1"""
    vg.reset()
    y_true, y_pred = [], []
    for _ in range(len(vg)):
        x_batch, y_batch = next(vg)
        pred = model.predict(x_batch, verbose=0)
        y_true.extend(np.argmax(y_batch, axis=1))
        y_pred.extend(np.argmax(pred, axis=1))

    acc = accuracy_score(y_true, y_pred)
    f1w = f1_score(y_true, y_pred, average="weighted")
    f1m = f1_score(y_true, y_pred, average="macro")
    return acc, f1w, f1m


def run():
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="MobileNetV2_CPU"):
        mlflow.set_tag("ml.step", "training_evaluation")

        train_gen = ImageDataGenerator(rescale=1.0 / 255.0, rotation_range=15, horizontal_flip=True)
        val_gen = ImageDataGenerator(rescale=1.0 / 255.0)

        tg = train_gen.flow_from_directory(
            TRAIN_DIR,
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            shuffle=True,
        )
        vg = val_gen.flow_from_directory(
            VAL_DIR,
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            shuffle=False,
        )

        input_shape = IMAGE_SIZE + (3,)
        model, base_model = build_model(input_shape, tg.num_classes)

        # === Phase 1: Feature Extraction === #
        early_stop = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-6, verbose=1)

        print("\n=== Phase 1: Feature Extraction (frozen base) ===")
        model.fit(
            tg,
            epochs=EPOCHS_STAGE1,
            validation_data=vg,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )

        acc, f1w, f1m = evaluate_model(model, vg)
        print(f"Phase 1 → ACC={acc:.3f} | F1w={f1w:.3f} | F1m={f1m:.3f}")

        # === Phase 2: Fine-tuning (unfreeze top 10 layers) === #
        print("\n=== Phase 2: Fine-tuning last 10 layers ===")
        for layer in base_model.layers[-10:]:
            layer.trainable = True

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        early_stop_ft = EarlyStopping(monitor="val_loss", patience=1, restore_best_weights=True, verbose=1)
        reduce_lr_ft = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-6, verbose=1)

        model.fit(
            tg,
            epochs=EPOCHS_STAGE2,
            validation_data=vg,
            callbacks=[early_stop_ft, reduce_lr_ft],
            verbose=1
        )

        acc, f1w, f1m = evaluate_model(model, vg)
        print(f"Phase 2 → ACC={acc:.3f} | F1w={f1w:.3f} | F1m={f1m:.3f}")

        # === Log to MLflow === #
        mlflow.log_metrics({
            "val_accuracy": acc,
            "val_f1_weighted": f1w,
            "val_f1_macro": f1m
        })

        mlflow.tensorflow.log_model(model, "model")
        passed = (acc >= THRESH_ACC) or (f1w >= THRESH_F1W)
        mlflow.set_tag("passed_threshold", str(passed))

        if passed:
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
            mlflow.register_model(model_uri, MODEL_NAME)
            print(f"✅ Registered model: {MODEL_NAME}")
        else:
            print("⚠️ Threshold not met → not registered")


if __name__ == "__main__":
    run()
