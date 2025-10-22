# 03_train_evaluate_register.py
import os, json, numpy as np, mlflow, mlflow.tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import accuracy_score, f1_score

TRAIN_DIR = "data_splits/train"
VAL_DIR   = "data_splits/val"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
THRESH_ACC = 0.85
THRESH_F1W = 0.85
EXPERIMENT = "Fruits Freshness - Model Training"
MODEL_NAME = "fruits-freshness-classifier"

def build_model(input_shape, num_classes):
    return Sequential([
        Conv2D(32, (3,3), activation="relu", input_shape=input_shape),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation="relu"),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax"),
    ])

def run():
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"cnn_e{EPOCHS}"):
        mlflow.set_tag("ml.step","training_evaluation")
        mlflow.log_params({
            "image_size": IMAGE_SIZE, "batch_size": BATCH_SIZE, "epochs": EPOCHS,
            "arch": "simple_cnn_32_64"
        })

        train_gen = ImageDataGenerator(rescale=1./255, rotation_range=20, horizontal_flip=True)
        val_gen   = ImageDataGenerator(rescale=1./255)

        tg = train_gen.flow_from_directory(TRAIN_DIR, target_size=IMAGE_SIZE,
                                           batch_size=BATCH_SIZE, class_mode="categorical", shuffle=True)
        vg = val_gen.flow_from_directory(VAL_DIR, target_size=IMAGE_SIZE,
                                         batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False)

        input_shape = IMAGE_SIZE + (3,)
        model = build_model(input_shape, tg.num_classes)
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        history = model.fit(tg, epochs=EPOCHS, validation_data=vg)  # ให้ Kerasคำนวณ steps เอง

        # Predict ทั้ง validation เพื่อคำนวณ F1
        vg.reset()
        y_true = []
        y_pred = []
        for _ in range(len(vg)):
            x, y = next(vg)
            pred = model.predict(x, verbose=0)
            y_true.extend(np.argmax(y, axis=1))
            y_pred.extend(np.argmax(pred, axis=1))

        acc = accuracy_score(y_true, y_pred)
        f1w = f1_score(y_true, y_pred, average="weighted")
        f1m = f1_score(y_true, y_pred, average="macro")
        mlflow.log_metrics({"val_accuracy": acc, "val_f1_weighted": f1w, "val_f1_macro": f1m})

        # log model
        mlflow.tensorflow.log_model(model, "model")
        passed = (acc >= THRESH_ACC) or (f1w >= THRESH_F1W)
        mlflow.set_tag("passed_threshold", str(passed))

        if passed:
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
            mlflow.register_model(model_uri, MODEL_NAME)
            print(f"Registered {MODEL_NAME}")
        else:
            print("Threshold not met → not registered")
        print(f"ACC={acc:.3f}  F1w={f1w:.3f}  F1m={f1m:.3f}")

if __name__ == "__main__":
    run()
