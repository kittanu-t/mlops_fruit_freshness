# 01_data_validation.py
import os, glob, json
from pathlib import Path
from PIL import Image
import mlflow

DATA_ROOT = Path("data_splits/train")   # ชี้ไป train ของชุดใหม่
EXPERIMENT = "Fruits Freshness - Data Validation"

def is_ok(p):
    try:
        Image.open(p).verify()
        return True
    except Exception:
        return False

def validate():
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="DataValidation"):
        mlflow.set_tag("ml.step", "data_validation")
        mlflow.log_param("data_path", str(DATA_ROOT))

        if not DATA_ROOT.is_dir():
            mlflow.set_tag("status", "fail:not_found")
            print("Data root not found:", DATA_ROOT)
            return

        classes = sorted([d.name for d in DATA_ROOT.iterdir() if d.is_dir()])
        mlflow.log_metric("num_classes", len(classes))

        exts = ("*.jpg","*.jpeg","*.png")
        report = []
        total, total_bad = 0, 0
        for c in classes:
            files = []
            for e in exts:
                files += glob.glob(str(DATA_ROOT/c/e))
            ok = sum(1 for f in files if is_ok(f))
            bad = len(files) - ok
            total += ok
            total_bad += bad
            report.append({"class": c, "count": ok, "bad": bad})

        mlflow.log_metric("images_train", total)
        mlflow.log_metric("corrupt_images", total_bad)
        mlflow.log_text(json.dumps(report, indent=2), "reports/class_counts.json")
        mlflow.set_tag("status", "success")
        print("Classes:", len(classes), "| images:", total, "| corrupt:", total_bad)

if __name__ == "__main__":
    validate()
