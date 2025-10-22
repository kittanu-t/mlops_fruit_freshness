#!/usr/bin/env python3
import glob
import json
from pathlib import Path
from PIL import Image
import mlflow

DATA_ROOT = Path("data_splits/train")
EXPERIMENT = "Fruits Freshness - Data Validation"


def _is_ok(path_str: str) -> bool:
    try:
        Image.open(path_str).verify()
        return True
    except Exception:
        return False


def validate() -> None:
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

        exts = ("*.jpg", "*.jpeg", "*.png")
        report = []
        total = 0
        total_bad = 0

        for cls in classes:
            files = []
            for e in exts:
                files += glob.glob(str(DATA_ROOT / cls / e))
            ok = sum(1 for f in files if _is_ok(f))
            bad = len(files) - ok
            total += ok
            total_bad += bad
            report.append({"class": cls, "count": ok, "bad": bad})

        mlflow.log_metric("images_train", total)
        mlflow.log_metric("corrupt_images", total_bad)
        mlflow.log_text(json.dumps(report, indent=2), "reports/class_counts.json")
        mlflow.set_tag("status", "success")
        print("Classes:", len(classes), "| images:", total, "| corrupt:", total_bad)


if __name__ == "__main__":
    validate()
