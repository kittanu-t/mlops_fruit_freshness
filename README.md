# Fruits Freshness Classification MLOps Project

## Overview
Complete MLflow-based MLOps pipeline for multiclass fruit freshness classification (10 classes).

## How to Run
1. **Data split**  
   `python mlops_pipeline/scripts/00_split_dataset.py`

2. **Validate → Preprocess → Train → Register**

python mlops_pipeline/scripts/01_data_validation.py
python mlops_pipeline/scripts/02_data_preprocessing.py
python mlops_pipeline/scripts/03_train_evaluate_register.py

3. **Serve API**

4. **Test prediction**

curl -X POST "http://127.0.0.1:8000/predict
" -F "file=@data_splits/test/stale_orange/example.png" #rotated_by_15_Screen Shot 2018-06-12 at 8.53.42 PM.png

## Tech Stack
- Python, TensorFlow/Keras  
- MLflow (Tracking + Model Registry)  
- FastAPI  
- GitHub Actions (CI/CD)
