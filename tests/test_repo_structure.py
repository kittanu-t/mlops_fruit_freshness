from pathlib import Path

def test_scripts_exist():
    required = [
        "mlops_pipeline/scripts/01_data_validation.py",
        "mlops_pipeline/scripts/02_data_preprocessing.py",
        "mlops_pipeline/scripts/03_train_evaluate_register.py",
        "mlops_pipeline/scripts/04_transition_model.py",
    ]
    for p in required:
        assert Path(p).exists(), f"Missing: {p}"
