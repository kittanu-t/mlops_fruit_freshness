#!/usr/bin/env python3
from mlflow.tracking import MlflowClient

MODEL_NAME = "fruits-freshness-classifier"
ALIAS = "Production"

def main():
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        raise SystemExit("No registered versions found.")
    latest = max(versions, key=lambda v: int(v.version))
    client.set_registered_model_alias(MODEL_NAME, ALIAS, latest.version)
    print(f"Set alias '{ALIAS}' -> version {latest.version}")

if __name__ == "__main__":
    main()
