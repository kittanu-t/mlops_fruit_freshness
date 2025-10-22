# tests/test_api_ping.py
import requests

def test_ping():
    r = requests.get("http://127.0.0.1:8000/ping")
    assert r.status_code == 200
    data = r.json()
    assert "model" in data
