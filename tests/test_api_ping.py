import os
import pytest

# ถ้าไม่มี requests ให้ข้ามไปเลย (ไม่ error)
requests = pytest.importorskip("requests")

# รันเทสต์นี้เฉพาะเมื่อผู้ใช้ตั้งค่าไว้ชัดเจน
if os.getenv("RUN_API_TESTS") != "1":
    pytest.skip("Skipping API ping test on CI (RUN_API_TESTS!=1)", allow_module_level=True)


def test_ping_local():
    r = requests.get("http://127.0.0.1:8000/ping", timeout=2)
    assert r.status_code == 200
    data = r.json()
    assert data.get("status") == "ok"
