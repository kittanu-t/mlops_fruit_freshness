from pathlib import Path
import json

def test_class_indices_optional():
    p = Path("processed_metadata/class_indices.json")
    # บน CI อาจไม่มีไฟล์นี้ ให้เทสต์ผ่านถ้าไม่มี เพื่อไม่ผูกกับ data
    if p.exists():
        m = json.loads(p.read_text(encoding="utf-8"))
        assert isinstance(m, dict)
        # ถ้ามี ให้มีอย่างน้อย 5 คลาส
        assert len(m) >= 5
