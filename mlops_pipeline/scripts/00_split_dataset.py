#!/usr/bin/env python3
import os, random, shutil
from pathlib import Path

# ==== config ====
SRC_DIR = Path("data")              # ต้นทางที่คุณมีตอนนี้
OUT_DIR = Path("data_splits")       # ปลายทางที่จะสร้าง
RATIOS = (0.8, 0.1, 0.1)            # train, val, test
SEED = 42

random.seed(SEED)
assert sum(RATIOS) == 1.0

classes = [d.name for d in SRC_DIR.iterdir() if d.is_dir()]
print("Found classes:", classes)

for split in ["train", "val", "test"]:
    for cls in classes:
        (OUT_DIR / split / cls).mkdir(parents=True, exist_ok=True)

for cls in classes:
    files = [p for p in (SRC_DIR / cls).iterdir() if p.is_file()]
    random.shuffle(files)
    n = len(files)
    n_train = int(n * RATIOS[0])
    n_val   = int(n * RATIOS[1])
    train_files = files[:n_train]
    val_files   = files[n_train:n_train+n_val]
    test_files  = files[n_train+n_val:]

    def _copy(fs, split):
        for f in fs:
            shutil.copy2(f, OUT_DIR / split / cls / f.name)

    _copy(train_files, "train")
    _copy(val_files,   "val")
    _copy(test_files,  "test")
    print(f"{cls:20s} -> train {len(train_files):5d} | val {len(val_files):5d} | test {len(test_files):5d}")

print("\nDone. New root:", OUT_DIR.resolve())
