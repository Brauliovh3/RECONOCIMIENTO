from pathlib import Path
import shutil
import random

BASE = Path(__file__).resolve().parent.parent

SOURCE = BASE / "dataset_real"
OUTPUT = BASE / "dataset_real_cls"

TRAIN_RATIO = 0.8
SEED = 42

random.seed(SEED)

classes = ["casco", "sin_casco"]

for split in ["train", "val"]:
    for cls in classes:
        folder = OUTPUT / split / cls
        folder.mkdir(parents=True, exist_ok=True)

# Limpiar archivos anteriores
for split in ["train", "val"]:
    for cls in classes:
        folder = OUTPUT / split / cls
        for file in folder.iterdir():
            if file.is_file():
                file.unlink()

for cls in classes:
    files = list((SOURCE / cls).glob("*.jpg"))
    random.shuffle(files)

    train_count = int(len(files) * TRAIN_RATIO)

    train_files = files[:train_count]
    val_files = files[train_count:]

    for file in train_files:
        shutil.copy2(
            file,
            OUTPUT / "train" / cls / file.name
        )

    for file in val_files:
        shutil.copy2(
            file,
            OUTPUT / "val" / cls / file.name
        )

    print(f"{cls}:")
    print(f"  Train: {len(train_files)}")
    print(f"  Val:   {len(val_files)}")

print("\nDataset V4 preparado correctamente.")