from pathlib import Path
import random
import shutil

# Carpetas originales
BASE = Path(__file__).resolve().parent.parent
SOURCE = BASE / "dataset"

# Carpetas nuevas
TRAIN = SOURCE / "train"
VAL = SOURCE / "val"

CLASSES = ["casco", "sin_casco"]

# Porcentaje para validación
VAL_RATIO = 0.25

# Extensiones aceptadas
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

random.seed(42)

for class_name in CLASSES:
    source_dir = SOURCE / class_name

    if not source_dir.exists():
        print(f"[ERROR] No existe: {source_dir}")
        continue

    images = [
        file for file in source_dir.iterdir()
        if file.is_file() and file.suffix.lower() in EXTENSIONS
    ]

    random.shuffle(images)

    if len(images) < 2:
        print(f"[ERROR] Muy pocas imágenes en {class_name}")
        continue

    val_count = max(1, round(len(images) * VAL_RATIO))

    val_images = images[:val_count]
    train_images = images[val_count:]

    train_dir = TRAIN / class_name
    val_dir = VAL / class_name

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    for image in train_images:
        shutil.copy2(image, train_dir / image.name)

    for image in val_images:
        shutil.copy2(image, val_dir / image.name)

    print(f"\n{class_name}:")
    print(f"  Total: {len(images)}")
    print(f"  Entrenamiento: {len(train_images)}")
    print(f"  Validación: {len(val_images)}")

print("\n================================")
print("DATASET PREPARADO")
print("================================")