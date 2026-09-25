from ultralytics import YOLO
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

DATASET = BASE / "dataset_augmented"
PROJECT = BASE / "model"

TRAIN = DATASET / "train"
VAL = DATASET / "val"

print("==========================================")
print("🪖 ENTRENAMIENTO CASCO V2")
print("==========================================")
print()

print(f"Dataset: {DATASET}")
print()

# ==========================================
# VERIFICAR DATASET
# ==========================================

train_casco = TRAIN / "casco"
train_sin_casco = TRAIN / "sin_casco"

val_casco = VAL / "casco"
val_sin_casco = VAL / "sin_casco"

train_casco_count = len(list(train_casco.glob("*")))
train_sin_casco_count = len(list(train_sin_casco.glob("*")))

val_casco_count = len(list(val_casco.glob("*")))
val_sin_casco_count = len(list(val_sin_casco.glob("*")))

print("TRAIN")
print(f"  casco:     {train_casco_count}")
print(f"  sin_casco: {train_sin_casco_count}")
print()

print("VALIDACIÓN")
print(f"  casco:     {val_casco_count}")
print(f"  sin_casco: {val_sin_casco_count}")
print()

if train_casco_count != 100:
    raise RuntimeError(
        f"TRAIN/casco incorrecto: {train_casco_count}"
    )

if train_sin_casco_count != 100:
    raise RuntimeError(
        f"TRAIN/sin_casco incorrecto: {train_sin_casco_count}"
    )

if val_casco_count != 2:
    raise RuntimeError(
        f"VAL/casco incorrecto: {val_casco_count}"
    )

if val_sin_casco_count != 2:
    raise RuntimeError(
        f"VAL/sin_casco incorrecto: {val_sin_casco_count}"
    )

print("✅ Dataset verificado correctamente")
print()

# ==========================================
# CARGAR MODELO
# ==========================================

print("Cargando modelo base...")

model = YOLO("yolo26n-cls.pt")

print("✅ Modelo base cargado")
print()

# ==========================================
# ENTRENAMIENTO
# ==========================================

print("==========================================")
print("🚀 INICIANDO ENTRENAMIENTO")
print("==========================================")
print()

results = model.train(
    data=str(DATASET),
    epochs=40,
    imgsz=224,
    batch=16,
    device="cpu",
    project=str(PROJECT),
    name="casco_classifier_v2",
    exist_ok=True,
    patience=10,
    verbose=True
)

# ==========================================
# FINAL
# ==========================================

print()
print("==========================================")
print("✅ ENTRENAMIENTO V2 TERMINADO")
print("==========================================")
print()

print("Modelo generado:")

print(
    PROJECT /
    "casco_classifier_v2" /
    "weights" /
    "best.pt"
)

print()