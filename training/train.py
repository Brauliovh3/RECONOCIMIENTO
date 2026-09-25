from ultralytics import YOLO
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATASET = BASE / "dataset"

print("================================")
print("🪖 ENTRENAMIENTO CASCO")
print("================================")
print(f"Dataset: {DATASET}")

# Modelo pequeño de clasificación
model = YOLO("yolo26n-cls.pt")

# Entrenamiento
results = model.train(
    data=str(DATASET / "train"),
    epochs=30,
    imgsz=224,
    batch=4,
    device="cpu",
    project=str(BASE / "model"),
    name="casco_classifier",
    exist_ok=True
)

print("\n================================")
print("✅ ENTRENAMIENTO TERMINADO")
print("================================")