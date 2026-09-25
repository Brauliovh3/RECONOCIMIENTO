from pathlib import Path
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "dataset_detection" / "data.yaml"
PROJECT = BASE / "model"

print("=" * 60)
print("        YOLO V3 - DETECTOR DE CASCO")
print("=" * 60)
print()
print(f"Dataset: {DATA}")
print()

if not DATA.exists():
    raise FileNotFoundError(f"No existe: {DATA}")

model = YOLO("yolo26n.pt")

results = model.train(
    data=str(DATA),
    epochs=80,
    imgsz=640,
    batch=4,
    device="cpu",
    project=str(PROJECT),
    name="casco_detection_v3",
    exist_ok=True,
    patience=20,
    workers=0,
    verbose=True
)

print()
print("=" * 60)
print("ENTRENAMIENTO TERMINADO")
print("=" * 60)
print()
print("Modelo:")
print(PROJECT / "casco_detection_v3" / "weights" / "best.pt")
