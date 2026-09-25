from pathlib import Path
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent.parent

DATASET = BASE / "dataset_real_cls"
PROJECT = BASE / "model"

model = YOLO("yolo26n-cls.pt")

results = model.train(
    data=str(DATASET),
    epochs=40,
    imgsz=224,
    batch=16,
    device="cpu",
    project=str(PROJECT),
    name="casco_classifier_v4",
    exist_ok=True,
    patience=10,
    workers=0,
    verbose=True
)

print("\n===================================")
print("     ENTRENAMIENTO V4 TERMINADO")
print("===================================")
print("Modelo:")
print(PROJECT / "casco_classifier_v4" / "weights" / "best.pt")