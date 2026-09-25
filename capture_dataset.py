import cv2
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent

DATASET = BASE / "dataset_real"

CLASSES = {
    "1": "casco",
    "2": "sin_casco"
}

IMAGES_PER_CLASS = 100
CAPTURE_INTERVAL = 0.25

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("No se pudo abrir la cámara.")

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("=" * 60)
print("       CAPTURADOR DE DATASET V4")
print("=" * 60)
print()
print("1 = CASCO")
print("2 = SIN CASCO")
print("ESC = SALIR")
print()
print("Colócate frente a la cámara.")
print()

for folder in CLASSES.values():
    (DATASET / folder).mkdir(
        parents=True,
        exist_ok=True
    )


def existing_count(folder):
    return len([
        f for f in folder.iterdir()
        if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ])


while True:

    print("Selecciona una clase:")
    print("1 = CASCO")
    print("2 = SIN CASCO")
    print("ESC = SALIR")

    key = input("Clase: ").strip()

    if key == "1":
        class_name = "casco"

    elif key == "2":
        class_name = "sin_casco"

    else:
        print("Saliendo...")
        break

    output_folder = DATASET / class_name

    current_count = existing_count(output_folder)

    if current_count >= IMAGES_PER_CLASS:
        print()
        print(
            f"{class_name.upper()} ya tiene "
            f"{current_count} imágenes."
        )
        print()
        continue

    print()
    print(f"CLASE SELECCIONADA: {class_name.upper()}")
    print(f"Imágenes actuales: {current_count}")
    print(f"Objetivo: {IMAGES_PER_CLASS}")
    print()

    print("Preparando cámara...")

    # Vista previa antes de comenzar
    while True:

        success, frame = camera.read()

        if not success:
            print("No se pudo leer la cámara.")
            break

        display = frame.copy()

        cv2.rectangle(
            display,
            (15, 15),
            (650, 115),
            (10, 10, 10),
            -1
        )

        cv2.putText(
            display,
            f"CLASE: {class_name.upper()}",
            (35, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2
        )

        cv2.putText(
            display,
            f"Fotos: {current_count}/{IMAGES_PER_CLASS}",
            (35, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "CAPTURA DATASET V4",
            display
        )

        camera_key = cv2.waitKey(1) & 0xFF

        # ENTER comienza captura
        if camera_key == 13:
            break

        # ESC cancela
        if camera_key == 27:
            cv2.destroyAllWindows()
            camera.release()
            raise SystemExit

    print()
    print("CAPTURANDO...")
    print("Muévete lentamente frente a la cámara.")
    print()

    last_capture = 0

    while current_count < IMAGES_PER_CLASS:

        success, frame = camera.read()

        if not success:
            print("Error leyendo cámara.")
            break

        display = frame.copy()

        now = time.time()

        if now - last_capture >= CAPTURE_INTERVAL:

            filename = (
                output_folder
                / f"{class_name}_{current_count + 1:03d}.jpg"
            )

            cv2.imwrite(
                str(filename),
                frame,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    95
                ]
            )

            current_count += 1
            last_capture = now

            print(
                f"[{current_count:03d}/{IMAGES_PER_CLASS}] "
                f"{filename.name}"
            )

        # HUD
        cv2.rectangle(
            display,
            (15, 15),
            (500, 125),
            (10, 10, 10),
            -1
        )

        cv2.putText(
            display,
            f"CAPTURANDO: {class_name.upper()}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2
        )

        cv2.putText(
            display,
            f"{current_count}/{IMAGES_PER_CLASS}",
            (30, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            "ESC = cancelar",
            (30, 112),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 180, 180),
            1
        )

        cv2.imshow(
            "CAPTURA DATASET V4",
            display
        )

        camera_key = cv2.waitKey(1) & 0xFF

        if camera_key == 27:
            print()
            print("Captura cancelada.")
            break

    cv2.destroyAllWindows()

    print()
    print(
        f"Captura de {class_name.upper()} terminada."
    )
    print(
        f"Total: {current_count}/{IMAGES_PER_CLASS}"
    )
    print()


cv2.destroyAllWindows()
camera.release()

print()
print("=" * 60)
print("CAPTURADOR FINALIZADO")
print("=" * 60)
print()
print(f"Dataset: {DATASET}")
print()
