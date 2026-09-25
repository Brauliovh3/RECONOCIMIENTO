import cv2
from pathlib import Path
import shutil

BASE = Path(r"C:\Users\Lab 401 B\SOFTWARE\RECONOCIMIENTO")
SOURCE = BASE / "dataset"
OUTPUT = BASE / "dataset_detection"

CLASSES = {
    0: "CASCO",
    1: "CABEZA",
    2: "OBJETO INCORRECTO"
}

# Buscar imágenes
images = []

for folder_name in ["casco", "sin_casco"]:
    folder = SOURCE / folder_name

    if not folder.exists():
        print(f"NO EXISTE: {folder}")
        continue

    for file in sorted(folder.iterdir()):
        if file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            images.append(file)

print()
print("==========================================")
print("       ANOTADOR V3 - CASCO")
print("==========================================")
print()
print(f"Imagenes encontradas: {len(images)}")
print()

if len(images) == 0:
    print("NO SE ENCONTRARON IMAGENES.")
    input("Presiona ENTER para cerrar...")
    raise SystemExit

# Crear carpetas
for split in ["train", "val"]:
    (OUTPUT / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUTPUT / "labels" / split).mkdir(parents=True, exist_ok=True)

# 80% train / 20% val
split_index = max(1, int(len(images) * 0.8))

train_images = images[:split_index]
val_images = images[split_index:]

all_images = []

for img in train_images:
    all_images.append(("train", img))

for img in val_images:
    all_images.append(("val", img))

window_name = "ANOTADOR V3 - CASCO"

cv2.namedWindow(
    window_name,
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    window_name,
    1100,
    750
)

current_class = 0
boxes = []
drawing = False
start_x = 0
start_y = 0


def mouse_callback(event, x, y, flags, param):

    global drawing
    global start_x
    global start_y

    if event == cv2.EVENT_LBUTTONDOWN:

        drawing = True
        start_x = x
        start_y = y

    elif event == cv2.EVENT_LBUTTONUP:

        drawing = False

        x1 = min(start_x, x)
        y1 = min(start_y, y)

        x2 = max(start_x, x)
        y2 = max(start_y, y)

        if (x2 - x1) > 8 and (y2 - y1) > 8:

            boxes.append(
                (
                    x1,
                    y1,
                    x2,
                    y2,
                    current_class
                )
            )

            print(
                f"Cuadro agregado: {CLASSES[current_class]}"
            )


cv2.setMouseCallback(
    window_name,
    mouse_callback
)


def fit_image(image, max_width=1050, max_height=650):

    height, width = image.shape[:2]

    scale = min(
        max_width / width,
        max_height / height,
        1.0
    )

    new_width = int(width * scale)
    new_height = int(height * scale)

    if scale != 1.0:

        image = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )

    return image


for number, (split, image_path) in enumerate(all_images, start=1):

    print()
    print("------------------------------------------")
    print(f"IMAGEN {number}/{len(all_images)}")
    print(f"Archivo: {image_path.name}")
    print(f"Carpeta: {image_path.parent.name}")
    print(f"Destino: {split}")
    print("------------------------------------------")

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR
    )

    if image is None:

        print("ERROR: OpenCV no pudo leer esta imagen.")
        continue

    print(
        f"Resolucion original: "
        f"{image.shape[1]} x {image.shape[0]}"
    )

    # Ajustar imagen para la ventana
    image = fit_image(image)

    original = image.copy()

    boxes = []

    while True:

        display = image.copy()

        # Fondo superior
        cv2.rectangle(
            display,
            (0, 0),
            (display.shape[1], 75),
            (15, 15, 15),
            -1
        )

        cv2.putText(
            display,
            f"IMAGEN {number}/{len(all_images)}",
            (15, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            f"CLASE: {CLASSES[current_class]}",
            (15, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        # Cuadros
        for x1, y1, x2, y2, cls in boxes:

            if cls == 0:
                color = (0, 255, 0)

            elif cls == 1:
                color = (0, 200, 255)

            else:
                color = (0, 0, 255)

            cv2.rectangle(
                display,
                (x1, y1),
                (x2, y2),
                color,
                3
            )

            cv2.putText(
                display,
                CLASSES[cls],
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        cv2.imshow(
            window_name,
            display
        )

        key = cv2.waitKey(30) & 0xFF

        # 1 = casco
        if key == ord("1"):

            current_class = 0
            print("CLASE -> CASCO")

        # 2 = cabeza
        elif key == ord("2"):

            current_class = 1
            print("CLASE -> CABEZA")

        # 3 = objeto incorrecto
        elif key == ord("3"):

            current_class = 2
            print("CLASE -> OBJETO INCORRECTO")

        # R = borrar ultimo
        elif key in [ord("r"), ord("R")]:

            if boxes:

                removed = boxes.pop()

                print(
                    f"Eliminado: {CLASSES[removed[4]]}"
                )

        # C = borrar todos
        elif key in [ord("c"), ord("C")]:

            boxes.clear()

            print("Todos los cuadros eliminados.")

        # ENTER = guardar
        elif key == 13:

            height, width = image.shape[:2]

            label_lines = []

            for x1, y1, x2, y2, cls in boxes:

                x_center = (
                    (x1 + x2) / 2
                ) / width

                y_center = (
                    (y1 + y2) / 2
                ) / height

                box_width = (
                    x2 - x1
                ) / width

                box_height = (
                    y2 - y1
                ) / height

                label_lines.append(
                    f"{cls} "
                    f"{x_center:.6f} "
                    f"{y_center:.6f} "
                    f"{box_width:.6f} "
                    f"{box_height:.6f}"
                )

            destination_image = (
                OUTPUT
                / "images"
                / split
                / image_path.name
            )

            destination_label = (
                OUTPUT
                / "labels"
                / split
                / f"{image_path.stem}.txt"
            )

            shutil.copy2(
                image_path,
                destination_image
            )

            destination_label.write_text(
                "\n".join(label_lines),
                encoding="utf-8"
            )

            print()
            print("GUARDADO")
            print(f"Imagen: {destination_image}")
            print(f"Etiqueta: {destination_label}")
            print(f"Objetos: {len(boxes)}")

            break

        # ESC
        elif key == 27:

            cv2.destroyAllWindows()

            print()
            print("Anotacion cancelada.")

            raise SystemExit


cv2.destroyAllWindows()

print()
print("==========================================")
print("       ANOTACION COMPLETADA")
print("==========================================")
print()
print(f"Dataset: {OUTPUT}")
print()

input("Presiona ENTER para cerrar...")
