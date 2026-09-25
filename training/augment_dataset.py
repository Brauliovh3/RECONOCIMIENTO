from pathlib import Path
import random
import shutil
import cv2

BASE = Path(__file__).resolve().parent.parent
SOURCE = BASE / "dataset"
OUTPUT = BASE / "dataset_augmented"

CLASSES = ["casco", "sin_casco"]

TRAIN_TARGET = 100
VAL_COUNT = 2

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

random.seed(42)


def images_in(folder):
    return [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    ]


def augment(image):

    h, w = image.shape[:2]

    # Volteo
    if random.random() < 0.5:
        image = cv2.flip(image, 1)

    # Rotación
    angle = random.uniform(-10, 10)

    matrix = cv2.getRotationMatrix2D(
        (w // 2, h // 2),
        angle,
        random.uniform(0.92, 1.08)
    )

    image = cv2.warpAffine(
        image,
        matrix,
        (w, h),
        borderMode=cv2.BORDER_REFLECT_101
    )

    # Brillo / contraste
    image = cv2.convertScaleAbs(
        image,
        alpha=random.uniform(0.85, 1.15),
        beta=random.randint(-25, 25)
    )

    # Desenfoque ocasional
    if random.random() < 0.15:
        image = cv2.GaussianBlur(image, (3, 3), 0)

    return image


# ==========================================
# LIMPIAR
# ==========================================

if OUTPUT.exists():
    shutil.rmtree(OUTPUT)


# ==========================================
# CREAR CARPETAS
# ==========================================

for split in ["train", "val"]:
    for class_name in CLASSES:
        (OUTPUT / split / class_name).mkdir(
            parents=True,
            exist_ok=True
        )


print()
print("==========================================")
print("GENERANDO DATASET")
print("==========================================")
print()


# ==========================================
# PROCESAR CADA CLASE
# ==========================================

for class_name in CLASSES:

    source = SOURCE / class_name
    original_images = images_in(source)

    print(f"CLASE: {class_name}")
    print(f"Originales: {len(original_images)}")

    if len(original_images) < 3:
        raise RuntimeError(
            f"No hay suficientes imágenes en {source}"
        )

    random.shuffle(original_images)

    # --------------------------------------
    # VALIDACIÓN
    # --------------------------------------

    val_images = original_images[:VAL_COUNT]

    # --------------------------------------
    # TRAIN
    # --------------------------------------

    train_images = original_images[VAL_COUNT:]

    train_dir = OUTPUT / "train" / class_name
    val_dir = OUTPUT / "val" / class_name

    # Copiar validación
    for i, source_file in enumerate(val_images):

        destination = val_dir / f"val_{i}.jpg"

        shutil.copy2(
            source_file,
            destination
        )

    # Copiar originales de train
    for i, source_file in enumerate(train_images):

        destination = train_dir / f"original_{i}.jpg"

        shutil.copy2(
            source_file,
            destination
        )

    # --------------------------------------
    # AUGMENTATION
    # --------------------------------------

    counter = 0

    while len(images_in(train_dir)) < TRAIN_TARGET:

        source_file = random.choice(train_images)

        image = cv2.imread(str(source_file))

        if image is None:
            raise RuntimeError(
                f"No se pudo leer {source_file}"
            )

        result = augment(image)

        output_file = (
            train_dir / f"aug_{counter:04d}.jpg"
        )

        if not cv2.imwrite(
            str(output_file),
            result,
            [cv2.IMWRITE_JPEG_QUALITY, 95]
        ):
            raise RuntimeError(
                f"No se pudo guardar {output_file}"
            )

        counter += 1

    print(
        f"TRAIN creadas: {len(images_in(train_dir))}"
    )

    print(
        f"VAL creadas: {len(images_in(val_dir))}"
    )

    print()


# ==========================================
# VERIFICACIÓN REAL
# ==========================================

print("==========================================")
print("VERIFICACIÓN")
print("==========================================")
print()

for class_name in CLASSES:

    train = OUTPUT / "train" / class_name
    val = OUTPUT / "val" / class_name

    train_count = len(images_in(train))
    val_count = len(images_in(val))

    print(
        f"{class_name}: "
        f"TRAIN={train_count} | "
        f"VAL={val_count}"
    )

    if train_count != TRAIN_TARGET:
        raise RuntimeError(
            f"ERROR: {class_name} no tiene {TRAIN_TARGET} imágenes"
        )

    if val_count != VAL_COUNT:
        raise RuntimeError(
            f"ERROR: {class_name} no tiene {VAL_COUNT} imágenes"
        )

print()
print("==========================================")
print("DATASET COMPLETO Y CORRECTO")
print("==========================================")
print()
