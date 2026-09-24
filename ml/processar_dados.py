import json
import os
import re

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
from tqdm import tqdm

from landmarks import normalize_landmarks

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset_limpo")
HAND_LANDMARKER_MODEL = os.path.join(BASE_DIR, "..", "android", "app", "src", "main", "assets", "hand_landmarker.task")
OUTPUT_FILE = os.path.join(BASE_DIR, "dados_processados.npz")
CLASSES_FILE = os.path.join(BASE_DIR, "classes.json")
STATS_FILE = os.path.join(BASE_DIR, "preprocessing_stats.json")

TARGET_SIZE = 256

# Muitas imagens são recortes justos da mão, onde o detetor de palmas do MediaPipe falha.
# Acrescentar margem à volta (e espelhar) recupera a maioria. Ordem escolhida por ganho.
VARIANTS = [
    (1.0, "replicate", False),
    (0.0, "black", True),
    (0.3, "replicate", False),
    (0.3, "black", True),
    (0.15, "black", False),
    (0.3, "gray", False),
    (1.0, "gray", False),
    (1.0, "black", True),
    (0.5, "black", False),
    (1.0, "replicate", True),
]
BORDERS = {
    "black": dict(borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)),
    "gray": dict(borderType=cv2.BORDER_CONSTANT, value=(128, 128, 128)),
    "replicate": dict(borderType=cv2.BORDER_REPLICATE),
}


def create_detector():
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_LANDMARKER_MODEL),
        running_mode=RunningMode.IMAGE,
        num_hands=1,
        # Todas as imagens do dataset contêm uma mão, por isso podemos ser permissivos.
        min_hand_detection_confidence=0.1,
        min_hand_presence_confidence=0.1,
    )
    return HandLandmarker.create_from_options(options)


def prepare(image, margin, border, flip):
    if flip:
        image = cv2.flip(image, 1)
    if margin:
        pad = int(max(image.shape[:2]) * margin)
        image = cv2.copyMakeBorder(image, pad, pad, pad, pad, **BORDERS[border])
    height, width = image.shape[:2]
    scale = TARGET_SIZE / max(height, width)
    return cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_CUBIC)


def extract_landmarks(detector, image_path):
    """Devolve os marcos normalizados (21, 3) da mão, ou None se nenhuma variante tiver deteção."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Aviso: não foi possível carregar a imagem {image_path}")
        return None

    for margin, border, flip in VARIANTS:
        variant = prepare(image, margin, border, flip)
        rgb = np.ascontiguousarray(cv2.cvtColor(variant, cv2.COLOR_BGR2RGB))
        result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not result.hand_landmarks:
            continue

        height, width = variant.shape[:2]
        points = np.array([[lm.x, lm.y, lm.z] for lm in result.hand_landmarks[0]], dtype=np.float32)
        if flip:
            points[:, 0] = 1.0 - points[:, 0]
        return normalize_landmarks(points, width, height)

    return None


def sort_key(filename):
    match = re.search(r"_(\d+)\.(jpg|jpeg)$", filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    digits = re.findall(r"\d+", filename)
    return int("".join(digits)) if digits else 0


def main():
    class_names = sorted(d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d)))
    if not class_names:
        print(f"Erro: nenhuma pasta de classe encontrada em '{DATASET_PATH}'.")
        return

    label_map = {name: i for i, name in enumerate(class_names)}
    detector = create_detector()

    features, labels, positions, files = [], [], [], []
    stats = {}

    for class_name in class_names:
        class_path = os.path.join(DATASET_PATH, class_name)
        image_files = sorted(
            (f for f in os.listdir(class_path) if f.lower().endswith((".jpg", ".jpeg"))),
            key=sort_key,
        )

        detected = 0
        for position, filename in enumerate(tqdm(image_files, desc=f"{class_name:>10}")):
            landmarks = extract_landmarks(detector, os.path.join(class_path, filename))
            if landmarks is None:
                continue
            detected += 1
            features.append(landmarks)
            labels.append(label_map[class_name])
            positions.append(position / len(image_files))
            files.append(f"{class_name}/{filename}")

        stats[class_name] = {"total_images": len(image_files), "detected": detected}

    np.savez_compressed(
        OUTPUT_FILE,
        X=np.array(features, dtype=np.float32),
        y=np.array(labels),
        position=np.array(positions, dtype=np.float32),
        files=np.array(files),
    )
    with open(CLASSES_FILE, "w") as f:
        json.dump(label_map, f, indent=4, ensure_ascii=False)
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=4, ensure_ascii=False)

    print("\nMãos detetadas:")
    for name, s in stats.items():
        print(f"  {name:>10}: {s['detected']:3d}/{s['total_images']} ({s['detected'] / s['total_images']:.0%})")
    print(f"\n{len(features)} amostras guardadas em '{os.path.basename(OUTPUT_FILE)}'.")


if __name__ == "__main__":
    main()
