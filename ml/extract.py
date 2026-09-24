"""Step 1: detect the hand in every dataset image and save its normalised landmarks."""
import json
import re

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
from tqdm import tqdm

import config
from landmarks import normalize_landmarks

TARGET_SIZE = 256

# Many images are tight crops of the hand, where MediaPipe's palm detector struggles.
# Adding a margin around the crop (and mirroring it) recovers most of them.
# Ordered by how many extra detections each variant adds.
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
        base_options=BaseOptions(model_asset_path=str(config.HAND_LANDMARKER_MODEL)),
        running_mode=RunningMode.IMAGE,
        num_hands=1,
        # Every dataset image contains a hand, so a permissive threshold is safe here.
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


def detect(detector, image):
    """Return the normalised (21, 3) landmarks of the hand in `image`, or None."""
    for margin, border, flip in VARIANTS:
        variant = prepare(image, margin, border, flip)
        rgb = np.ascontiguousarray(cv2.cvtColor(variant, cv2.COLOR_BGR2RGB))
        result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not result.hand_landmarks:
            continue

        points = np.array([[lm.x, lm.y, lm.z] for lm in result.hand_landmarks[0]], dtype=np.float32)
        if flip:
            points[:, 0] = 1.0 - points[:, 0]
        height, width = variant.shape[:2]
        return normalize_landmarks(points, width, height)
    return None


def frame_number(path):
    match = re.search(r"_(\d+)\.jpe?g$", path.name, re.IGNORECASE)
    if match:
        return int(match.group(1))
    digits = re.findall(r"\d+", path.name)
    return int("".join(digits)) if digits else 0


def main():
    detector = create_detector()
    samples, labels, positions, files = [], [], [], []
    stats = {}

    for gesture in config.GESTURES:
        images = sorted(
            (p for p in (config.DATASET_DIR / gesture).iterdir() if p.suffix.lower() in (".jpg", ".jpeg")),
            key=frame_number,
        )
        detected = 0
        for index, path in enumerate(tqdm(images, desc=f"{gesture:>10}")):
            image = cv2.imread(str(path))
            points = detect(detector, image) if image is not None else None
            if points is None:
                continue
            detected += 1
            samples.append(points)
            labels.append(gesture)
            # Relative position in the recording, used for block-wise cross-validation.
            positions.append(index / len(images))
            files.append(f"{gesture}/{path.name}")
        stats[gesture] = {"images": len(images), "detected": detected}

    config.ARTIFACTS_DIR.mkdir(exist_ok=True)
    np.savez_compressed(
        config.LANDMARKS_FILE,
        X=np.array(samples, dtype=np.float32),
        labels=np.array(labels),
        position=np.array(positions, dtype=np.float32),
        files=np.array(files),
    )
    config.EXTRACTION_STATS_FILE.write_text(json.dumps(stats, indent=2) + "\n")

    print("\nHands detected:")
    for gesture, s in stats.items():
        print(f"  {gesture:>10}: {s['detected']:3d}/{s['images']} ({s['detected'] / s['images']:.0%})")
    print(f"\nSaved {len(samples)} samples to {config.LANDMARKS_FILE.relative_to(config.ROOT_DIR)}")


if __name__ == "__main__":
    main()
