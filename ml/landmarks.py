import numpy as np

NUM_LANDMARKS = 21
NUM_FEATURES = NUM_LANDMARKS * 3
WRIST = 0


def normalize_landmarks(points, width, height):
    """Make hand landmarks independent of where the hand is and how big it looks.

    `points` are MediaPipe's 21 (x, y, z) landmarks in normalised image coordinates.
    They are converted to pixel proportions (so the image aspect ratio does not matter),
    moved so the wrist is at the origin and divided by the largest wrist distance in x-y.

    Must match `normalizeLandmarks` in the Android app and in web/js/classifier.js.
    """
    points = np.asarray(points, dtype=np.float32) * np.array([width, height, width], dtype=np.float32)
    points -= points[WRIST]
    scale = np.linalg.norm(points[:, :2], axis=1).max()
    return points / max(scale, 1e-6)


def augment(points, copies, rng):
    """Return `points` plus `copies` randomly rotated, mirrored and jittered versions.

    `points` has shape (n, 21, 3). Mirroring makes the model work for either hand.
    """
    out = [points]
    for _ in range(copies):
        p = points.copy()
        angle = np.radians(rng.uniform(-20, 20, len(p)))[:, None]
        x, y = p[..., 0].copy(), p[..., 1].copy()
        p[..., 0] = np.cos(angle) * x - np.sin(angle) * y
        p[..., 1] = np.sin(angle) * x + np.cos(angle) * y
        p[rng.random(len(p)) < 0.5, :, 0] *= -1
        p += rng.normal(0, 0.02, p.shape)
        out.append(p.astype(np.float32))
    return np.concatenate(out)
