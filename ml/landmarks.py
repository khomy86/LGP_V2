import numpy as np

NUM_LANDMARKS = 21
WRIST = 0


def normalize_landmarks(points, width, height):
    """Torna os marcos independentes da posição e do tamanho da mão na imagem.

    `points` são os 21 marcos (x, y, z) do MediaPipe, normalizados para [0, 1].
    Passam para proporções de píxel (para não depender do formato da imagem), ficam
    relativos ao pulso e são divididos pelo maior afastamento ao pulso no plano x-y.

    Tem de coincidir com `normalizeLandmarks` em GestureRecognizerHelper.kt.
    """
    points = np.asarray(points, dtype=np.float32) * np.array([width, height, width], dtype=np.float32)
    points -= points[WRIST]
    scale = np.linalg.norm(points[:, :2], axis=1).max()
    return points / max(scale, 1e-6)
