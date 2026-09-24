import json
import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "dados_processados.npz")
CLASSES_FILE = os.path.join(BASE_DIR, "classes.json")
MODEL_FILE = os.path.join(BASE_DIR, "modelo_gestos_lgp.keras")
RESULTS_FILE = os.path.join(BASE_DIR, "training_results.json")

SEED = 42
FOLDS = 5
EPOCHS = 60
BATCH_SIZE = 32
AUGMENTED_COPIES = 5

# A app faz a média das probabilidades das últimas N frames com mão detetada.
# Tem de coincidir com SMOOTHING_WINDOW em GestureRecognizerHelper.kt.
SMOOTHING_WINDOW = 10
THRESHOLDS = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8]


def augment(X, copies, rng):
    """Rotação no plano, espelhamento (mão esquerda/direita) e ruído nas coordenadas."""
    out = [X]
    for _ in range(copies):
        points = X.copy()
        angle = np.radians(rng.uniform(-20, 20, len(points)))[:, None]
        x, y = points[..., 0].copy(), points[..., 1].copy()
        points[..., 0] = np.cos(angle) * x - np.sin(angle) * y
        points[..., 1] = np.sin(angle) * x + np.cos(angle) * y
        points[rng.random(len(points)) < 0.5, :, 0] *= -1
        points += rng.normal(0, 0.02, points.shape)
        out.append(points)
    return np.concatenate(out)


def build_model(num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input((63,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def train(X, y, num_classes, rng):
    X_aug = augment(X, AUGMENTED_COPIES, rng)
    y_aug = np.tile(y, AUGMENTED_COPIES + 1)
    model = build_model(num_classes)
    model.fit(X_aug.reshape(len(X_aug), -1), y_aug, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True, verbose=0)
    return model


def smoothed_windows(probs, y, position, num_classes):
    """Média das probabilidades em janelas de frames consecutivas, como na app."""
    window_probs, window_labels = [], []
    for c in range(num_classes):
        idx = np.where(y == c)[0]
        idx = idx[np.argsort(position[idx])]
        for start in range(len(idx) - SMOOTHING_WINDOW + 1):
            window_probs.append(probs[idx[start:start + SMOOTHING_WINDOW]].mean(axis=0))
            window_labels.append(c)
    return np.array(window_probs), np.array(window_labels)


def cross_validate(X, y, position, class_names, rng):
    """Validação cruzada por blocos contíguos de cada classe.

    As imagens vizinhas são quase iguais, por isso uma divisão aleatória daria
    resultados demasiado otimistas.
    """
    num_classes = len(class_names)
    fold = np.minimum((position * FOLDS).astype(int), FOLDS - 1)
    frame_true, frame_pred, win_probs, win_true = [], [], [], []

    for k in range(FOLDS):
        train_idx, val_idx = np.where(fold != k)[0], np.where(fold == k)[0]
        model = train(X[train_idx], y[train_idx], num_classes, rng)
        probs = model.predict(X[val_idx].reshape(len(val_idx), -1), verbose=0)

        frame_true.append(y[val_idx])
        frame_pred.append(probs.argmax(axis=1))
        wp, wt = smoothed_windows(probs, y[val_idx], position[val_idx], num_classes)
        win_probs.append(wp)
        win_true.append(wt)

        print(f"  bloco {k + 1}/{FOLDS}: frame {accuracy_score(frame_true[-1], frame_pred[-1]):.3f}"
              f"  |  média de {SMOOTHING_WINDOW} frames {accuracy_score(wt, wp.argmax(axis=1)):.3f}")

    frame_true, frame_pred = np.concatenate(frame_true), np.concatenate(frame_pred)
    win_probs, win_true = np.concatenate(win_probs), np.concatenate(win_true)
    win_pred = win_probs.argmax(axis=1)

    thresholds = []
    for t in THRESHOLDS:
        accepted = win_probs.max(axis=1) >= t
        thresholds.append({
            "threshold": t,
            "coverage": float(accepted.mean()),
            "accuracy": float(accuracy_score(win_true[accepted], win_pred[accepted])) if accepted.any() else None,
        })

    labels = list(range(num_classes))
    return {
        "per_frame": {
            "accuracy": float(accuracy_score(frame_true, frame_pred)),
            "macro_f1": float(f1_score(frame_true, frame_pred, average="macro")),
        },
        "smoothed": {
            "window": SMOOTHING_WINDOW,
            "accuracy": float(accuracy_score(win_true, win_pred)),
            "macro_f1": float(f1_score(win_true, win_pred, average="macro")),
            "classification_report": classification_report(
                win_true, win_pred, labels=labels, target_names=class_names, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(win_true, win_pred, labels=labels).tolist(),
            "thresholds": thresholds,
        },
    }


def main():
    if not os.path.exists(DATA_FILE):
        print(f"Erro: '{os.path.basename(DATA_FILE)}' não encontrado. Execute primeiro 'processar_dados.py'.")
        return

    tf.keras.utils.set_random_seed(SEED)
    rng = np.random.default_rng(SEED)

    data = np.load(DATA_FILE)
    X, y, position = data["X"], data["y"], data["position"]
    with open(CLASSES_FILE) as f:
        label_map = json.load(f)
    class_names = [name for name, _ in sorted(label_map.items(), key=lambda item: item[1])]

    print(f"{len(X)} amostras, {len(class_names)} classes: {class_names}")
    print(f"\nValidação cruzada ({FOLDS} blocos):")
    results = cross_validate(X, y, position, class_names, rng)

    smoothed = results["smoothed"]
    print(f"\nPor frame:             exatidão {results['per_frame']['accuracy']:.3f}, "
          f"F1 macro {results['per_frame']['macro_f1']:.3f}")
    print(f"Média de {SMOOTHING_WINDOW} frames:    exatidão {smoothed['accuracy']:.3f}, "
          f"F1 macro {smoothed['macro_f1']:.3f}")
    print("\nPor gesto (média de frames):")
    for name in class_names:
        r = smoothed["classification_report"][name]
        print(f"  {name:>10}: precisão {r['precision']:.2f}  recall {r['recall']:.2f}  F1 {r['f1-score']:.2f}")
    print("\nLimiar de confiança:")
    for t in smoothed["thresholds"]:
        acc = f"{t['accuracy']:.3f}" if t["accuracy"] is not None else "-"
        print(f"  >= {t['threshold']:.1f}: aceita {t['coverage']:.0%} das janelas, exatidão {acc}")

    print("\nA treinar o modelo final com todos os dados...")
    model = train(X, y, len(class_names), rng)
    model.save(MODEL_FILE)

    results["samples"] = int(len(X))
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Modelo guardado em '{os.path.basename(MODEL_FILE)}' e resultados em '{os.path.basename(RESULTS_FILE)}'.")


if __name__ == "__main__":
    main()
