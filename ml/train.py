"""Step 2: evaluate the classifier with block-wise cross-validation, then train the final model."""
import json

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

import config
import plots
from landmarks import NUM_FEATURES, augment

FOLDS = 5
EPOCHS = 60
BATCH_SIZE = 32
AUGMENTED_COPIES = 5
THRESHOLDS = [0.4, 0.5, 0.6, 0.7, 0.8]


def build_model(num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input((NUM_FEATURES,), name="landmarks"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation="softmax", name="probabilities"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def train(X, y, num_classes, rng):
    X_train = augment(X, AUGMENTED_COPIES, rng).reshape(-1, NUM_FEATURES)
    y_train = np.tile(y, AUGMENTED_COPIES + 1)
    model = build_model(num_classes)
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True, verbose=0)
    return model


def smoothed(probs, y, position, num_classes):
    """Average predictions over sliding windows of consecutive frames, like the apps do."""
    window_probs, window_labels = [], []
    for c in range(num_classes):
        idx = np.where(y == c)[0]
        idx = idx[np.argsort(position[idx])]
        for start in range(len(idx) - config.SMOOTHING_WINDOW + 1):
            window_probs.append(probs[idx[start:start + config.SMOOTHING_WINDOW]].mean(axis=0))
            window_labels.append(c)
    return np.array(window_probs), np.array(window_labels)


def cross_validate(X, y, position, num_classes, rng):
    """Cross-validation where each fold is a contiguous block of every gesture's images.

    Neighbouring images are near-duplicates, so a random split would leak them into
    validation and overstate accuracy.
    """
    fold = np.minimum((position * FOLDS).astype(int), FOLDS - 1)
    frame_true, frame_pred, window_probs, window_true = [], [], [], []

    for k in range(FOLDS):
        train_idx, val_idx = np.where(fold != k)[0], np.where(fold == k)[0]
        model = train(X[train_idx], y[train_idx], num_classes, rng)
        probs = model.predict(X[val_idx].reshape(-1, NUM_FEATURES), verbose=0)

        frame_true.append(y[val_idx])
        frame_pred.append(probs.argmax(axis=1))
        wp, wt = smoothed(probs, y[val_idx], position[val_idx], num_classes)
        window_probs.append(wp)
        window_true.append(wt)
        print(f"  fold {k + 1}/{FOLDS}: per frame {accuracy_score(frame_true[-1], frame_pred[-1]):.3f}, "
              f"smoothed {accuracy_score(wt, wp.argmax(axis=1)):.3f}")

    return (np.concatenate(frame_true), np.concatenate(frame_pred),
            np.concatenate(window_probs), np.concatenate(window_true))


def summarize(frame_true, frame_pred, window_probs, window_true, gestures):
    labels = list(range(len(gestures)))
    window_pred = window_probs.argmax(axis=1)
    confidence = window_probs.max(axis=1)

    thresholds = []
    for t in THRESHOLDS:
        accepted = confidence >= t
        thresholds.append({
            "threshold": t,
            "answered": float(accepted.mean()),
            "accuracy": float(accuracy_score(window_true[accepted], window_pred[accepted])) if accepted.any() else None,
        })

    return {
        "gestures": gestures,
        "per_frame": {
            "accuracy": float(accuracy_score(frame_true, frame_pred)),
            "macro_f1": float(f1_score(frame_true, frame_pred, average="macro")),
        },
        "smoothed": {
            "window": config.SMOOTHING_WINDOW,
            "accuracy": float(accuracy_score(window_true, window_pred)),
            "macro_f1": float(f1_score(window_true, window_pred, average="macro")),
            "report": classification_report(window_true, window_pred, labels=labels, target_names=gestures,
                                            output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(window_true, window_pred, labels=labels).tolist(),
            "thresholds": thresholds,
        },
    }


def select(data, gestures):
    keep = np.isin(data["labels"], gestures)
    y = np.array([gestures.index(label) for label in data["labels"][keep]])
    return data["X"][keep], y, data["position"][keep]


def main():
    if not config.LANDMARKS_FILE.exists():
        print("Landmarks not found. Run extract.py first.")
        return

    tf.keras.utils.set_random_seed(config.SEED)
    rng = np.random.default_rng(config.SEED)

    data = np.load(config.LANDMARKS_FILE)
    gestures = config.ENABLED_GESTURES
    X, y, position = select(data, gestures)

    print(f"{len(X)} samples, {len(gestures)} gestures: {', '.join(gestures)}")
    print(f"\n{FOLDS}-fold block cross-validation:")
    metrics = summarize(*cross_validate(X, y, position, len(gestures), rng), gestures)

    smoothed_metrics = metrics["smoothed"]
    print(f"\nPer frame:          accuracy {metrics['per_frame']['accuracy']:.3f}, "
          f"macro F1 {metrics['per_frame']['macro_f1']:.3f}")
    print(f"Smoothed ({config.SMOOTHING_WINDOW} frames): accuracy {smoothed_metrics['accuracy']:.3f}, "
          f"macro F1 {smoothed_metrics['macro_f1']:.3f}\n")
    for gesture in gestures:
        r = smoothed_metrics["report"][gesture]
        print(f"  {gesture:>10}: precision {r['precision']:.2f}  recall {r['recall']:.2f}  F1 {r['f1-score']:.2f}")
    print()
    for t in smoothed_metrics["thresholds"]:
        print(f"  confidence >= {t['threshold']:.1f}: answers {t['answered']:.0%} of windows, "
              f"accuracy {t['accuracy']:.3f}")

    all_gestures = list(config.GESTURES)
    if all_gestures != gestures:
        print(f"\nFor comparison, all {len(all_gestures)} gestures:")
        X_all, y_all, position_all = select(data, all_gestures)
        comparison = summarize(*cross_validate(X_all, y_all, position_all, len(all_gestures), rng), all_gestures)
        metrics["all_gestures"] = {
            "gestures": all_gestures,
            "per_frame": comparison["per_frame"],
            "smoothed": {k: comparison["smoothed"][k] for k in ("accuracy", "macro_f1")},
        }
        print(f"  smoothed accuracy {comparison['smoothed']['accuracy']:.3f}, "
              f"macro F1 {comparison['smoothed']['macro_f1']:.3f}")

    print("\nTraining the final model on all samples...")
    model = train(X, y, len(gestures), rng)
    config.ARTIFACTS_DIR.mkdir(exist_ok=True)
    model.save(config.MODEL_FILE)

    metrics["samples"] = int(len(X))
    config.METRICS_FILE.write_text(json.dumps(metrics, indent=2) + "\n")
    plots.save_all(metrics)
    print(f"Saved {config.MODEL_FILE.relative_to(config.ROOT_DIR)}, "
          f"{config.METRICS_FILE.relative_to(config.ROOT_DIR)} and figures in "
          f"{config.FIGURES_DIR.relative_to(config.ROOT_DIR)}")


if __name__ == "__main__":
    main()
