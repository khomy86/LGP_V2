"""Step 3: export the trained model to the Android app (TFLite) and the web demo (JSON)."""
import json

import numpy as np
import tensorflow as tf

import config
from landmarks import NUM_FEATURES

TFLITE_FILE = config.ANDROID_ASSETS_DIR / "gesture_classifier.tflite"
WEB_WEIGHTS_FILE = config.WEB_MODEL_DIR / "classifier.json"
GESTURES_FILENAME = "gestures.json"
FIXTURE_FILE = config.TEST_FIXTURES_DIR / "classifier_samples.json"


def gesture_metadata():
    return {
        "smoothingWindow": config.SMOOTHING_WINDOW,
        "minConfidence": config.MIN_CONFIDENCE,
        "gestures": [
            {
                "id": g,
                "name": config.GESTURES[g]["name"],
                "meaning": config.GESTURES[g]["meaning"],
                "handShape": config.GESTURES[g]["hand_shape"],
            }
            for g in config.ENABLED_GESTURES
        ],
    }


def dense_layers(model):
    layers = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            kernel, bias = layer.get_weights()
            layers.append({
                "units": int(kernel.shape[1]),
                "activation": layer.get_config()["activation"],
                "kernel": np.round(kernel, 6).tolist(),
                "bias": np.round(bias, 6).tolist(),
            })
    return layers


def run_tflite(tflite_model, X):
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    probs = []
    for sample in X:
        interpreter.set_tensor(input_index, sample[None])
        interpreter.invoke()
        probs.append(interpreter.get_tensor(output_index)[0])
    return np.array(probs)


def main():
    if not config.MODEL_FILE.exists():
        print("Model not found. Run train.py first.")
        return

    model = tf.keras.models.load_model(config.MODEL_FILE)
    metadata = json.dumps(gesture_metadata(), indent=2, ensure_ascii=False) + "\n"

    tflite_model = tf.lite.TFLiteConverter.from_keras_model(model).convert()
    config.ANDROID_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    TFLITE_FILE.write_bytes(tflite_model)
    (config.ANDROID_ASSETS_DIR / GESTURES_FILENAME).write_text(metadata, encoding="utf-8")

    config.WEB_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    WEB_WEIGHTS_FILE.write_text(json.dumps({"layers": dense_layers(model)}, separators=(",", ":")) + "\n")
    (config.WEB_MODEL_DIR / GESTURES_FILENAME).write_text(metadata, encoding="utf-8")

    data = np.load(config.LANDMARKS_FILE)
    keep = np.isin(data["labels"], config.ENABLED_GESTURES)
    X = data["X"][keep].reshape(-1, NUM_FEATURES).astype(np.float32)
    keras_probs = model.predict(X, verbose=0)
    tflite_probs = run_tflite(tflite_model, X)
    print(f"TFLite vs Keras: max difference {np.abs(tflite_probs - keras_probs).max():.1e}, "
          f"same prediction for {(tflite_probs.argmax(1) == keras_probs.argmax(1)).mean():.1%} of samples")

    # A few real samples with the expected output, so the web implementation can be tested.
    rng = np.random.default_rng(config.SEED)
    picks = rng.choice(len(X), size=8, replace=False)
    config.TEST_FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    FIXTURE_FILE.write_text(json.dumps({
        "inputs": np.round(X[picks], 6).tolist(),
        "probabilities": np.round(keras_probs[picks], 6).tolist(),
    }) + "\n")

    for path in (TFLITE_FILE, WEB_WEIGHTS_FILE):
        print(f"Wrote {path.relative_to(config.ROOT_DIR)} ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
