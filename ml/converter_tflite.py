import os
import shutil

import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KERAS_MODEL_FILE = os.path.join(BASE_DIR, "modelo_gestos_lgp.keras")
DATA_FILE = os.path.join(BASE_DIR, "dados_processados.npz")
CLASSES_FILE = os.path.join(BASE_DIR, "classes.json")
ANDROID_ASSETS = os.path.join(BASE_DIR, "..", "android", "app", "src", "main", "assets")
TFLITE_MODEL_FILE = os.path.join(ANDROID_ASSETS, "modelo_gestos_lgp.tflite")


def check_equivalence(model, tflite_model):
    """Confirma que o modelo TFLite dá as mesmas previsões que o modelo Keras."""
    if not os.path.exists(DATA_FILE):
        return
    X = np.load(DATA_FILE)["X"].reshape(-1, 63).astype(np.float32)

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    tflite_probs = []
    for sample in X:
        interpreter.set_tensor(input_index, sample[None])
        interpreter.invoke()
        tflite_probs.append(interpreter.get_tensor(output_index)[0])

    keras_probs = model.predict(X, verbose=0)
    max_diff = np.abs(np.array(tflite_probs) - keras_probs).max()
    agreement = (np.argmax(tflite_probs, axis=1) == keras_probs.argmax(axis=1)).mean()
    print(f"Keras vs TFLite: diferença máxima {max_diff:.2e}, mesma previsão em {agreement:.1%} das amostras")


def main():
    if not os.path.exists(KERAS_MODEL_FILE):
        print(f"Erro: '{os.path.basename(KERAS_MODEL_FILE)}' não encontrado. Execute primeiro 'treinar_modelo.py'.")
        return

    model = tf.keras.models.load_model(KERAS_MODEL_FILE)
    tflite_model = tf.lite.TFLiteConverter.from_keras_model(model).convert()

    os.makedirs(ANDROID_ASSETS, exist_ok=True)
    with open(TFLITE_MODEL_FILE, "wb") as f:
        f.write(tflite_model)
    shutil.copy(CLASSES_FILE, ANDROID_ASSETS)

    print(f"Modelo TFLite ({len(tflite_model) / 1024:.0f} KB) e classes copiados para "
          f"'{os.path.relpath(ANDROID_ASSETS, BASE_DIR)}'.")
    check_equivalence(model, tflite_model)


if __name__ == "__main__":
    main()
