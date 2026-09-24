export const NUM_LANDMARKS = 21;
export const NUM_FEATURES = NUM_LANDMARKS * 3;

/**
 * Make hand landmarks independent of where the hand is and how big it looks.
 * Same maths as ml/landmarks.py and the Android app.
 *
 * @param {{x: number, y: number, z: number}[]} landmarks MediaPipe's 21 normalised landmarks
 * @param {number} width  image width in pixels
 * @param {number} height image height in pixels
 */
export function normalizeLandmarks(landmarks, width, height) {
  const wrist = landmarks[0];
  const features = new Float32Array(NUM_FEATURES);
  let scale = 0;
  landmarks.forEach((lm, i) => {
    const x = (lm.x - wrist.x) * width;
    const y = (lm.y - wrist.y) * height;
    features[i * 3] = x;
    features[i * 3 + 1] = y;
    features[i * 3 + 2] = (lm.z - wrist.z) * width;
    scale = Math.max(scale, Math.hypot(x, y));
  });
  scale = Math.max(scale, 1e-6);
  return features.map((v) => v / scale);
}

const activations = {
  relu: (values) => values.map((v) => Math.max(0, v)),
  softmax: (values) => {
    const max = Math.max(...values);
    const exps = values.map((v) => Math.exp(v - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map((v) => v / sum);
  },
  linear: (values) => values,
};

/** The trained dense network, evaluated directly from its exported weights. */
export class Classifier {
  constructor(layers) {
    this.layers = layers.map(({ kernel, bias, activation }) => ({
      kernel: kernel.map((row) => Float32Array.from(row)),
      bias: Float32Array.from(bias),
      activation: activations[activation],
    }));
  }

  static async load(url) {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Could not load ${url}: ${response.status}`);
    return new Classifier((await response.json()).layers);
  }

  predict(features) {
    let values = Float32Array.from(features);
    for (const { kernel, bias, activation } of this.layers) {
      const out = Float32Array.from(bias);
      for (let i = 0; i < values.length; i++) {
        const v = values[i];
        if (v === 0) continue;
        const row = kernel[i];
        for (let j = 0; j < out.length; j++) out[j] += v * row[j];
      }
      values = activation(out);
    }
    return values;
  }
}

/** Averages the last `size` predictions, like the Android app. */
export class Smoother {
  constructor(size) {
    this.size = size;
    this.history = [];
  }

  push(probabilities) {
    this.history.push(probabilities);
    if (this.history.length > this.size) this.history.shift();
  }

  clear() {
    this.history = [];
  }

  get count() {
    return this.history.length;
  }

  average() {
    if (this.history.length === 0) return null;
    const sum = new Float32Array(this.history[0].length);
    for (const p of this.history) p.forEach((v, i) => (sum[i] += v));
    return sum.map((v) => v / this.history.length);
  }
}
