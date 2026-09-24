import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { Classifier, Smoother, normalizeLandmarks } from "../js/classifier.js";

const fixtures = new URL("../../ml/tests/fixtures/", import.meta.url);
const readJson = (url) => JSON.parse(readFileSync(url, "utf8"));

test("normalizeLandmarks matches the Python implementation", () => {
  const { landmarks, width, height, expected } = readJson(new URL("normalization.json", fixtures));
  const points = landmarks.map(([x, y, z]) => ({ x, y, z }));
  const result = normalizeLandmarks(points, width, height);
  result.forEach((v, i) => assert.ok(Math.abs(v - expected[i]) < 1e-5, `feature ${i}: ${v} != ${expected[i]}`));
});

test("Classifier reproduces the Keras model's output", () => {
  const { layers } = readJson(new URL("../model/classifier.json", import.meta.url));
  const { inputs, probabilities } = readJson(new URL("classifier_samples.json", fixtures));
  const classifier = new Classifier(layers);
  inputs.forEach((input, n) => {
    const result = classifier.predict(input);
    result.forEach((v, i) => assert.ok(Math.abs(v - probabilities[n][i]) < 1e-4, `sample ${n}, class ${i}`));
  });
});

test("Smoother averages only the most recent predictions", () => {
  const smoother = new Smoother(2);
  assert.equal(smoother.average(), null);
  smoother.push([1, 0]);
  smoother.push([0, 1]);
  smoother.push([0, 1]);
  assert.equal(smoother.count, 2);
  assert.deepEqual(Array.from(smoother.average()), [0, 1]);
});
