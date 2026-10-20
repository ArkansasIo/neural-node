import { strict as assert } from "node:assert";
import test from "node:test";
import { ModelRegistry, binaryAccuracy, clipGradients, meanAbsoluteError, splitDataset } from "../src/index.js";

test("dataset splitting preserves all samples", () => {
  const split = splitDataset([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.2, 0.1);
  assert.equal(split.train.length + split.validation.length + split.test.length, 10);
});

test("metrics return expected values", () => {
  assert.equal(meanAbsoluteError([1, 3], [2, 2]), 1);
  assert.equal(binaryAccuracy([0.1, 0.9, 0.8], [0, 1, 1]), 1);
});

test("gradient clipping limits global norm", () => {
  const gradients = [[3, 4], [0]];
  clipGradients(gradients, 1);
  assert.ok(Math.abs(Math.hypot(gradients[0][0], gradients[0][1]) - 1) < 1e-9);
});

test("model registry stores and retrieves models", () => {
  const registry = new ModelRegistry();
  const model = { name: "demo", version: "1.0.0", architecture: "MLP", model: {} };
  registry.register(model);
  assert.equal(registry.require("demo"), model);
  assert.deepEqual(registry.names(), ["demo"]);
});
