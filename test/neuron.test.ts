import test from "node:test";
import assert from "node:assert/strict";
import { activate, runNeuron } from "../src/neuron.js";

test("computes weighted sum and ReLU activation", () => {
  assert.deepEqual(
    runNeuron({ inputs: [1, 2], weights: [0.5, 0.25], activation: "relu" }),
    { value: 1, weightedSum: 1, activation: "relu" },
  );
});

test("supports sigmoid for very large positive and negative values", () => {
  assert.equal(activate(1000, "sigmoid"), 1);
  assert.equal(activate(-1000, "sigmoid"), 0);
});

test("supports tanh and linear activation", () => {
  assert.equal(activate(2, "linear"), 2);
  assert.equal(activate(0, "tanh"), 0);
});

test("rejects mismatched input and weight counts", () => {
  assert.throws(
    () => runNeuron({ inputs: [1, 2], weights: [1] }),
    /same length/,
  );
});

test("rejects empty and non-finite arrays", () => {
  assert.throws(() => runNeuron({ inputs: [], weights: [] }), /non-empty/);
  assert.throws(() => runNeuron({ inputs: [Number.NaN], weights: [1] }), /finite/);
});

test("rejects non-finite bias", () => {
  assert.throws(() => runNeuron({ inputs: [1], weights: [1], bias: Infinity }), /bias/);
});
