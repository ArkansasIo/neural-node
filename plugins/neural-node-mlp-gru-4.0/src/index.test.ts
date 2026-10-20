import { strict as assert } from "node:assert";
import test from "node:test";
import { GRU, GRUMLPModel, MLP, createPlugin } from "./index.js";

test("plugin metadata is available", () => {
  const plugin = createPlugin();
  assert.equal(plugin.name, "neural-node-mlp-gru");
  assert.deepEqual(plugin.architectures, ["MLP", "GRU", "GRU-MLP"]);
});

test("MLP produces configured output size", () => {
  const model = new MLP({ inputSize: 3, hiddenSizes: [8, 4], outputSize: 2 });
  assert.equal(model.forward([1, 2, 3]).length, 2);
});

test("GRU consumes a sequence and returns hidden state", () => {
  const gru = new GRU({ inputSize: 2, hiddenSize: 6 });
  assert.equal(gru.forward([[0, 1], [1, 0], [0.5, 0.5]]).length, 6);
});

test("hybrid GRU-MLP model combines temporal and static features", () => {
  const model = new GRUMLPModel({
    sequence: { inputSize: 2, hiddenSize: 4 },
    staticInputSize: 3,
    headHiddenSizes: [8, 4],
    outputSize: 1,
  });
  const output = model.forward([[1, 0], [0, 1]], [0.2, 0.4, 0.6]);
  assert.equal(output.length, 1);
  assert.ok(Number.isFinite(output[0]));
});
