import { strict as assert } from "node:assert";
import test from "node:test";
import { Adam, TrainableGRUMLP, TrainableTrainer, meanSquaredError } from "../src/index.js";

test("trainable GRU-MLP forward/backward returns finite gradients", () => {
  const model = new TrainableGRUMLP({ sequenceInputSize:2, hiddenSize:4, staticInputSize:2, headHiddenSizes:[6], outputSize:1 });
  const prediction = model.forward([[1,0],[0,1],[0.5,0.5]],[0.2,0.8]);
  assert.equal(prediction.length,1);
  const before = model.parameterRefs().map(p => p.value.map(v => v));
  const result = model.backward([prediction[0]-0.5]);
  assert.equal(result.inputGradient.length,3);
  assert.ok(model.parameterRefs().every(p => p.gradient.every(Number.isFinite)));
  assert.ok(before.some((row,i)=>row.some((v,j)=>v!==model.parameterRefs()[i].value[j])) === false);
});

test("trainer performs optimizer updates and records history", () => {
  const model = new TrainableGRUMLP({ sequenceInputSize:1, hiddenSize:3, staticInputSize:1, headHiddenSizes:[4], outputSize:1 });
  const data = Array.from({length:8},(_,i)=>({sequence:[[i/8],[1-i/8]],staticFeatures:[i/8],target:[0.5]}));
  const trainer = new TrainableTrainer(new Adam({learningRate:0.01}),{epochs:3,clipNorm:5});
  const before = model.forward(data[0].sequence,data[0].staticFeatures)[0];
  const history = trainer.train(model,data);
  const after = model.forward(data[0].sequence,data[0].staticFeatures)[0];
  assert.equal(history.length,3);
  assert.ok(history.every(h=>Number.isFinite(h.loss)&&Number.isFinite(h.gradientNorm)));
  assert.ok(Number.isFinite(after));
  assert.notEqual(before,after);
  assert.ok(meanSquaredError([after],[0.5]) <= 1);
});
