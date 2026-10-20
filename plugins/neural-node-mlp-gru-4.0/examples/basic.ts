import { GRUMLPModel, softmax } from "../src/index.js";

const model = new GRUMLPModel({
  sequence: { inputSize: 4, hiddenSize: 16 },
  staticInputSize: 3,
  headHiddenSizes: [16, 8],
  outputSize: 3,
});

const logits = model.forward(
  [[0.1, 0.2, 0.3, 0.4], [0.2, 0.1, 0.5, 0.3], [0.4, 0.2, 0.2, 0.1]],
  [1, 0.5, 0.25],
);

console.log({ logits, probabilities: softmax(logits) });
