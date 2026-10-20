import { GRUMLPModel, meanAbsoluteError } from "../src/index.js";

const model = new GRUMLPModel({
  sequence: { inputSize: 4, hiddenSize: 16 },
  staticInputSize: 3,
  headHiddenSizes: [16, 8],
  outputSize: 1,
});

const sequence = [
  [0.1, 0.2, 0.3, 0.4],
  [0.2, 0.3, 0.4, 0.5],
  [0.3, 0.4, 0.5, 0.6],
];
const staticFeatures = [0.25, 0.5, 0.75];
const prediction = model.forward(sequence, staticFeatures);

console.log({ prediction, maeAgainstExample: meanAbsoluteError(prediction, [0.5]) });
