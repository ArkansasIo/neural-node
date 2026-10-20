export * from "./core/tensor.js";
export * from "./core/activations.js";
export * from "./core/initializer.js";
export * from "./layers/dense.js";
export * from "./layers/mlp.js";
export * from "./layers/gru.js";
export * from "./layers/dropout.js";
export * from "./layers/layer-norm.js";
export * from "./models/gru-mlp.js";
export * from "./training/loss.js";
export * from "./training/optimizer.js";
export * from "./training/adam.js";
export * from "./training/sgd.js";
export * from "./training/trainer.js";
export * from "./data/dataset.js";
export * from "./data/normalizer.js";
export * from "./serialization/model-json.js";

export const NEURAL_NODE_MLP_GRU_VERSION = "4.0.0-alpha.1";

export interface NeuralNodePlugin {
  name: string;
  version: string;
  architectures: string[];
}

export function createPlugin(): NeuralNodePlugin {
  return {
    name: "neural-node-mlp-gru",
    version: NEURAL_NODE_MLP_GRU_VERSION,
    architectures: ["MLP", "GRU", "GRU-MLP"],
  };
}
