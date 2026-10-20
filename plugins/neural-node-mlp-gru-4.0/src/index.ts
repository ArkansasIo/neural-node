export * from "./core/tensor.js";
export * from "./layers/dense.js";
export * from "./layers/mlp.js";
export * from "./layers/gru.js";
export * from "./models/gru-mlp.js";

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
