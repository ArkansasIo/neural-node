export interface NeuralNodeConfig {
  seed?: number;
  learningRate?: number;
  batchSize?: number;
  epochs?: number;
  gradientClipNorm?: number;
  shuffle?: boolean;
}

export const DEFAULT_NEURAL_NODE_CONFIG: Required<NeuralNodeConfig> = {
  seed: 0,
  learningRate: 0.001,
  batchSize: 32,
  epochs: 10,
  gradientClipNorm: 5,
  shuffle: true,
};

export function resolveConfig(config: NeuralNodeConfig = {}): Required<NeuralNodeConfig> {
  const resolved = { ...DEFAULT_NEURAL_NODE_CONFIG, ...config };
  if (resolved.learningRate <= 0) throw new Error("learningRate must be positive");
  if (!Number.isInteger(resolved.batchSize) || resolved.batchSize <= 0) throw new Error("batchSize must be positive");
  if (!Number.isInteger(resolved.epochs) || resolved.epochs <= 0) throw new Error("epochs must be positive");
  if (resolved.gradientClipNorm <= 0) throw new Error("gradientClipNorm must be positive");
  return resolved;
}
