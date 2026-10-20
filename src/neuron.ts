export type ActivationName = "linear" | "relu" | "sigmoid" | "tanh";

export interface NeuronInput {
  inputs: number[];
  weights: number[];
  bias?: number;
  activation?: ActivationName;
}

export interface NeuronResult {
  value: number;
  weightedSum: number;
  activation: ActivationName;
}

function assertFiniteArray(values: number[], name: string): void {
  if (!Array.isArray(values) || values.length === 0) {
    throw new TypeError(`${name} must be a non-empty array`);
  }
  if (values.some((value) => typeof value !== "number" || !Number.isFinite(value))) {
    throw new TypeError(`${name} must contain only finite numbers`);
  }
}

export function activate(value: number, activation: ActivationName = "linear"): number {
  if (!Number.isFinite(value)) throw new TypeError("value must be finite");

  switch (activation) {
    case "linear":
      return value;
    case "relu":
      return Math.max(0, value);
    case "sigmoid":
      // Numerically stable sigmoid to avoid overflow for large magnitudes.
      if (value >= 0) {
        const z = Math.exp(-value);
        return 1 / (1 + z);
      } else {
        const z = Math.exp(value);
        return z / (1 + z);
      }
    case "tanh":
      return Math.tanh(value);
    default: {
      const neverActivation: never = activation;
      throw new TypeError(`Unsupported activation: ${String(neverActivation)}`);
    }
  }
}

export function runNeuron(input: NeuronInput): NeuronResult {
  if (!input || typeof input !== "object") {
    throw new TypeError("input must be an object");
  }
  assertFiniteArray(input.inputs, "inputs");
  assertFiniteArray(input.weights, "weights");
  if (input.inputs.length !== input.weights.length) {
    throw new RangeError("inputs and weights must have the same length");
  }

  const bias = input.bias ?? 0;
  if (!Number.isFinite(bias)) throw new TypeError("bias must be finite");
  const activation = input.activation ?? "linear";
  const weightedSum = input.inputs.reduce(
    (sum, value, index) => sum + value * input.weights[index]!,
    bias,
  );
  if (!Number.isFinite(weightedSum)) {
    throw new RangeError("weighted sum is not finite");
  }

  return {
    value: activate(weightedSum, activation),
    weightedSum,
    activation,
  };
}
