import { Vector, assertVector } from "../core/tensor.js";

export function meanSquaredError(prediction: Vector, target: Vector): number {
  assertVector(prediction, "prediction");
  assertVector(target, "target");
  if (prediction.length !== target.length) throw new Error("MSE dimensions must match");
  return prediction.reduce((sum, value, i) => sum + (value - target[i]) ** 2, 0) / prediction.length;
}

export function mseGradient(prediction: Vector, target: Vector): Vector {
  if (prediction.length !== target.length) throw new Error("MSE dimensions must match");
  return prediction.map((value, i) => 2 * (value - target[i]) / prediction.length);
}

export function binaryCrossEntropy(probability: number, target: number): number {
  const p = Math.min(1 - 1e-12, Math.max(1e-12, probability));
  return -(target * Math.log(p) + (1 - target) * Math.log(1 - p));
}

export function softmax(logits: Vector): Vector {
  assertVector(logits, "logits");
  const max = Math.max(...logits);
  const exps = logits.map(x => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(x => x / sum);
}
