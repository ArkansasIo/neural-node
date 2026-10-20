export function meanAbsoluteError(prediction: number[], target: number[]): number {
  assertSameLength(prediction, target);
  if (prediction.length === 0) throw new Error("metric vectors must not be empty");
  return prediction.reduce((sum, value, i) => sum + Math.abs(value - target[i]), 0) / prediction.length;
}

export function rootMeanSquaredError(prediction: number[], target: number[]): number {
  assertSameLength(prediction, target);
  if (prediction.length === 0) throw new Error("metric vectors must not be empty");
  const mse = prediction.reduce((sum, value, i) => sum + (value - target[i]) ** 2, 0) / prediction.length;
  return Math.sqrt(mse);
}

export function accuracy(prediction: number[], target: number[]): number {
  assertSameLength(prediction, target);
  if (prediction.length === 0) throw new Error("metric vectors must not be empty");
  let correct = 0;
  for (let i = 0; i < prediction.length; i++) if (prediction[i] === target[i]) correct++;
  return correct / prediction.length;
}

export function binaryAccuracy(probabilities: number[], target: number[], threshold = 0.5): number {
  if (threshold < 0 || threshold > 1) throw new Error("threshold must be between 0 and 1");
  assertSameLength(probabilities, target);
  if (probabilities.length === 0) throw new Error("metric vectors must not be empty");
  let correct = 0;
  for (let i = 0; i < probabilities.length; i++) {
    if ((probabilities[i] >= threshold ? 1 : 0) === target[i]) correct++;
  }
  return correct / probabilities.length;
}

function assertSameLength(a: number[], b: number[]): void {
  if (a.length !== b.length) throw new Error("metric vector length mismatch");
}
