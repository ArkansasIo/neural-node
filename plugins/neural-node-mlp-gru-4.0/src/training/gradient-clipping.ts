export function globalGradientNorm(gradients: number[][]): number {
  let sum = 0;
  for (const gradient of gradients) for (const value of gradient) sum += value * value;
  return Math.sqrt(sum);
}

export function clipGradients(gradients: number[][], maxNorm: number): number {
  if (maxNorm <= 0 || !Number.isFinite(maxNorm)) throw new Error("maxNorm must be positive and finite");
  const norm = globalGradientNorm(gradients);
  if (norm === 0 || norm <= maxNorm) return norm;
  const scale = maxNorm / norm;
  for (const gradient of gradients) for (let i = 0; i < gradient.length; i++) gradient[i] *= scale;
  return norm;
}
