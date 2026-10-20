export type Vector = number[];

export function assertVector(value: Vector, name: string): void {
  if (!Array.isArray(value) || value.length === 0) {
    throw new Error(`${name} must be a non-empty vector`);
  }
  if (value.some((x) => !Number.isFinite(x))) {
    throw new Error(`${name} contains a non-finite value`);
  }
}

export function zeros(size: number): Vector {
  if (!Number.isInteger(size) || size <= 0) throw new Error("size must be a positive integer");
  return new Array(size).fill(0);
}

export function concat(...vectors: Vector[]): Vector {
  vectors.forEach((v, i) => assertVector(v, `vectors[${i}]`));
  return vectors.flat();
}

export function dot(a: Vector, b: Vector): number {
  if (a.length !== b.length) throw new Error("vector dimensions must match");
  let result = 0;
  for (let i = 0; i < a.length; i++) result += a[i] * b[i];
  return result;
}

export function sigmoid(x: number): number {
  if (x >= 0) {
    const z = Math.exp(-x);
    return 1 / (1 + z);
  }
  const z = Math.exp(x);
  return z / (1 + z);
}

export function tanh(x: number): number {
  return Math.tanh(x);
}

export function relu(x: number): number {
  return Math.max(0, x);
}
