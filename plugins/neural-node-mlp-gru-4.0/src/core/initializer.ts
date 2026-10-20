export type Initializer = (rows: number, cols: number) => number[][];

export function xavierUniform(rows: number, cols: number, rng: () => number = Math.random): number[][] {
  const limit = Math.sqrt(6 / (rows + cols));
  return Array.from({ length: rows }, () => Array.from({ length: cols }, () => (rng() * 2 - 1) * limit));
}

export function heUniform(rows: number, cols: number, rng: () => number = Math.random): number[][] {
  const limit = Math.sqrt(6 / cols);
  return Array.from({ length: rows }, () => Array.from({ length: cols }, () => (rng() * 2 - 1) * limit));
}

export function zeros(rows: number, cols: number): number[][] {
  return Array.from({ length: rows }, () => new Array(cols).fill(0));
}
