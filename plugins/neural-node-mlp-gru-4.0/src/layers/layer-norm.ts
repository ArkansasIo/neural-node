import { Vector, assertVector } from "../core/tensor.js";

export class LayerNorm {
  readonly size: number;
  readonly epsilon: number;
  readonly gamma: Vector;
  readonly beta: Vector;

  constructor(size: number, epsilon = 1e-5) {
    if (!Number.isInteger(size) || size <= 0) throw new Error("layer norm size must be positive");
    this.size = size;
    this.epsilon = epsilon;
    this.gamma = new Array(size).fill(1);
    this.beta = new Array(size).fill(0);
  }

  forward(input: Vector): Vector {
    assertVector(input, "layer norm input");
    if (input.length !== this.size) throw new Error("layer norm dimension mismatch");
    const mean = input.reduce((a, b) => a + b, 0) / this.size;
    const variance = input.reduce((a, b) => a + (b - mean) ** 2, 0) / this.size;
    const scale = 1 / Math.sqrt(variance + this.epsilon);
    return input.map((x, i) => ((x - mean) * scale) * this.gamma[i] + this.beta[i]);
  }
}
