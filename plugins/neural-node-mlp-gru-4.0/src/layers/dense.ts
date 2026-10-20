import { Vector, assertVector, dot, relu } from "../core/tensor.js";

export type DenseActivation = "linear" | "relu" | "tanh" | "sigmoid";

export class DenseLayer {
  readonly inputSize: number;
  readonly outputSize: number;
  readonly weights: number[][];
  readonly bias: Vector;
  readonly activation: DenseActivation;

  constructor(inputSize: number, outputSize: number, activation: DenseActivation = "linear") {
    if (inputSize <= 0 || outputSize <= 0) throw new Error("layer dimensions must be positive");
    this.inputSize = inputSize;
    this.outputSize = outputSize;
    this.activation = activation;
    const scale = Math.sqrt(2 / inputSize);
    this.weights = Array.from({ length: outputSize }, () =>
      Array.from({ length: inputSize }, () => (Math.random() * 2 - 1) * scale),
    );
    this.bias = new Array(outputSize).fill(0);
  }

  forward(input: Vector): Vector {
    assertVector(input, "input");
    if (input.length !== this.inputSize) throw new Error("dense input dimension mismatch");
    return this.weights.map((row, i) => this.activate(dot(row, input) + this.bias[i]));
  }

  private activate(x: number): number {
    switch (this.activation) {
      case "relu": return relu(x);
      case "tanh": return Math.tanh(x);
      case "sigmoid": return 1 / (1 + Math.exp(-x));
      default: return x;
    }
  }
}
