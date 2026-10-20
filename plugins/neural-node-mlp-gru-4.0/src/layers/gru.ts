import { Vector, assertVector, sigmoid, tanh, zeros } from "../core/tensor.js";

export interface GRUConfig {
  inputSize: number;
  hiddenSize: number;
}

export class GRU {
  readonly inputSize: number;
  readonly hiddenSize: number;
  private h: Vector;

  // Gate matrices are stored as [hidden][input + hidden].
  private wz: number[][];
  private wr: number[][];
  private wh: number[][];
  private bz: Vector;
  private br: Vector;
  private bh: Vector;

  constructor(config: GRUConfig) {
    if (config.inputSize <= 0 || config.hiddenSize <= 0) throw new Error("GRU dimensions must be positive");
    this.inputSize = config.inputSize;
    this.hiddenSize = config.hiddenSize;
    this.h = zeros(config.hiddenSize);
    const n = config.inputSize + config.hiddenSize;
    const init = () => Array.from({ length: config.hiddenSize }, () => Array.from({ length: n }, () => (Math.random() * 2 - 1) * Math.sqrt(1 / n)));
    this.wz = init();
    this.wr = init();
    this.wh = init();
    this.bz = zeros(config.hiddenSize);
    this.br = zeros(config.hiddenSize);
    this.bh = zeros(config.hiddenSize);
  }

  reset(): void {
    this.h.fill(0);
  }

  step(input: Vector): Vector {
    assertVector(input, "GRU input");
    if (input.length !== this.inputSize) throw new Error("GRU input dimension mismatch");
    const combined = [...input, ...this.h];
    const z = this.wz.map((row, i) => sigmoid(this.dot(row, combined) + this.bz[i]));
    const r = this.wr.map((row, i) => sigmoid(this.dot(row, combined) + this.br[i]));
    const candidateInput = [...input, ...this.h.map((value, i) => value * r[i])];
    const candidate = this.wh.map((row, i) => tanh(this.dot(row, candidateInput) + this.bh[i]));
    this.h = this.h.map((oldValue, i) => (1 - z[i]) * oldValue + z[i] * candidate[i]);
    return [...this.h];
  }

  forward(sequence: Vector[]): Vector {
    if (!Array.isArray(sequence) || sequence.length === 0) throw new Error("GRU sequence must not be empty");
    this.reset();
    for (const input of sequence) this.step(input);
    return [...this.h];
  }

  private dot(a: Vector, b: Vector): number {
    let result = 0;
    for (let i = 0; i < a.length; i++) result += a[i] * b[i];
    return result;
  }
}
