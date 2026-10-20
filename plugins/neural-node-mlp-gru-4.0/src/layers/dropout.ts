import { Vector, assertVector } from "../core/tensor.js";

export class DropoutLayer {
  readonly rate: number;
  constructor(rate = 0.1) {
    if (rate < 0 || rate >= 1) throw new Error("dropout rate must be in [0, 1)");
    this.rate = rate;
  }

  forward(input: Vector, training = false, rng: () => number = Math.random): Vector {
    assertVector(input, "dropout input");
    if (!training || this.rate === 0) return [...input];
    const scale = 1 / (1 - this.rate);
    return input.map(value => rng() < this.rate ? 0 : value * scale);
  }
}
