import { Optimizer, ParameterRef } from "./optimizer.js";

export class SGD implements Optimizer {
  constructor(readonly learningRate = 0.01, readonly momentum = 0) {
    if (learningRate <= 0) throw new Error("learning rate must be positive");
    if (momentum < 0 || momentum >= 1) throw new Error("momentum must be in [0, 1)");
  }

  private velocity = new WeakMap<ParameterRef, number[]>();

  step(parameters: ParameterRef[]): void {
    for (const parameter of parameters) {
      if (parameter.value.length !== parameter.gradient.length) throw new Error("parameter gradient mismatch");
      let velocity = this.velocity.get(parameter);
      if (!velocity) {
        velocity = new Array(parameter.value.length).fill(0);
        this.velocity.set(parameter, velocity);
      }
      for (let i = 0; i < parameter.value.length; i++) {
        velocity[i] = this.momentum * velocity[i] - this.learningRate * parameter.gradient[i];
        parameter.value[i] += velocity[i];
      }
    }
  }

  reset(): void {
    this.velocity = new WeakMap();
  }
}
