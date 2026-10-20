import { Optimizer, ParameterRef } from "./optimizer.js";

export interface AdamConfig {
  learningRate?: number;
  beta1?: number;
  beta2?: number;
  epsilon?: number;
  weightDecay?: number;
}

export class Adam implements Optimizer {
  private readonly lr: number;
  private readonly beta1: number;
  private readonly beta2: number;
  private readonly epsilon: number;
  private readonly weightDecay: number;
  private stepCount = 0;
  private moments = new WeakMap<ParameterRef, { m: number[]; v: number[] }>();

  constructor(config: AdamConfig = {}) {
    this.lr = config.learningRate ?? 0.001;
    this.beta1 = config.beta1 ?? 0.9;
    this.beta2 = config.beta2 ?? 0.999;
    this.epsilon = config.epsilon ?? 1e-8;
    this.weightDecay = config.weightDecay ?? 0;
    if (this.lr <= 0) throw new Error("learning rate must be positive");
  }

  step(parameters: ParameterRef[]): void {
    this.stepCount++;
    for (const parameter of parameters) {
      if (parameter.value.length !== parameter.gradient.length) throw new Error("parameter gradient mismatch");
      let state = this.moments.get(parameter);
      if (!state) {
        state = { m: new Array(parameter.value.length).fill(0), v: new Array(parameter.value.length).fill(0) };
        this.moments.set(parameter, state);
      }
      const bias1 = 1 - Math.pow(this.beta1, this.stepCount);
      const bias2 = 1 - Math.pow(this.beta2, this.stepCount);
      for (let i = 0; i < parameter.value.length; i++) {
        const g = parameter.gradient[i] + this.weightDecay * parameter.value[i];
        state.m[i] = this.beta1 * state.m[i] + (1 - this.beta1) * g;
        state.v[i] = this.beta2 * state.v[i] + (1 - this.beta2) * g * g;
        const mHat = state.m[i] / bias1;
        const vHat = state.v[i] / bias2;
        parameter.value[i] -= this.lr * mHat / (Math.sqrt(vHat) + this.epsilon);
      }
    }
  }

  reset(): void {
    this.stepCount = 0;
    this.moments = new WeakMap();
  }
}
