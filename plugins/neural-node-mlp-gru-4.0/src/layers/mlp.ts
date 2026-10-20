import { DenseActivation, DenseLayer } from "./dense.js";
import { Vector, assertVector } from "../core/tensor.js";

export interface MLPConfig {
  inputSize: number;
  hiddenSizes: number[];
  outputSize: number;
  hiddenActivation?: DenseActivation;
  outputActivation?: DenseActivation;
}

export class MLP {
  readonly layers: DenseLayer[];

  constructor(config: MLPConfig) {
    const hiddenActivation = config.hiddenActivation ?? "relu";
    const outputActivation = config.outputActivation ?? "linear";
    const sizes = [config.inputSize, ...config.hiddenSizes, config.outputSize];
    this.layers = sizes.slice(0, -1).map((size, i) =>
      new DenseLayer(size, sizes[i + 1], i === sizes.length - 2 ? outputActivation : hiddenActivation),
    );
  }

  forward(input: Vector): Vector {
    assertVector(input, "MLP input");
    return this.layers.reduce((value, layer) => layer.forward(value), input);
  }
}
