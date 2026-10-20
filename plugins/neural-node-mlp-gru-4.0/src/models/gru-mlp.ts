import { concat, Vector, assertVector } from "../core/tensor.js";
import { GRU, GRUConfig } from "../layers/gru.js";
import { MLP, MLPConfig } from "../layers/mlp.js";

export interface GRUMLPConfig {
  sequence: GRUConfig;
  staticInputSize: number;
  headHiddenSizes: number[];
  outputSize: number;
}

export class GRUMLPModel {
  readonly encoder: GRU;
  readonly head: MLP;

  constructor(config: GRUMLPConfig) {
    this.encoder = new GRU(config.sequence);
    const headConfig: MLPConfig = {
      inputSize: config.sequence.hiddenSize + config.staticInputSize,
      hiddenSizes: config.headHiddenSizes,
      outputSize: config.outputSize,
      hiddenActivation: "relu",
      outputActivation: "linear",
    };
    this.head = new MLP(headConfig);
  }

  forward(sequence: Vector[], staticFeatures: Vector): Vector {
    assertVector(staticFeatures, "staticFeatures");
    const temporal = this.encoder.forward(sequence);
    return this.head.forward(concat(temporal, staticFeatures));
  }
}
