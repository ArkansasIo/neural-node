import { Optimizer } from "./optimizer.js";

export interface TrainingExample<Input, Target> { input: Input; target: Target; }
export interface TrainerConfig { epochs?: number; }

export class Trainer<Input, Target> {
  constructor(private readonly optimizer: Optimizer, private readonly config: TrainerConfig = {}) {}

  train(
    data: TrainingExample<Input, Target>[],
    forward: (input: Input) => Target,
    loss: (prediction: Target, target: Target) => number,
  ): number[] {
    if (data.length === 0) throw new Error("training dataset must not be empty");
    const epochs = this.config.epochs ?? 1;
    if (!Number.isInteger(epochs) || epochs <= 0) throw new Error("epochs must be positive");
    const history: number[] = [];
    for (let epoch = 0; epoch < epochs; epoch++) {
      let total = 0;
      for (const sample of data) total += loss(forward(sample.input), sample.target);
      history.push(total / data.length);
    }
    return history;
  }

  reset(): void { this.optimizer.reset(); }
}
