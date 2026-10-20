# Neural Node MLP-GRU 4.0 Plugin

CPU-first TypeScript plugin for Neural Node with MLP, GRU, and hybrid GRU-MLP primitives. The hybrid model combines a GRU sequence encoder with an MLP prediction head, while exposing a separate trainable GRU-MLP implementation and optimizer-based training utilities.

## Folder layout

```text
neural-node-mlp-gru-4.0/
├── plugin.json
├── package.json
├── tsconfig.json
├── README.md
├── docs/
│   └── ARCHITECTURE.md
├── examples/
│   ├── basic.ts
│   └── gru-mlp.ts
├── src/
│   ├── index.ts
│   ├── trainable.ts
│   ├── metrics.ts
│   ├── config.ts
│   ├── core/
│   ├── data/
│   ├── layers/
│   ├── models/
│   ├── serialization/
│   └── training/
└── test/
    ├── plugin.test.ts
    └── trainable.test.ts
```

## Architecture

```text
Temporal sequence ──> GRU encoder ──┐
                                    ├─> concatenate ─> MLP head ─> output
Static features ────────────────────┘
```

## Features

- Dense layers and configurable MLPs
- GRU update/reset gates and recurrent hidden state
- Hybrid GRU + MLP model for sequential and static features
- Trainable GRU-MLP model with explicit backward-pass gradient calculation
- Mean-squared-error training utility and optimizer integration
- Adam and SGD optimizers
- Global gradient-norm calculation and gradient clipping
- Dropout, layer normalization, and initialization helpers
- Sequence datasets, normalization, and dataset splitting
- Model metrics and JSON serialization utilities
- Plugin manifest, examples, TypeScript declarations, and automated tests

## Requirements

- Node.js 20 or newer
- npm 10 or newer

## Build and test

Run commands from the plugin directory:

```bash
cd plugins/neural-node-mlp-gru-4.0
npm install
npm run typecheck
npm run build
npm test
```

`npm test` compiles the plugin and tests, then runs the Node.js test runner. The plugin CI workflow runs typecheck, production compilation, and tests when files under this directory change.

## Minimal use

Import the model and trainer from the plugin entry point:

```ts
import {
  Adam,
  TrainableGRUMLP,
  TrainableTrainer,
} from "./src/index.js";

const model = new TrainableGRUMLP({
  sequenceInputSize: 2,
  hiddenSize: 4,
  staticInputSize: 1,
  headHiddenSizes: [6],
  outputSize: 1,
});

const trainer = new TrainableTrainer(new Adam({ learningRate: 0.001 }), {
  epochs: 10,
  clipNorm: 5,
});

const history = trainer.train(model, [
  {
    sequence: [[0.1, 0.2], [0.2, 0.3]],
    staticFeatures: [0.5],
    target: [0.25],
  },
]);

console.log(history.at(-1)?.loss);
```

## Training scope and limitations

The trainable GRU-MLP path includes an explicit backward pass and optimizer updates. It is a small CPU-first research/learning implementation, not a mature deep-learning framework. Validate gradients with numerical gradient checks and representative learning tasks before relying on training results. The implementation does not provide GPU acceleration, automatic differentiation, batching, or a production model-serving lifecycle.

This plugin is ordinary software for neural-network experiments. It does not connect to, record from, stimulate, or control biological neural tissue or medical devices. Do not use it for diagnosis or safety-critical control.

## Version

`4.0.0-alpha.2`

This is the plugin's own version number, not an industry-wide neural-network standard.
