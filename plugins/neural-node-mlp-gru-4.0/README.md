# Neural Node MLP-GRU 4.0 Plugin

CPU-first TypeScript plugin for Neural Node with MLP, GRU, and hybrid GRU-MLP primitives. The architecture follows the common pattern of using a GRU to encode sequential features and an MLP head to decode the resulting representation.

## Folder layout

```text
neural-node-mlp-gru-4.0/
├── plugin.json
├── package.json
├── tsconfig.json
├── README.md
├── examples/
│   └── basic.ts
└── src/
    ├── index.ts
    ├── index.test.ts
    ├── core/
    │   ├── tensor.ts
    │   ├── activations.ts
    │   └── initializer.ts
    ├── layers/
    │   ├── dense.ts
    │   ├── mlp.ts
    │   ├── gru.ts
    │   ├── dropout.ts
    │   └── layer-norm.ts
    ├── models/
    │   └── gru-mlp.ts
    ├── training/
    │   ├── loss.ts
    │   ├── optimizer.ts
    │   ├── adam.ts
    │   ├── sgd.ts
    │   └── trainer.ts
    ├── data/
    │   ├── dataset.ts
    │   └── normalizer.ts
    └── serialization/
        └── model-json.ts
```

## Architecture

```text
Temporal sequence ──> GRU encoder ──┐
                                    ├─> concatenate ─> MLP head ─> output
Static features ────────────────────┘
```

## Included

- Dense fully-connected layers
- Configurable MLP
- GRU update/reset gates and recurrent hidden state
- Hybrid GRU + MLP model
- Dropout
- Layer normalization
- Xavier/He initialization helpers
- MSE, BCE and softmax utilities
- Adam and SGD optimizer implementations
- Generic trainer scaffold
- Sequence dataset and min/max normalization
- JSON model serialization helpers
- Plugin manifest and example
- TypeScript declarations via the build

## Build

```bash
cd plugins/neural-node-mlp-gru-4.0
npm run typecheck
npm run build
```

The current trainer is deliberately framework-neutral. The MLP and GRU layers currently provide forward inference; full automatic differentiation/BPTT and parameter-gradient plumbing are the next training-engine layer.

## Version

`4.0.0-alpha.1`

This is a Neural Node plugin version, not an industry-wide neural-network standard.
