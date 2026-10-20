# Neural Node MLP-GRU 4.0 Architecture

## Purpose

This plugin provides a CPU-first TypeScript foundation for multilayer perceptrons (MLP), gated recurrent units (GRU), and a hybrid GRU-MLP model.

## Data flow

```text
sequence features ──> GRU encoder ──┐
                                    ├─> feature fusion ──> MLP head ──> output
static features ────────────────────┘
```

## Modules

- `core/`: vectors, activations, initialization.
- `layers/`: Dense, MLP, GRU, Dropout, and LayerNorm primitives.
- `models/`: composite model definitions and model registry.
- `training/`: loss functions, optimizers, trainer, and gradient clipping.
- `data/`: datasets, normalization, splitting, and shuffling utilities.
- `serialization/`: model metadata and JSON serialization helpers.
- `metrics.ts`: regression and classification metrics.
- `config.ts`: validated runtime/training defaults.

## Hybrid model

`GRUMLPModel` encodes a variable-length temporal sequence into a fixed hidden vector, concatenates that vector with static features, and evaluates the fused vector with an MLP head.

## Versioning

The plugin currently identifies itself as `4.0.0-alpha.1`. The `4.0` label is a project version, not an industry-wide neural-network standard.

## Production roadmap

The next training-engine milestone is full automatic differentiation: Dense backward propagation, GRU backpropagation through time (BPTT), hybrid-model gradients, optimizer parameter binding, checkpoints, and deterministic seeded initialization.
