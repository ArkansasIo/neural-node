# Neural Node MLP-GRU 4.0 Plugin

A CPU-first TypeScript plugin that adds MLP, GRU, and hybrid GRU+MLP inference primitives to Neural Node.

## Architecture

```text
Sequence Input ──> GRU Encoder ──┐
                                  ├─> Concatenate ─> MLP Head ─> Output
Static Input ─────────────────────┘
```

The plugin is intentionally dependency-light so it can run in Node.js, tests, servers, or be embedded into a larger Neural Node runtime.

## Modules

- `src/core/tensor.ts` — small numeric vector helpers
- `src/layers/dense.ts` — fully connected layer
- `src/layers/mlp.ts` — configurable multilayer perceptron
- `src/layers/gru.ts` — single-layer GRU sequence encoder
- `src/models/gru-mlp.ts` — hybrid sequential + static model
- `src/index.ts` — public plugin API

## Version

4.0.0-alpha.1

This is a Neural Node plugin version, not an industry-wide neural-network standard.
