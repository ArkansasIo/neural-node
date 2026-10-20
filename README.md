# Neural Node

A small, testable TypeScript foundation for experimenting with neural-network nodes and inference pipelines. The current scaffold is intentionally CPU-only and dependency-light; it is not a trained model or a biological-neural simulation.

## Requirements

- Node.js 20 or newer
- npm 10 or newer

## Quick start

```bash
npm install
npm test
npm run build
npm start
```

The HTTP server listens on `127.0.0.1:3000` by default. Configure `HOST` and `PORT` with environment variables.

## API

- `GET /api/health` — health/readiness response
- `GET /api/info` — runtime name and version
- `POST /api/neuron/activate` — apply a weighted sum and activation function

Example:

```bash
curl -s http://127.0.0.1:3000/api/health
curl -s -X POST http://127.0.0.1:3000/api/neuron/activate \
  -H 'content-type: application/json' \
  -d '{"inputs":[1,2],"weights":[0.5,0.25],"bias":0,"activation":"relu"}'
```

Expected activation result: `{"value":1}`.

## Safety and scope

This repository provides software scaffolding for experimentation. It does not connect to, record from, stimulate, or control biological neural tissue or medical devices. Do not use it for diagnosis or safety-critical control.

## Project structure

- `src/neuron.ts` — pure neuron math and validated activation functions
- `src/server.ts` — HTTP API and request validation
- `test/neuron.test.ts` — node:test unit tests
- `.github/workflows/ci.yml` — reproducible CI checks

## Development

```bash
npm run dev
npm test
npm run build
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution checks and conventions.
