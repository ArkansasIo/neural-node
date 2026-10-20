# Contributing

## Local requirements

Use Node.js 20+ and npm 10+. Install dependencies with `npm install`.

## Required checks

Before submitting a change, run:

```bash
npm run typecheck
npm test
npm run build
```

## Guidelines

- Keep neuron math deterministic and independent of the HTTP layer.
- Validate all untrusted HTTP input; use bounded request bodies.
- Add tests for new activation functions, validation rules, and API behavior.
- Do not commit build output, dependency directories, secrets, tokens, or datasets containing private information.
- Changes involving biological interfaces or medical devices require a separate safety review and are out of scope for this scaffold.
