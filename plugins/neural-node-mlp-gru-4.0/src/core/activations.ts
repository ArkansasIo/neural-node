export type ActivationName = "linear" | "relu" | "sigmoid" | "tanh";

export function activate(name: ActivationName, x: number): number {
  switch (name) {
    case "relu": return Math.max(0, x);
    case "sigmoid": return x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x));
    case "tanh": return Math.tanh(x);
    default: return x;
  }
}

export function derivative(name: ActivationName, x: number): number {
  switch (name) {
    case "relu": return x > 0 ? 1 : 0;
    case "sigmoid": { const y = activate("sigmoid", x); return y * (1 - y); }
    case "tanh": { const y = Math.tanh(x); return 1 - y * y; }
    default: return 1;
  }
}
