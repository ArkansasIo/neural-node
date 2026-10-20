export interface ParameterRef {
  value: number[];
  gradient: number[];
}

export interface Optimizer {
  step(parameters: ParameterRef[]): void;
  reset(): void;
}
