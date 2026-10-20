export class MinMaxNormalizer {
  private min = 0;
  private max = 1;
  private fitted = false;

  fit(values: number[]): void {
    if (values.length === 0) throw new Error("values must not be empty");
    if (values.some(v => !Number.isFinite(v))) throw new Error("values must be finite");
    this.min = Math.min(...values);
    this.max = Math.max(...values);
    this.fitted = true;
  }

  transform(value: number): number {
    if (!this.fitted) throw new Error("normalizer is not fitted");
    if (this.max === this.min) return 0;
    return (value - this.min) / (this.max - this.min);
  }

  inverse(value: number): number {
    if (!this.fitted) throw new Error("normalizer is not fitted");
    return value * (this.max - this.min) + this.min;
  }

  transformVector(values: number[]): number[] { return values.map(v => this.transform(v)); }
}
