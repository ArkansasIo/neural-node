export interface SequenceExample {
  sequence: number[][];
  staticFeatures?: number[];
  target: number[];
}

export class SequenceDataset {
  constructor(readonly examples: SequenceExample[]) {
    if (examples.length === 0) throw new Error("dataset must not be empty");
  }

  get size(): number { return this.examples.length; }
  at(index: number): SequenceExample { return this.examples[index]; }

  batches(batchSize: number): SequenceExample[][] {
    if (!Number.isInteger(batchSize) || batchSize <= 0) throw new Error("batchSize must be positive");
    const result: SequenceExample[][] = [];
    for (let i = 0; i < this.examples.length; i += batchSize) result.push(this.examples.slice(i, i + batchSize));
    return result;
  }
}
