export interface DatasetSplit<T> {
  train: T[];
  validation: T[];
  test: T[];
}

export function splitDataset<T>(items: readonly T[], validationRatio = 0.2, testRatio = 0.1): DatasetSplit<T> {
  if (validationRatio < 0 || testRatio < 0 || validationRatio + testRatio >= 1) {
    throw new Error("validationRatio + testRatio must be between 0 and 1");
  }
  const copy = [...items];
  const testCount = Math.floor(copy.length * testRatio);
  const validationCount = Math.floor(copy.length * validationRatio);
  return {
    train: copy.slice(0, copy.length - testCount - validationCount),
    validation: copy.slice(copy.length - testCount - validationCount, copy.length - testCount),
    test: testCount === 0 ? [] : copy.slice(copy.length - testCount),
  };
}

export function shuffled<T>(items: readonly T[], random: () => number = Math.random): T[] {
  const result = [...items];
  for (let i = result.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [result[i], result[j]] = [result[j], result[i]];
  }
  return result;
}
