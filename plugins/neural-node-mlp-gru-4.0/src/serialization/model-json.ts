export interface SerializableModel { toJSON(): unknown; }

export function serializeModel(model: SerializableModel): string {
  return JSON.stringify(model.toJSON(), null, 2);
}

export function parseModel<T>(text: string): T {
  const value: unknown = JSON.parse(text);
  return value as T;
}

export function saveMetadata(name: string, version: string, architecture: string): Record<string, string> {
  return { name, version, architecture, format: "neural-node-model-json-v1" };
}
