export interface RegisteredModel {
  name: string;
  version: string;
  architecture: string;
  model: unknown;
}

export class ModelRegistry {
  private readonly models = new Map<string, RegisteredModel>();

  register(model: RegisteredModel): void {
    if (!model.name.trim()) throw new Error("model name must not be empty");
    if (this.models.has(model.name)) throw new Error(`model already registered: ${model.name}`);
    this.models.set(model.name, model);
  }

  get(name: string): RegisteredModel | undefined {
    return this.models.get(name);
  }

  require(name: string): RegisteredModel {
    const model = this.get(name);
    if (!model) throw new Error(`model not found: ${name}`);
    return model;
  }

  has(name: string): boolean {
    return this.models.has(name);
  }

  names(): string[] {
    return [...this.models.keys()];
  }

  remove(name: string): boolean {
    return this.models.delete(name);
  }

  clear(): void {
    this.models.clear();
  }
}
