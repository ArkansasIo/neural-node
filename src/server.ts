import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { runNeuron, type ActivationName, type NeuronInput } from "./neuron.js";

const HOST = process.env.HOST ?? "127.0.0.1";
const parsedPort = Number(process.env.PORT ?? "3000");
const PORT = Number.isInteger(parsedPort) && parsedPort >= 1 && parsedPort <= 65535
  ? parsedPort
  : 3000;
const MAX_BODY_BYTES = 16 * 1024;

function sendJson(res: ServerResponse, status: number, payload: unknown): void {
  const body = JSON.stringify(payload);
  res.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "content-length": Buffer.byteLength(body),
    "x-content-type-options": "nosniff",
    "cache-control": "no-store",
  });
  res.end(body);
}

async function readJson(req: IncomingMessage): Promise<unknown> {
  const chunks: Buffer[] = [];
  let length = 0;

  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
    length += buffer.length;
    if (length > MAX_BODY_BYTES) {
      throw Object.assign(new Error("Request body exceeds 16 KiB"), { statusCode: 413 });
    }
    chunks.push(buffer);
  }

  try {
    return JSON.parse(Buffer.concat(chunks).toString("utf8")) as unknown;
  } catch {
    throw Object.assign(new Error("Request body must be valid JSON"), { statusCode: 400 });
  }
}

function isActivationName(value: unknown): value is ActivationName {
  return value === "linear" || value === "relu" || value === "sigmoid" || value === "tanh";
}

function isNeuronInput(value: unknown): value is NeuronInput {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const candidate = value as Record<string, unknown>;
  if (!Array.isArray(candidate.inputs) || !Array.isArray(candidate.weights)) return false;
  if (!candidate.inputs.every((n) => typeof n === "number" && Number.isFinite(n))) return false;
  if (!candidate.weights.every((n) => typeof n === "number" && Number.isFinite(n))) return false;
  if (candidate.inputs.length < 1 || candidate.inputs.length !== candidate.weights.length) return false;
  if (candidate.bias !== undefined && (typeof candidate.bias !== "number" || !Number.isFinite(candidate.bias))) return false;
  if (candidate.activation !== undefined && !isActivationName(candidate.activation)) return false;
  return true;
}

const server = createServer(async (req, res) => {
  const method = req.method ?? "GET";
  const pathname = new URL(req.url ?? "/", `http://${req.headers.host ?? "localhost"}`).pathname;

  if (method === "GET" && pathname === "/api/health") {
    sendJson(res, 200, { status: "ok", service: "neural-node" });
    return;
  }
  if (method === "GET" && pathname === "/api/info") {
    sendJson(res, 200, { name: "neural-node", version: "0.1.0", runtime: process.version });
    return;
  }
  if (method === "POST" && pathname === "/api/neuron/activate") {
    try {
      const body = await readJson(req);
      if (!isNeuronInput(body)) {
        sendJson(res, 400, {
          error: "Invalid neuron input",
          details: "Provide equal-length, non-empty finite-number inputs and weights; bias must be finite; activation must be linear, relu, sigmoid, or tanh.",
        });
        return;
      }
      try {
        sendJson(res, 200, runNeuron(body));
      } catch (error) {
        const err = error as Error;
        sendJson(res, 400, { error: err.message || "Invalid neuron input" });
      }
    } catch (error) {
      const err = error as Error & { statusCode?: number };
      sendJson(res, err.statusCode ?? 400, { error: err.message || "Bad request" });
    }
    return;
  }

  if (method === "GET" && pathname === "/") {
    sendJson(res, 200, {
      name: "neural-node",
      endpoints: ["GET /api/health", "GET /api/info", "POST /api/neuron/activate"],
    });
    return;
  }

  sendJson(res, 404, { error: "Not found", path: pathname });
});

server.listen(PORT, HOST, () => {
  console.log(`neural-node listening on http://${HOST}:${PORT}`);
});

function shutdown(signal: NodeJS.Signals): void {
  console.log(`Received ${signal}; shutting down`);
  server.close((error) => {
    if (error) {
      console.error(error);
      process.exitCode = 1;
    }
  });
  setTimeout(() => {
    console.error("Graceful shutdown timed out");
    process.exit(1);
  }, 5_000).unref();
}

process.on("SIGINT", () => shutdown("SIGINT"));
process.on("SIGTERM", () => shutdown("SIGTERM"));
