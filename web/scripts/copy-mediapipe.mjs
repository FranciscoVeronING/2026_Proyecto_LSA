/**
 * Copia @mediapipe/holistic a public/ para servir en el mismo origen (evita CORS en .wasm/.data).
 */
import { cpSync, existsSync, mkdirSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const src = join(__dirname, "../node_modules/@mediapipe/holistic");
const dest = join(__dirname, "../public/mediapipe/holistic");

if (!existsSync(src)) {
  console.warn("[copy-mediapipe] @mediapipe/holistic no instalado, omitiendo.");
  process.exit(0);
}

mkdirSync(dest, { recursive: true });
for (const name of readdirSync(src)) {
  if (name === "README.md" || name === "package.json" || name === "index.d.ts") continue;
  cpSync(join(src, name), join(dest, name), { force: true });
}
console.log("[copy-mediapipe] Archivos copiados a public/mediapipe/holistic/");
