import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import readline from "node:readline";
import { chromium } from "playwright";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const VENDOR = path.join(__dirname, "node_modules", "@mediapipe", "holistic");
const INBOX = path.join(__dirname, "inbox");
const REQUIRE_GPU = process.env.HOLISTIC_WEB_REQUIRE_GPU !== "0";
const HEADED = process.env.HOLISTIC_WEB_HEADED === "1";

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".wasm": "application/wasm",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".data": "application/octet-stream",
  ".binarypb": "application/octet-stream",
  ".tflite": "application/octet-stream",
};

function log(msg) {
  process.stderr.write(`[holistic-web] ${msg}\n`);
}

function reply(obj) {
  process.stdout.write(JSON.stringify(obj) + "\n");
}

function startStaticServer() {
  return new Promise((resolve, reject) => {
    const server = http.createServer((req, res) => {
      const url = new URL(req.url || "/", "http://127.0.0.1");
      let rel = url.pathname;
      if (rel === "/") rel = "/extractor.html";
      if (rel === "/favicon.ico") {
        res.writeHead(204);
        res.end();
        return;
      }
      let filePath;
      let allowedRoot = __dirname;
      if (rel.startsWith("/vendor/")) {
        filePath = path.join(VENDOR, rel.slice("/vendor/".length));
        allowedRoot = VENDOR;
      } else if (rel.startsWith("/inbox/")) {
        filePath = path.join(INBOX, rel.slice("/inbox/".length));
        allowedRoot = INBOX;
      } else {
        filePath = path.join(__dirname, rel.replace(/^\/+/, ""));
      }
      filePath = path.normalize(filePath);
      if (!filePath.startsWith(allowedRoot)) {
        res.writeHead(403);
        res.end("forbidden");
        return;
      }
      fs.readFile(filePath, (err, data) => {
        if (err) {
          if (!rel.startsWith("/inbox/")) log(`404 ${rel} -> ${filePath}`);
          res.writeHead(404);
          res.end("not found");
          return;
        }
        const ext = path.extname(filePath);
        res.writeHead(200, {
          "Content-Type": MIME[ext] || "application/octet-stream",
          "Cache-Control": "no-store",
        });
        res.end(data);
      });
    });
    server.listen(0, "127.0.0.1", () => {
      const addr = server.address();
      resolve({ server, port: addr.port });
    });
    server.on("error", reject);
  });
}

const GPU_ARGS = [
  "--ignore-gpu-blocklist",
  "--enable-gpu",
  "--enable-webgl",
  "--enable-webgl2",
  "--use-gl=angle",
  "--use-angle=d3d11",
  "--disable-software-rasterizer",
  "--disable-gpu-sandbox",
  "--disable-features=Translate,MediaRouter",
];

async function launchGpuBrowser() {
  const attempts = [
    { channel: "chrome", label: "Google Chrome" },
    { channel: "msedge", label: "Microsoft Edge" },
    { label: "Playwright Chromium" },
  ];
  let lastError = null;
  for (const attempt of attempts) {
    try {
      const browser = await chromium.launch({
        channel: attempt.channel,
        headless: !HEADED,
        args: GPU_ARGS,
        ignoreDefaultArgs: ["--disable-gpu"],
      });
      log(`browser: ${attempt.label}`);
      return browser;
    } catch (err) {
      lastError = err;
      log(`no se pudo abrir ${attempt.label}: ${err.message}`);
    }
  }
  throw new Error(
    `No hay Chrome/Edge/Chromium. Último error: ${lastError?.message || "desconocido"}`
  );
}

async function main() {
  if (!fs.existsSync(path.join(VENDOR, "holistic.js"))) {
    throw new Error(
      "Falta node_modules/@mediapipe/holistic. En src/holistic_web corré: npm install"
    );
  }
  fs.mkdirSync(INBOX, { recursive: true });

  const { server, port } = await startStaticServer();
  const origin = `http://127.0.0.1:${port}`;
  log(`static ${origin}`);

  let browser = await launchGpuBrowser();
  let page = null;
  let initOpts = {
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5,
  };
  let ready = false;
  let crashed = false;

  async function attachPage(p) {
    p.setDefaultTimeout(60000);
    p.on("console", (msg) => log(`page: ${msg.type()} ${msg.text()}`));
    p.on("pageerror", (err) => log(`pageerror: ${err.message}`));
    p.on("crash", () => {
      crashed = true;
      log("page crash");
    });
  }

  async function openPage() {
    if (page) {
      try {
        await page.close();
      } catch {
        /* ignore */
      }
    }
    page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
    await attachPage(page);
    await page.goto(`${origin}/extractor.html`, { waitUntil: "load", timeout: 60000 });
    crashed = false;
  }

  async function recycleBrowser() {
    log("recycle browser");
    try {
      await browser.close();
    } catch {
      /* ignore */
    }
    browser = await launchGpuBrowser();
    await openPage();
    const gpuAfter = await page.evaluate(async (opts) => window.__mpBoot(opts), initOpts);
    ready = true;
    return gpuAfter;
  }

  await openPage();
  const gpu = await page.evaluate(() => window.__mpGpuInfo());
  if (REQUIRE_GPU && !gpu.ok) {
    throw new Error(
      `WebGL no está en GPU (vendor=${gpu.vendor || "?"}, renderer=${gpu.renderer || gpu.reason}). ` +
        "Instalá Chrome estable y desactivá el renderizado por software."
    );
  }
  log(`webgl vendor=${gpu.vendor} renderer=${gpu.renderer}`);

  reply({
    ok: true,
    gpu,
    package: "@mediapipe/holistic@0.5.1675471629",
    delegate: gpu.ok ? "GPU" : "CPU",
  });

  const rl = readline.createInterface({ input: process.stdin, terminal: false });

  const shutdown = async () => {
    try {
      await browser.close();
    } catch {
      /* ignore */
    }
    server.close();
    process.exit(0);
  };

  for await (const line of rl) {
    if (!line.trim()) continue;
    let msg;
    try {
      msg = JSON.parse(line);
    } catch (err) {
      reply({ error: `JSON inválido: ${err.message}` });
      continue;
    }
    try {
      if (msg.cmd === "close") {
        await shutdown();
        return;
      }
      if (msg.cmd === "init") {
        initOpts = {
          minDetectionConfidence: msg.minDetectionConfidence ?? 0.5,
          minTrackingConfidence: msg.minTrackingConfidence ?? 0.5,
        };
        const gpuAfter = await page.evaluate(async (opts) => window.__mpBoot(opts), initOpts);
        ready = true;
        reply({ ok: true, gpu: gpuAfter });
        continue;
      }
      if (msg.cmd === "recycle") {
        const gpuAfter = await recycleBrowser();
        reply({ ok: true, gpu: gpuAfter, recycled: true });
        continue;
      }
      if (!ready) {
        reply({ error: "Holistic no inicializado (falta cmd=init)" });
        continue;
      }
      if (crashed || page.isClosed()) {
        await recycleBrowser();
      }
      if (msg.cmd === "reset") {
        await page.evaluate(() => window.__mpReset());
        reply({ ok: true });
        continue;
      }
      if (msg.cmd === "frame") {
        const file = path.basename(String(msg.file || "frame.jpg"));
        const landmarks = await page.evaluate(async (url) => {
          return window.__mpProcessDataUrl(url);
        }, `${origin}/inbox/${file}?t=${Date.now()}`);
        reply(landmarks);
        continue;
      }
      reply({ error: `cmd desconocido: ${msg.cmd}` });
    } catch (err) {
      const text = err.message || String(err);
      log(`cmd ${msg.cmd} failed: ${text}`);
      if (/crash|closed|destroyed/i.test(text)) {
        try {
          await recycleBrowser();
        } catch (recycleErr) {
          reply({ error: `crash y recycle falló: ${recycleErr.message}` });
          continue;
        }
      }
      reply({ error: text });
    }
  }
  await shutdown();
}

main().catch((err) => {
  log(err.stack || err.message);
  reply({ ok: false, error: err.message || String(err) });
  process.exit(1);
});
