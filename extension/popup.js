/**
 * Popup: estado del motor, salida Meet, landmarks (debug) y reporte.
 */

const REPO_ISSUES =
  "https://github.com/FranciscoVeronING/2026_Proyecto_LSA/issues/new";

const dot = document.getElementById("dot");
const healthText = document.getElementById("health-text");
const pageHint = document.getElementById("page-hint");
const metaMode = document.getElementById("meta-mode");
const debugBlock = document.getElementById("debug-block");
const chkLandmarks = document.getElementById("chk-landmarks");
const reportBox = document.getElementById("report-box");
const reportText = document.getElementById("report-text");
const reportHint = document.getElementById("report-hint");
const outputHint = document.getElementById("output-hint");
const outputField = document.getElementById("output-field");
const inputHint = document.getElementById("input-hint");
const inputField = document.getElementById("input-field");
const pauseField = document.getElementById("pause-field");
const pauseSelect = document.getElementById("pause-select");

let lastHealth = null;
let inMeet = false;

document.getElementById("btn-setup").addEventListener("click", () => {
  chrome.tabs.create({ url: chrome.runtime.getURL("welcome.html") });
});

document.getElementById("btn-report").addEventListener("click", () => {
  reportBox.hidden = !reportBox.hidden;
  if (!reportBox.hidden) reportText.focus();
});

document.getElementById("btn-report-send").addEventListener("click", () => {
  openReport();
});

chkLandmarks.addEventListener("change", () => {
  LsaPrefs.set({ showLandmarks: chkLandmarks.checked });
});

document.querySelectorAll("[data-output]").forEach((btn) => {
  btn.addEventListener("click", () => {
    const outputMode = btn.getAttribute("data-output");
    paintOutput(outputMode);
    LsaPrefs.set({ outputMode });
  });
});

document.querySelectorAll("[data-input]").forEach((btn) => {
  btn.addEventListener("click", () => {
    const hearingInput = btn.getAttribute("data-input");
    paintInput(hearingInput);
    LsaPrefs.set({ hearingInput });
  });
});

pauseSelect.addEventListener("change", () => {
  const utterancePauseSec = normalizePauseSec(pauseSelect.value);
  pauseSelect.value = String(utterancePauseSec);
  LsaPrefs.set({ utterancePauseSec });
});

/**
 * @param {string} hearingInput
 */
function paintInput(hearingInput) {
  const mode = hearingInput === "voice" || hearingInput === "text" ? hearingInput : "both";
  document.querySelectorAll("[data-input]").forEach((btn) => {
    btn.classList.toggle("on", btn.getAttribute("data-input") === mode);
  });
  if (mode === "voice") {
    inputHint.textContent = "Solo el micrófono de Meet. Lo que escribas en el chat no se traduce.";
  } else if (mode === "text") {
    inputHint.textContent = "Solo el chat de Meet. El micrófono no se transcribe.";
  } else {
    inputHint.textContent = "El micrófono de Meet y el chat se traducen a señas.";
  }
}

/**
 * @param {string} outputMode
 */
function paintOutput(outputMode) {
  document.querySelectorAll("[data-output]").forEach((btn) => {
    btn.classList.toggle("on", btn.getAttribute("data-output") === outputMode);
  });
  if (outputMode === "audio") {
    outputHint.textContent =
      "Lee la traducción en voz alta en esta PC. Los demás no ven texto en tu cámara.";
  } else if (outputMode === "both") {
    outputHint.textContent =
      "Subtítulo en tu video y voz en esta PC.";
  } else {
    outputHint.textContent = "El español se pinta en tu cámara; los demás lo leen en tu video.";
  }
}

/**
 * @param {boolean} motorOk
 * @returns {Promise<void>}
 */
async function inspectTab(motorOk) {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  inMeet = Boolean(tab && tab.url && tab.url.startsWith("https://meet.google.com/"));
  if (!motorOk) {
    pageHint.textContent =
      "Para traducir, Encendé ILSA. Si no lo querés, dejalo apagado: Meet funciona normal.";
    return;
  }
  if (inMeet) {
    chrome.runtime.sendMessage({ type: "lsa-meet-sync", tabId: tab.id }).catch(() => {});
    pageHint.textContent =
      "Estás en Meet: la traducción ya está activa. Para pararla, Apagá ILSA en el exe.";
  } else {
    pageHint.textContent =
      "Abrí Google Meet: con ILSA encendido se traduce solo. Para parar, Apagá el exe.";
  }
}

function openReport() {
  const user = (reportText.value || "").trim();
  if (!user) {
    reportHint.textContent = "Escribí qué pasó, aunque sea en una línea.";
    return;
  }
  const manifest = chrome.runtime.getManifest();
  const h = lastHealth;
  const mode =
    h && h.mode === "hearing" ? "Oyente" : h && h.ok ? "Sordo" : "Motor apagado";
  const body = [
    "## Qué pasó",
    user,
    "",
    "## Contexto",
    `- Extensión: ${manifest.version}`,
    `- Motor: ${h && h.ok ? "encendido" : "apagado"}`,
    `- Modo: ${mode}`,
    `- En Meet: ${inMeet ? "sí" : "no"}`,
  ].join("\n");
  const url =
    REPO_ISSUES +
    "?title=" +
    encodeURIComponent("Fallo en ILSA / Meet") +
    "&body=" +
    encodeURIComponent(body);
  chrome.tabs.create({ url });
}

LsaPrefs.get().then((prefs) => {
  chkLandmarks.checked = prefs.showLandmarks;
  paintOutput(prefs.outputMode);
  paintInput(prefs.hearingInput);
  pauseSelect.value = String(normalizePauseSec(prefs.utterancePauseSec));
});

LsaApi.health()
  .then((h) => {
    if (!h.ok) throw new Error("no listo");
    lastHealth = h;
    dot.classList.add("on");
    const signer = h.mode !== "hearing";
    metaMode.textContent = signer ? "Sordo (LSA → español)" : "Oyente (voz/chat → glosas)";
    healthText.textContent = "Motor conectado";
    if (signer) {
      debugBlock.hidden = false;
      outputField.hidden = false;
      pauseField.hidden = false;
      inputField.hidden = false;
      inputHint.textContent = "La captura de voz o chat se usa con ILSA en modo oyente.";
    } else {
      debugBlock.hidden = true;
      outputField.hidden = true;
      pauseField.hidden = true;
      inputField.hidden = false;
    }
    return inspectTab(true);
  })
  .catch(() => {
    lastHealth = null;
    healthText.textContent = "Motor apagado — descargá ILSA en Instalar motor";
    metaMode.textContent = "—";
    outputField.hidden = true;
    pauseField.hidden = true;
    inputField.hidden = false;
    return inspectTab(false);
  });
