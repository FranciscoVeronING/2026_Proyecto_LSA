/**
 * Popup: estado del motor, salida Meet, landmarks (debug) y reporte.
 */

const REPO_ISSUES =
  "https://github.com/FranciscoVeronING/2026_Proyecto_LSA/issues/new";

const dot = document.getElementById("dot");
const healthText = document.getElementById("health-text");
const pageHint = document.getElementById("page-hint");
const metaMode = document.getElementById("meta-mode");
const metaTranslatorRow = document.getElementById("meta-translator-row");
const metaTranslator = document.getElementById("meta-translator");
const debugBlock = document.getElementById("debug-block");
const chkLandmarks = document.getElementById("chk-landmarks");
const reportBox = document.getElementById("report-box");
const reportText = document.getElementById("report-text");
const reportHint = document.getElementById("report-hint");
const outputHint = document.getElementById("output-hint");

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

document.querySelectorAll(".seg-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    const outputMode = btn.getAttribute("data-output");
    paintOutput(outputMode);
    LsaPrefs.set({ outputMode });
  });
});

/**
 * @param {string} outputMode
 */
function paintOutput(outputMode) {
  document.querySelectorAll(".seg-btn").forEach((btn) => {
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
  const translator = (h && h.semantic_label) || "—";
  const body = [
    "## Qué pasó",
    user,
    "",
    "## Contexto",
    `- Extensión: ${manifest.version}`,
    `- Motor: ${h && h.ok ? "encendido" : "apagado"}`,
    `- Modo: ${mode}`,
    `- Traductor: ${translator}`,
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

/**
 * @param {object} h
 * @returns {string}
 */
function translatorLine(h) {
  const name =
    h.semantic_label ||
    (h.semantic_ready ? "Traducción precisa" : "Cargando el traductor…");
  const load = h.semantic_load;
  if (!load) return name;
  const hint =
    load === "pesado"
      ? "cómputo pesado"
      : load === "liviano"
        ? "cómputo liviano"
        : "cómputo medio";
  return `${name} · ${hint}`;
}

LsaPrefs.get().then((prefs) => {
  chkLandmarks.checked = prefs.showLandmarks;
  paintOutput(prefs.outputMode);
});

LsaApi.health()
  .then((h) => {
    if (!h.ok) throw new Error("no listo");
    lastHealth = h;
    dot.classList.add("on");
    const signer = h.mode !== "hearing";
    metaMode.textContent = signer ? "Sordo (LSA → español)" : "Oyente (voz → subtítulos)";
    healthText.textContent = "Motor conectado";
    if (signer) {
      metaTranslatorRow.hidden = false;
      metaTranslator.textContent = translatorLine(h);
      debugBlock.hidden = false;
      document.getElementById("output-field").hidden = false;
    } else {
      metaTranslatorRow.hidden = true;
      debugBlock.hidden = true;
      document.getElementById("output-field").hidden = true;
    }
    return inspectTab(true);
  })
  .catch(() => {
    lastHealth = null;
    healthText.textContent = "Motor apagado — descargá ILSA en Instalar motor";
    metaMode.textContent = "—";
    metaTranslatorRow.hidden = true;
    document.getElementById("output-field").hidden = true;
    return inspectTab(false);
  });
