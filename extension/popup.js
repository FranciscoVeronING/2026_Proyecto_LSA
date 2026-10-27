/**
 * Popup: engine status, Meet output, landmarks (debug), and report.
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
    inputHint.textContent = "Meet microphone only. Chat text is not translated.";
  } else if (mode === "text") {
    inputHint.textContent = "Meet chat only. The microphone is not transcribed.";
  } else {
    inputHint.textContent = "Meet microphone and chat are translated into signs.";
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
      "Reads the translation aloud on this PC. Others do not see text on your camera.";
  } else if (outputMode === "both") {
    outputHint.textContent =
      "Captions on your video and voice on this PC.";
  } else {
    outputHint.textContent = "Spanish is painted on your camera; others read it on your video.";
  }
}

/**
 * @param {boolean} engineOk
 * @returns {Promise<void>}
 */
async function inspectTab(engineOk) {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  inMeet = Boolean(tab && tab.url && tab.url.startsWith("https://meet.google.com/"));
  if (!engineOk) {
    pageHint.textContent =
      "To translate, Start ILSA. If you do not want it, leave it off: Meet works as usual.";
    return;
  }
  if (inMeet) {
    chrome.runtime.sendMessage({ type: "lsa-meet-sync", tabId: tab.id }).catch(() => {});
    pageHint.textContent =
      "You are in Meet: translation is already on. To stop it, Stop ILSA in the exe.";
  } else {
    pageHint.textContent =
      "Open Google Meet: with ILSA on, translation starts by itself. To stop, Stop the exe.";
  }
}

function openReport() {
  const user = (reportText.value || "").trim();
  if (!user) {
    reportHint.textContent = "Write what happened, even if it is one line.";
    return;
  }
  const manifest = chrome.runtime.getManifest();
  const h = lastHealth;
  const mode =
    h && h.mode === "hearing" ? "Hearing" : h && h.ok ? "Signer" : "Engine off";
  const body = [
    "## What happened",
    user,
    "",
    "## Context",
    `- Extension: ${manifest.version}`,
    `- Engine: ${h && h.ok ? "on" : "off"}`,
    `- Mode: ${mode}`,
    `- In Meet: ${inMeet ? "yes" : "no"}`,
  ].join("\n");
  const url =
    REPO_ISSUES +
    "?title=" +
    encodeURIComponent("ILSA / Meet failure") +
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
    if (!h.ok) throw new Error("not ready");
    lastHealth = h;
    dot.classList.add("on");
    const signer = h.mode !== "hearing";
    metaMode.textContent = signer ? "Signer (LSA → Spanish)" : "Hearing (voice/chat → glosses)";
    healthText.textContent = "Engine connected";
    if (signer) {
      debugBlock.hidden = false;
      outputField.hidden = false;
      pauseField.hidden = false;
      inputField.hidden = false;
      inputHint.textContent = "Voice or chat capture is used with ILSA in hearing mode.";
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
    healthText.textContent = "Engine off — download ILSA in Install engine";
    metaMode.textContent = "—";
    outputField.hidden = true;
    pauseField.hidden = true;
    inputField.hidden = false;
    return inspectTab(false);
  });
