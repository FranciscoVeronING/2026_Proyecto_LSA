/**
 * Guide: download ILSA (the packaged exe/zip) and check that the engine responds.
 *
 * The package does NOT leave localhost: it lives in `extension/bin/ILSA.zip`
 * (built by packaging/build_exe.bat) or on GitHub Releases. An <a download>
 * avoids loading the whole file into memory.
 */

const PACKAGED_ZIP = chrome.runtime.getURL("bin/ILSA.zip");
const RELEASE_ZIP =
  "https://github.com/FranciscoVeronING/2026_Proyecto_LSA/releases/download/ilsa-llama-1b/ILSA.zip";

const dot = document.getElementById("dot");
const healthText = document.getElementById("health-text");
const btnCheck = document.getElementById("btn-check");
const btnExe = document.getElementById("btn-exe");
const exeHint = document.getElementById("exe-hint");

/**
 * Is there an ILSA.zip inside the extension? HEAD, without downloading the binary.
 * @param {string} url
 * @returns {Promise<boolean>}
 */
async function resourceExists(url) {
  try {
    const res = await fetch(url, { method: "HEAD" });
    return res.ok;
  } catch {
    return false;
  }
}

/**
 * Trigger a browser download (does not navigate to a URL).
 * @param {string} url
 * @param {string} filename
 */
function saveAs(url, filename) {
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.rel = "noopener";
  document.body.appendChild(a);
  a.click();
  a.remove();
}

/**
 * Download ILSA.zip from the extension or, if missing, from GitHub Releases.
 * @returns {Promise<void>}
 */
async function downloadIlsa() {
  btnExe.disabled = true;
  exeHint.textContent = "Preparing download…";
  try {
    if (await resourceExists(PACKAGED_ZIP)) {
      saveAs(PACKAGED_ZIP, "ILSA.zip");
      exeHint.textContent =
        "Done. Unzip ILSA.zip, open ILSA.exe, and leave that window open.";
      return;
    }
    if (await resourceExists(RELEASE_ZIP)) {
      saveAs(RELEASE_ZIP, "ILSA.zip");
      exeHint.textContent =
        "Downloading from GitHub. Unzip, open ILSA.exe, and leave the window open.";
      return;
    }
    exeHint.textContent =
      "ILSA.zip was not found in the ilsa release.";
  } catch (err) {
    exeHint.textContent = err && err.message ? err.message : "Could not download ILSA.";
  } finally {
    btnExe.disabled = false;
  }
}

/**
 * GET /health: ILSA is already running on this PC.
 * @returns {Promise<void>}
 */
async function refreshHealth() {
  try {
    const h = await LsaApi.health();
    if (!h.ok) throw new Error("pipeline not ready");
    dot.classList.add("on");
    const mode = h.mode === "hearing" ? "hearing · voice → captions" : "signer · LSA";
    const extra =
      h.mode === "hearing"
        ? "Transcription ready."
        : h.semantic_ready
          ? "Translator ready."
          : "Translator unavailable.";
    healthText.textContent = `ILSA running (${mode}). ${extra}`;
  } catch {
    dot.classList.remove("on");
    healthText.textContent = "ILSA is not open. Download it, run it, and check again.";
  }
}

btnExe.addEventListener("click", downloadIlsa);
btnCheck.addEventListener("click", refreshHealth);

refreshHealth();
setInterval(refreshHealth, 3000);
