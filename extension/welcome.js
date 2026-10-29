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
  exeHint.textContent = "Preparando la descarga…";
  try {
    if (await resourceExists(PACKAGED_ZIP)) {
      saveAs(PACKAGED_ZIP, "ILSA.zip");
      exeHint.textContent =
        "Listo. Descomprimí ILSA.zip, abrí ILSA.exe y dejá esa ventana abierta.";
      return;
    }
    if (await resourceExists(RELEASE_ZIP)) {
      saveAs(RELEASE_ZIP, "ILSA.zip");
      exeHint.textContent =
        "Descargando desde GitHub. Descomprimí, abrí ILSA.exe y dejá la ventana abierta.";
      return;
    }
    exeHint.textContent =
      "No se encontró ILSA.zip en el release de ilsa.";
  } catch (err) {
    exeHint.textContent = err && err.message ? err.message : "No se pudo descargar ILSA.";
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
    const mode = h.mode === "hearing" ? "oyente · voz → subtítulos" : "sordo · LSA";
    const extra =
      h.mode === "hearing"
        ? "Transcripción lista."
        : h.semantic_ready
          ? "Traductor listo."
          : "Traductor no disponible.";
    healthText.textContent = `ILSA encendido (${mode}). ${extra}`;
  } catch {
    dot.classList.remove("on");
    healthText.textContent = "ILSA no está abierto. Descargalo, ejecutalo y volvé a comprobar.";
  }
}

btnExe.addEventListener("click", downloadIlsa);
btnCheck.addEventListener("click", refreshHealth);

refreshHealth();
setInterval(refreshHealth, 3000);
