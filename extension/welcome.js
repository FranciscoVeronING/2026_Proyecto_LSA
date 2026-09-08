/**
 * Guía: descargar IRIS (el exe/zip empaquetado) y comprobar que el motor responde.
 *
 * El paquete NO sale de localhost: vive en `extension/bin/IRIS.zip` (lo genera
 * packaging/build_exe.bat) o en GitHub Releases. Un <a download> evita cargar
 * el archivo entero en memoria.
 */

const PACKAGED_ZIP = chrome.runtime.getURL("bin/IRIS.zip");
const RELEASE_ZIP =
  "https://github.com/FranciscoVeronING/2026_Proyecto_LSA/releases/latest/download/IRIS.zip";

const dot = document.getElementById("dot");
const healthText = document.getElementById("health-text");
const btnCheck = document.getElementById("btn-check");
const btnExe = document.getElementById("btn-exe");
const exeHint = document.getElementById("exe-hint");

/**
 * ¿Hay un IRIS.zip dentro de la extensión? HEAD, sin bajar el binario.
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
 * Dispara la descarga del navegador (no navega a una URL).
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
 * Baja IRIS.zip desde la extensión o, si no está, desde GitHub Releases.
 * @returns {Promise<void>}
 */
async function downloadIris() {
  btnExe.disabled = true;
  exeHint.textContent = "Preparando descarga…";
  try {
    if (await resourceExists(PACKAGED_ZIP)) {
      saveAs(PACKAGED_ZIP, "IRIS.zip");
      exeHint.textContent =
        "Listo. Descomprimí IRIS.zip, abrí IRIS.exe y dejá esa ventana abierta.";
      return;
    }
    if (await resourceExists(RELEASE_ZIP)) {
      saveAs(RELEASE_ZIP, "IRIS.zip");
      exeHint.textContent =
        "Descargando desde GitHub. Descomprimí, abrí IRIS.exe y dejá la ventana abierta.";
      return;
    }
    exeHint.textContent =
      "Todavía no hay un IRIS.zip en esta extensión. Hay que generarlo una vez con packaging\\build_exe.bat y recargar la extensión en chrome://extensions.";
  } catch (err) {
    exeHint.textContent = err && err.message ? err.message : "No se pudo descargar IRIS.";
  } finally {
    btnExe.disabled = false;
  }
}

/**
 * GET /health: IRIS ya está corriendo en este PC.
 * @returns {Promise<void>}
 */
async function refreshHealth() {
  try {
    const h = await LsaApi.health();
    if (!h.ok) throw new Error("pipeline no listo");
    dot.classList.add("on");
    const llm = h.semantic_ready ? "LLM lista" : "LLM cargando";
    healthText.textContent = `IRIS en marcha (${h.device || "cpu"}). ${llm}.`;
  } catch {
    dot.classList.remove("on");
    healthText.textContent = "IRIS no está abierto. Descargalo, ejecutalo y volvé a comprobar.";
  }
}

btnExe.addEventListener("click", downloadIris);
btnCheck.addEventListener("click", refreshHealth);

refreshHealth();
setInterval(refreshHealth, 3000);
