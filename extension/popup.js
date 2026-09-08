/**
 * Popup de la extensión: health del motor y ON/OFF de subtítulos en Meet.
 */

const dot = document.getElementById("dot");
const healthText = document.getElementById("health-text");
const btnMeet = document.getElementById("btn-meet");
const pageHint = document.getElementById("page-hint");

let motorOk = false;
let meetTab = null;
let meetRunning = false;

document.getElementById("btn-setup").addEventListener("click", () => {
  chrome.tabs.create({ url: chrome.runtime.getURL("welcome.html") });
});

btnMeet.addEventListener("click", async () => {
  if (!meetTab) return;
  btnMeet.disabled = true;
  try {
    if (meetRunning) {
      await chrome.runtime.sendMessage({ type: "lsa-meet-stop-request" });
      meetRunning = false;
      btnMeet.textContent = "Subtítulos en Meet";
      pageHint.textContent = "Subtítulos detenidos.";
    } else {
      const res = await chrome.runtime.sendMessage({
        type: "lsa-meet-start-request",
        tabId: meetTab.id,
        leftHanded: false,
      });
      if (!res || !res.ok) throw new Error((res && res.error) || "No se pudo iniciar");
      meetRunning = true;
      btnMeet.textContent = "Detener subtítulos";
      pageHint.textContent =
        "Activá subtítulos, recargá Meet si hace falta, y apagá/prendé la cámara. Con LSA off, Meet usa tu cámara normal.";
    }
  } catch (err) {
    pageHint.textContent = err.message || String(err);
  } finally {
    btnMeet.disabled = false;
  }
});

/**
 * Detecta si la pestaña activa es Meet para mostrar el botón de subtítulos.
 * @returns {Promise<void>}
 */
async function inspectTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  meetTab = tab && tab.url && tab.url.startsWith("https://meet.google.com/") ? tab : null;
  if (meetTab) {
    btnMeet.hidden = false;
    pageHint.textContent =
      "Esta pestaña es Meet. El subtítulo se pinta en tu cámara; los demás lo ven en tu video.";
  } else {
    btnMeet.hidden = true;
    pageHint.textContent = "Para subtítulos, abrí este popup desde una pestaña de Meet.";
  }
}

LsaApi.health()
  .then((h) => {
    if (!h.ok) throw new Error("no listo");
    motorOk = true;
    dot.classList.add("on");
    healthText.textContent = "Motor conectado";
    if (meetTab) btnMeet.disabled = false;
  })
  .catch(() => {
    healthText.textContent = "Motor apagado — descargá IRIS en Instalar motor";
    btnMeet.disabled = true;
  });

inspectTab().then(() => {
  btnMeet.disabled = !motorOk;
});
