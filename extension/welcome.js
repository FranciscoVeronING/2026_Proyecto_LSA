const dot = document.getElementById("dot");
const healthText = document.getElementById("health-text");
const btnOpen = document.getElementById("btn-open");
const btnCheck = document.getElementById("btn-check");

async function refreshHealth() {
  try {
    const h = await LsaApi.health();
    if (!h.ok) throw new Error("pipeline no listo");
    dot.classList.add("on");
    const llm = h.semantic_ready ? "LLM lista" : "LLM cargando o desactivada";
    healthText.textContent = `Motor en línea (${h.device || "cpu"}). ${llm}.`;
    btnOpen.disabled = false;
  } catch {
    dot.classList.remove("on");
    healthText.textContent =
      "No hay motor en http://127.0.0.1:8765. Ejecutá el exe o python run_backend.py.";
    btnOpen.disabled = true;
  }
}

btnCheck.addEventListener("click", refreshHealth);
btnOpen.addEventListener("click", () => {
  chrome.tabs.create({ url: chrome.runtime.getURL("translator.html") });
});

refreshHealth();
setInterval(refreshHealth, 3000);
