# Extensión Chrome y Google Meet

La extensión **LSA Intérprete** (`extension/`, Manifest V3, versión en
`manifest.json`) no clasifica ni traduce. Extrae esqueleto, recorta señas,
toca el mic de Meet, pinta el canvas y habla con ILSA en
`http://127.0.0.1:8765`.

Producto y API: [ARQUITECTURA.md](ARQUITECTURA.md). Esqueleto oyente:
[AVATAR.md](AVATAR.md).

---

## 1. Por qué hay tantos archivos

Manifest V3 bloquea `blob:` / `unsafe-eval` en páginas de extensión.
MediaPipe Holistic **necesita** crear workers desde blobs.

Solución:

- `sandbox.html` + `sandbox.js` corren Holistic con CSP relajado.
- `offscreen.html` manda un JPEG al iframe y recibe
  `{ pose, left_hand, right_hand }`.
- Los `.wasm` / `.tflite` viven en `extension/vendor/mediapipe/` (gitignored;
  `packaging/fetch_extension_assets.py`).

No se llama `holistic.close()` en cada stop de cámara: recargar el WASM es
lento. Se reutiliza el grafo.

Meet pide la cámara con `navigator.mediaDevices.getUserMedia`. Si se envuelve
ese método **siempre**, Meet ve la cámara bloqueada o un pipeline muerto. El
hook vive en el mundo **MAIN** y solo actúa con LSA habilitado.

---

## 2. Dos mundos en la pestaña de Meet

| Mundo | Scripts | Puede | No puede |
|-------|---------|-------|----------|
| **MAIN** (`document_start`) | `skeleton.js`, `inject-gum.js` | Envolver `getUserMedia`, dibujar canvas, tap del mic de Meet | `chrome.runtime`, `fetch` con privilegios de extensión |
| **Aislado** | `api.js`, `prefs.js`, `meet.js` + `meet.css` | `chrome.runtime`, `LsaApi` a ILSA, HUD | Tocar el prototipo de `getUserMedia` |

Se hablan con `window.postMessage` (`source: "lsa-ext"`).

`skeleton.js` se inyecta **antes** que `inject-gum.js`.

Si se recarga la extensión con Meet abierto, el content script aislado muere
(`Extension context invalidated`). Hay que recargar la pestaña de Meet.

---

## 3. Flujo de un frame (modo sordo)

```
#lsa-real-cam  (video oculto, stream real)
  → JPEG (~320 px, calidad ~0.62)
  → meet.js
  → background.js
  → offscreen.html  (Holistic + capture.js + LsaApi)
  → POST /sign  o  /utterance/end
  → mensaje de caption
  → HUD + postMessage LSA_CAPTION → canvas
```

MediaPipe no corre en `meet.js`. Corre offscreen para no pelear con la UI de
Meet y para sobrevivir si la pestaña pierde foco.

---

## 4. Canvas que ve Meet

`inject-gum.js`:

- Solo envuelve `getUserMedia` cuando LSA está **habilitado**.
- Crea `<video id="lsa-real-cam">` oculto con el stream real.
- Un canvas `captureStream(24)` es lo que Meet muestra como cámara.
- **Sordo:** cuadro entero de cámara; español abajo. Meet espeja el preview
  local con CSS; el texto sordo se dibuja espejado para leerse bien en esa
  miniatura. Quien recibe ve el canvas sin ese CSS.
- **Oyente:** mitad izquierda cámara (`object-fit: cover`), mitad derecha
  esqueleto espejado. El preview local **deshace** el CSS de Meet para que
  cámara y glosas se lean igual que para quien recibe.
- Apagar LSA no hace `stop()` de los tracks reales.

Subtítulos: `SUBTITLE_HOLD_MS = 8000` en `inject-gum.js` y `meet.js`. Un texto
nuevo reinicia el reloj. En oyente, si hay clip, el hold se alarga a la
duración de la seña. El offscreen **no** reenvía el último español en cada
tick.

---

## 5. Mute, cámara apagada, STT

El tap de audio del oyente **no** abre un `getUserMedia` nuevo. Copia el
track de mic que Meet ya tiene, lo pasa por un `AudioWorklet` (PCM 16 kHz →
WAV → `LSA_AUDIO_WAV`).

La frase de voz se cierra con **1 s** de silencio después de detectar energía,
o a los **4,5 s** si no hay pausa.

| Evento | Efecto |
|--------|--------|
| Mic mute en la UI de Meet | Corta el tap; no hay `/stt` |
| Captura “Texto” | Corta el tap; entra el chat |
| Cámara apagada (UI Meet / track) | No se mandan JPEG; no se clasifica; el canvas no finge video |

La detección de mute/cámara mira botones de Meet (`data-is-muted`, labels) y
el estado del track. Si Chrome o Meet cambian el DOM, hay que ajustar
`inject-gum.js` y recargar la pestaña.

---

## 6. Voz, texto o ambas (oyente)

Preferencia `hearingInput` en `chrome.storage.local` (`lib/prefs.js`):
`voice` | `text` | `both` (default). Se cambia en el popup y en el cartel LSA
de la llamada, con la reunión abierta.

| Elección | Entra | No entra |
|----------|-------|----------|
| Voz | Mic de Meet, si no está mute | Chat |
| Texto | Chat al enviar (Enter o Enviar) | Mic |
| Ambas | Los dos | — |

El chat usa el `contenteditable` interno de Meet (no un `<textarea>`). El
modo sordo no usa este control: ahí la salida es subtítulo / audio / ambos
(`outputMode`).

El cartel LSA se arrastra; la posición queda en `hudLeft` / `hudTop`.

---

## 7. Mensajes MAIN ↔ aislado

| Tipo (`source: "lsa-ext"`) | Dirección | Qué hace |
|----------------------------|-----------|----------|
| `LSA_SET_ENABLED` | meet.js → MAIN | Prende/apaga canvas, modo hearing, tap de mic |
| `LSA_PREFS` | meet.js → MAIN | `hearingInput`; si pasa a solo texto, corta el tap |
| `LSA_CAPTION` | meet.js → MAIN | Línea de subtítulo |
| `LSA_AVATAR` | meet.js → MAIN | Empieza el esqueleto (`LsaAvatar`) |
| `LSA_SPEECH_STOP` | meet.js → MAIN | Corta el tap |
| `LSA_AUDIO_WAV` | MAIN → meet.js | WAV para `POST /stt` |
| `LSA_CAMERA` | MAIN → meet.js | Estado de cámara / mute |

---

## 8. Archivos

| Archivo | Rol |
|---------|-----|
| `manifest.json` | MV3, permisos Meet, sandbox, CSP, content scripts |
| `background.js` | Offscreen, reenvío de frames y captions, inyección |
| `popup.html` / `popup.js` | Estado del motor, salida Meet, “Qué se capta” |
| `welcome.html` / `welcome.js` | Guía: descargar ILSA y `/health` |
| `sandbox.html` / `sandbox.js` | Holistic WASM |
| `offscreen.html` / `offscreen.js` | JPEG → Holistic → `capture.js` → API |
| `lib/api.js` | Cliente HTTP `127.0.0.1:8765` |
| `lib/prefs.js` | Preferencias y posición del HUD |
| `lib/landmarks.js` | Empaquetado, presencia de manos, debug |
| `lib/capture.js` | Máquina de estados de la seña |
| `content/inject-gum.js` | Hook de cámara, canvas, mic, mute |
| `content/skeleton.js` | `window.LsaAvatar` |
| `content/meet.js` | HUD, JPEG, chat, `/stt` `/hearing` |
| `content/meet.css` | Cartel LSA |

`host_permissions` incluyen GitHub: el botón **Descargar ILSA** baja el zip
publicado. `GET /mediapipe/{name}` en ILSA sirve los WASM si el `file://` de
la extensión no alcanza.

---

## 9. Qué mirar si Meet no traduce

| Lo que ves | Dónde está |
|------------|------------|
| Cámara cruda, sin HUD | Extensión no cargada, o no es este `extension/` |
| HUD “motor apagado” | ILSA no Encendido; popup → recargar |
| HUD ON y no hay glosas | WASM faltantes; consola offscreen; log `/sign` |
| Español pegado para siempre | Recargar extensión y Meet (`SUBTITLE_HOLD_MS`) |
| Context invalidated | Recargar **Meet** después de recargar la extensión |
| STT con mute | Recargar Meet; revisar botón de mic |
| Esqueleto | [AVATAR.md](AVATAR.md) § problemas |
