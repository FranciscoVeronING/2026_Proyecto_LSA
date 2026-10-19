# El avatar en Meet

Cómo se ve el esqueleto en una llamada, y qué código lo hace. El personaje no es un muñeco 3D: es el mismo esqueleto que dibuja `PoseVisualizer` a partir de los `.pose`.

Hay dos lecturas. La primera sigue una llamada de principio a fin. La segunda nombra archivos, mensajes y el JSON que viaja.

---

## 1. Una llamada, contada despacio

ILSA está en modo **oyente** y encendido. La extensión está cargada desde esta carpeta (`extension/`). Alguien entra a Google Meet con la cámara prendida.

Meet pide la cámara. La extensión no le devuelve la cámara cruda: le devuelve un video dibujado en un canvas, a 24 cuadros por segundo (`canvas.captureStream`). Quien está del otro lado ve ese dibujo.

El cuadro se parte al medio, las dos mitades con la misma altura:

| Izquierda | Derecha |
|---|---|
| La cámara real, recortada para llenar la mitad sin estirarse | El esqueleto en espera: torso de un cuadro real, manos adelante |

Ese quieto es el **idle**. El esqueleto no seña “por las dudas”. Espera.

### Ejemplo: “hola mamá”

La persona oyente dice *hola mamá* con el micrófono de Meet (o lo escribe en el chat y aprieta Enter).

1. La extensión manda el audio a ILSA (`POST /stt`). ILSA devuelve el texto `hola mamá`. Si vino del chat, este paso no existe: el texto ya está.
2. ILSA manda ese español al traductor (`POST /hearing`). Vuelve una lista de glosas, por ejemplo `HOLA MAMÁ`.
3. Esa lista se pinta abajo del video, para que quien recibe la llamada la lea.
4. Con esas glosas ILSA pide el clip al **servidor de T-pose** (Cloud Run, rama
   `avatar-cloud`). Los `.pose` no están en esta PC.
5. El canvas dibuja los fotogramas en la mitad derecha y, al terminar, el
   esqueleto vuelve a los brazos colgados.

Si una palabra no tiene clip en el servidor, no se salta: se deletrea.
*Francisco* sin `FRANCISCO` se seña `F`, `R`, `A`, `N`, `C`, `I`, `S`, `C`, `O`.
`CH` y `Ñ` son una sola letra. Si el servidor no responde, el esqueleto se
queda en idle y el HUD avisa.

### Qué hace cada mano

Cada dedo tiene su color, igual en las dos manos: pulgar rojo, índice naranja, medio amarillo, anular verde, meñique azul. La palma es blanca. Así se lee la forma aunque las manos se crucen.

En un plano de frente, la mano derecha de la persona queda a la **izquierda de la imagen**. El canvas no da vuelta el esqueleto.

Meet espeja tu miniatura con CSS. El canvas no: la cámara queda a la izquierda y el esqueleto a la derecha, igual para quien recibe. En tu video local la extensión deshace ese espejo, así que también lo ves derecho.

---

## 2. Voz, texto o las dos

**Qué se capta** tiene tres posiciones. Está en el popup de la extensión y, con ILSA en modo oyente, también en el cartel LSA de la llamada (Voz, Texto, Ambas). Se guarda en `chrome.storage.local` (`hearingInput`) y se puede cambiar con la llamada abierta.

| Elección | Qué entra | Qué no entra |
|---|---|---|
| Voz | El micrófono que Meet ya está usando, solo si no está silenciado | El chat. Con el micrófono en mute no se manda audio |
| Texto | Lo que escribís en el chat de Meet y enviás (Enter o el botón Enviar) | El micrófono. Se corta la escucha |
| Ambas | Micrófono y chat | — |

Ambas es el valor inicial. En el popup el control se ve también con el motor apagado o en modo sordo (ahí no se usa: la pista lo dice). En el cartel de Meet los tres botones aparecen solo en modo oyente.

El cartel se arrastra desde cualquier parte que no sea un botón. Al soltarlo, el lugar queda en `hudLeft` y `hudTop`. La próxima llamada abre en ese sitio. Si no lo moviste, sigue arriba a la derecha.

Por voz, la frase se cierra cuando pasa **1 segundo** sin audio después de haber detectado voz (energía del micrófono). Si seguís hablando sin pausa, se corta igual a los 4,5 segundos para no mandar un audio eterno.

El modo **sordo** no usa este control. Ahí la cámara clasifica señas y la salida es subtítulo, audio o los dos. El esqueleto de este documento es solo del oyente.

---

## 3. Recorrido técnico

```
Meet getUserMedia
    → inject-gum.js arma el canvas y captureStream(24)
    → mitad izquierda: video real (object-fit: cover)
    → mitad derecha: LsaAvatar (idle hasta que haya frames)

Mic de Meet                         Chat de Meet
    → WAV 16 kHz                        → texto al apretar Enter
    → POST /stt                         → (sin STT)
         └──────── español ────────────┘
                      → POST /hearing
                      → glosas
                      → GET T-pose /tposes/{GLOSA}.pose
                      → LSA_CAPTION + LSA_AVATAR
                      → el canvas reproduce y vuelve a idle
```

Todo eso ocurre si ILSA está en `hearing`. `POST /avatar` en modo sordo responde 400.

### Quién habla con quién

La página de Meet tiene dos mundos. El mundo **MAIN** puede envolver `getUserMedia`. El mundo **aislado** puede usar `chrome.runtime` y `fetch` a ILSA. No se ven las variables del otro: se pasan mensajes con `window.postMessage`.

| Mensaje (`source: "lsa-ext"`) | Quién lo manda | Qué hace el canvas |
|---|---|---|
| `LSA_SET_ENABLED` | `meet.js` al encender | Prende el modo oyente, parte el cuadro, y escucha el mic si la captura no es solo texto |
| `LSA_PREFS` | `meet.js` al leer o cambiar preferencias | `hearingInput`: `voice`, `text` o `both`. Si pasa a solo texto, corta el tap de audio |
| `LSA_CAPTION` | `meet.js` cuando hay glosas | Dibuja la línea abajo. En oyente el texto no se espeja: lo lee quien recibe |
| `LSA_AVATAR` | `meet.js` con el JSON de `/avatar` | Empieza la reproducción. Alarga el subtítulo para que dure lo que dura la seña |
| `LSA_SPEECH_STOP` | `meet.js` al apagar | Corta el tap de audio |

Del mundo MAIN hacia la extensión salen `LSA_AUDIO_WAV` (el wav en base64), `LSA_CAMERA` y errores de voz.

### Archivos

| Archivo | Rol |
|---|---|
| `extension/content/inject-gum.js` | Hook de `getUserMedia`, canvas, 50/50, subtítulo, tap del mic |
| `extension/content/skeleton.js` | `window.LsaAvatar`: idle, lerp de entrada y salida, dibujo |
| `extension/content/meet.js` | HUD, chat, `POST /stt`, `POST /hearing`, `POST /avatar` |
| `extension/content/meet.css` | Cartel: colores, botones de captura y cursor de arrastre |
| `extension/popup.html` / `popup.js` | “Qué se capta”. Visible también si el motor está apagado |
| `extension/lib/prefs.js` | `hearingInput`, y `hudLeft` / `hudTop` del cartel |
| `extension/lib/api.js` | Cliente HTTP a `127.0.0.1:8765` |
| `src/backend/server.py` | `POST /hearing` pide T-pose; `POST /avatar` por si hace falta |
| `src/avatar/remote.py` | `GET /tposes/{GLOSA}.pose` y caché en `%LOCALAPPDATA%\\ILSA\\tposes` |
| `src/avatar/names.py` | Stem de igualdad: mayúsculas, masculino, sin tildes |
| `extraction/meet_sequence.py` | Compositor: `.pose` bajados → frames del canvas |

`skeleton.js` se inyecta en el mundo MAIN **antes** que `inject-gum.js` (`manifest.json` y `background.js`).

### Qué hace `POST /avatar`

ILSA no guarda el diccionario. Por cada glosa pide
`GET https://tpose-server-705664945052.southamerica-east1.run.app/tposes/{GLOSA}.pose`
(el stem sale de `avatar.names.tpose_stem`: mayúsculas, forma masculina,
sin tildes ni signos; `Ñ` → `ENIE`, `PAPA` → `PADRE`). El compositor local
arma los frames del canvas.

Cada clip pasa por `prepare_sign`, en este orden: fps 30, recorte, detección de mano idle, relleno de huecos cortos, park o snap, EMA, canvas 720×780 con los hombros a 250 px. Los clips ya preparados quedan en un caché en memoria. Entre clip y clip, `compose_sequence` fabrica el puente (4 a 8 frames, smoothstep, según la distancia de las muñecas).

Tope: 24 señas y 1200 frames.

La respuesta no manda los 543 puntos. Manda lo que el canvas dibuja:

| Campo | Qué es |
|---|---|
| `p` | 8 puntos de pose: hombros, codos, muñecas, caderas. Sin piernas, sin cara de pose, sin puntas de los dedos del cuerpo |
| `l` / `r` | 21 puntos de cada mano. `r` es la derecha anatómica, a la izquierda de la imagen |
| `f` | Contorno de cara: óvalo, cejas, ojos, nariz, labios. Sin la malla de 468 y sin iris |
| `face_edges` | Qué pares unir, una sola vez por respuesta |
| `fps`, `w`, `h`, `shoulder`, `shoulder_y` | Para reproducir y para que el idle use el mismo cuerpo |
| `played` / `missing` | Glosas que se animan. Si la palabra no tiene archivo, `played` trae sus letras. `missing` es lo que ni deletreando se encontró |
| `ok: false` | No hay glosas, el servidor de T-pose no respondió, o no hay clip |

Un punto con confianza menor a 0,15 viaja como `-1` y no se dibuja.

Coordenadas: píxeles del canvas 720×780, +X a la derecha, +Y hacia abajo. El visor las escala para que entren en la mitad derecha sin recortar manos (`contain`).

### Cómo se reproduce

`LsaAvatar.start` guarda los frames y la hora. `sample` corre en cada cuadro del canvas, por reloj, no por el fps del `captureStream`:

1. **Entrada** (~6/30 s). Interpola del idle al primer frame. La cara aparece con el avance.
2. **Seña.** Frame `floor((t - entrada) * fps)`.
3. **Salida** (otros ~6/30 s). Vuelve al idle. La cara de la seña se apaga y queda la cara de líneas.
4. Idle otra vez. El clip se suelta.

La curva de la entrada y la salida es smoothstep (`3t² − 2t³`), la misma idea que el puente del compositor. El puente **entre** señas ya viene horneado en la lista de frames; el navegador no lo recalcula.

El idle no lee un `.pose` de seña completa. El cuerpo y las manos salen de los primeros cuadros de `HOLA.pose` (la espera del video), escalados al ancho de hombros del canvas. El antebrazo queda casi horizontal, un poco hacia abajo, y la mano sigue esa línea: la muñeca no se quiebra. La cara de espera sigue siendo el óvalo con cejas, ojos, nariz y boca.

Colores, en el canvas (no se espejan):

| Parte | Color |
|---|---|
| Cuerpo y cara en espera | `#d5dde6` |
| Cara mientras seña | gris claro, el contorno del clip |
| Palma | `#f4f4f4` |
| Pulgar, índice, medio, anular, meñique | rojo, naranja, amarillo, verde, azul |
| Fondo de la mitad derecha | `#121216` |

Fondo del compositor en el mp4 de prueba: `(18, 18, 22)` en OpenCV. El de Meet es el mismo tono.

### El diccionario

El diccionario de movimiento vive en el servidor de T-pose, no en
`extraction/out/poses/`. Si el HUD dice que falta el clip, el remoto no
reconoció esas glosas o no responde.

El pipeline que fabrica los `.pose` (PDF, QR, YouTube, Holistic, recorte de mosaico, ventanas) es el de `docs/Rama_avatar.md`. Este documento no lo repite. Acá solo se **reproducen**.

---

## 4. Qué mirar si no se mueve

| Lo que ves | Dónde está |
|---|---|
| Cámara entera, sin esqueleto | ILSA no está en modo oyente, o la extensión que cargaste no es la de este worktree |
| Esqueleto quieto y el HUD dice que falta el clip | El servidor de T-pose no respondió, o esas glosas no tienen clip |
| Glosas en pantalla y el esqueleto deletrea | Esa palabra no tiene `.pose`. Se parte en letras (`CH` y `Ñ` son una seña). Si una letra tampoco está, queda en «Sin clip» |
| “Esqueleto en reposo” | El servidor de T-pose no respondió o `LSA_AVATAR_URL` está mal |
| El naranja está “del lado equivocado” | Recargá la extensión y Meet. El preview local se endereza; quien recibe ya veía el esqueleto sin espejo |
| Cambiaste Voz / Texto y no pasa nada | Recargá la extensión una vez. Después el cambio entra en la llamada abierta |
