# Esqueleto oyente en Meet (T-pose)

El personaje no es un muñeco 3D. Es el mismo esqueleto que dibuja
`PoseVisualizer` a partir de archivos `.pose`. En vivo, ILSA **no** lee
`extraction/out/poses/`: pide cada clip al servidor de T-pose.

Cómo se **fabricaron** esos clips (PDF, QR, YouTube, Holistic):
[DICCIONARIO.md](DICCIONARIO.md). Meet y canvas: [EXTENSION.md](EXTENSION.md).

---

## 1. Una llamada

ILSA en modo **oyente**, extensión de esta carpeta, cámara de Meet prendida.

Meet pide la cámara. La extensión devuelve un canvas a 24 fps
(`canvas.captureStream`). Mitades a la misma altura:

| Izquierda | Derecha |
|-----------|---------|
| Cámara real, recorte `cover` | Esqueleto en **idle** (espera) |

El idle no seña “por las dudas”.

### Ejemplo: “hola mamá”

1. Mic de Meet → `POST /stt` → `hola mamá`. Si vino del chat, no hay STT.
2. `POST /hearing` (ILSA) → Cloud Run `POST /hearing` → p. ej. `HOLA MAMÁ`.
3. Esas glosas se pintan abajo del video (quien recibe las lee).
4. Por cada glosa, `GET /tposes/{STEM}.pose` al servidor.
5. El compositor local empalma clips. El canvas reproduce a la derecha y
   vuelve al idle.

Si una palabra no tiene clip, **se deletrea**. *Francisco* sin `FRANCISCO` se
seña letra por letra. `CH` y `Ñ` cuentan una. Si el servidor no responde, el
esqueleto se queda en idle y el HUD avisa.

Cada dedo tiene color (pulgar rojo … meñique azul). Palma blanca. El panel del
esqueleto se **espeja** al dibujar: la mano derecha anatómica queda a la
derecha de esa mitad. Cámara y subtítulos no se espejan.

---

## 2. Servidor y stems

URL default (`src/avatar/config.py`):

`https://tpose-server-705664945052.southamerica-east1.run.app`

Override: `LSA_AVATAR_URL` o `%LOCALAPPDATA%\ILSA\avatar_url.txt`.
Caché de archivos: `%LOCALAPPDATA%\ILSA\tposes\{STEM}.pose`.
Código: `src/avatar/remote.py`, `src/avatar/names.py`.

El servidor compara **igualdad de nombre**. ILSA tiene que pedir el stem
exacto.

`tpose_stem`:

1. `strip` + mayúsculas.
2. Si termina en `_RA`, se corta; si no, si termina en `_A`, se corta
   (forma masculina de glosas tipo `ABOGADO_A`).
3. Tildes ÁÉÍÓÚÜ → vocales.
4. Se tira todo lo que no sea `[A-Z0-9Ñ]`.
5. Si el compacto es solo `Ñ` → `ENIE`.
6. Toda `Ñ` interior → `N` (`ESPAÑA` → `ESPANA`).
7. Si el compacto es `PAPA` → `PADRE` (en el servidor el clip de padre es
   `PADRE.pose`; `PAPA.pose` puede existir y **no** se pide).

Ejemplos: `HOLA` → `HOLA`; `MAMÁ` → `MAMA`; `AÑO` → `ANO`; `CAFÉ` → `CAFE`;
`Ñ` → `ENIE`; `ABOGADO_A` → `ABOGADO`.

`resolve_chain` prueba de a 3, 2 y 1 token (`BUENOS AIRES` puede ser un
archivo). Un dígito suelto se busca como `CERO`…`NUEVE`. Si no hay clip, se
deletrea con `expand_spell` (CH, Ñ). Tope 24 señas.

Health del servidor: `GET /health`. Un stem inexistente: HTTP 404 → se sigue
con la siguiente letra o `missing`.

---

## 3. Recorrido técnico

```
Meet getUserMedia
    → inject-gum.js  canvas 50/50, LsaAvatar idle

Mic / chat
    → POST /stt (si hay WAV)
    → POST /hearing
         → Cloud Run POST /hearing
         → GET /tposes/{STEM}.pose
         → sequence_from_files
    → LSA_CAPTION + LSA_AVATAR
    → reproduce y vuelve a idle
```

Solo con `_mode == hearing`. El clip viaja en `POST /hearing` (`avatar`).

`HearingSession.ingest_speech` llama `clip_for_glosses`. `meet.js`
`applyAvatarClip` usa ese JSON.

---

## 4. Compositor

`extraction/meet_sequence.py` no resuelve nombres. Recibe `(glosa, Path)` ya
bajados. Carga `render_pose_pair.py` en caliente.

Cada clip, `prepare_sign`: fps 30, recorte, mano idle, huecos cortos, park o
snap, EMA, canvas **720×780**, hombros a 250 px. Clips preparados en caché de
memoria. Entre clips, `compose_sequence` fabrica un puente (4 a 8 frames,
smoothstep, según distancia de muñecas).

Tope: 1200 frames.

La respuesta **no** manda los 543 puntos de Holistic. Manda lo dibujable:

| Campo | Qué es |
|-------|--------|
| `p` | 8 puntos: hombros, codos, muñecas, caderas |
| `l` / `r` | 21 puntos de cada mano; `r` es la derecha anatómica |
| `f` | Contorno de cara (óvalo, cejas, ojos, nariz, labios), sin iris |
| `face_edges` | Pares a unir, una vez por respuesta |
| `fps`, `w`, `h`, `shoulder`, `shoulder_y` | Reproducción e idle |
| `played` / `missing` | Qué se anima; lo que ni deletreando está |
| `ok: false` | Sin glosas, T-pose caído, o sin clip |

Confianza menor a 0,15 viaja como `-1` y no se dibuja. Coordenadas: píxeles
720×780, +X derecha, +Y abajo. El visor escala en `contain` en la mitad
derecha.

`clip_for_glosses` mete la raíz del repo en `sys.path` para importar
`extraction.meet_sequence` (el motor solo agrega `src/` al path).

---

## 5. Reproducción en el navegador

`window.LsaAvatar` (`content/skeleton.js`). `sample` corre por reloj, no por
el fps de `captureStream`.

1. **Entrada** (~6/30 s). Idle → primer frame (smoothstep).
2. **Seña.** `floor((t - entrada) * fps)`.
3. **Salida** (~6/30 s). Vuelve al idle.
4. Se suelta el clip.

El puente **entre** señas ya viene en la lista de frames.

El idle no es un `.pose` de seña completa. Cuerpo y manos salen de los
primeros cuadros de `HOLA.pose` (espera del video), escalados al ancho de
hombros. Antebrazo casi horizontal, un poco abajo; la muñeca no se quiebra.

| Parte | Color |
|-------|--------|
| Cuerpo y cara en espera | `#d5dde6` |
| Cara mientras seña | gris claro del clip |
| Palma | `#f4f4f4` |
| Dedos | rojo, naranja, amarillo, verde, azul |
| Fondo derecha | `#121216` |

---

## 6. Archivos

| Archivo | Rol |
|---------|-----|
| `src/avatar/config.py` | URL y token |
| `src/avatar/names.py` | `tpose_stem`, `expand_spell` |
| `src/avatar/remote.py` | GET, caché, `resolve_chain`, `clip_for_glosses` |
| `src/backend/hearing_session.py` | Tras Cloud Run `/hearing`, pide el clip |
| `src/backend/server.py` | `POST /hearing` (glosas + clip) |
| `extraction/meet_sequence.py` | Paths → JSON de frames |
| `extraction/render_pose_pair.py` | Compositor (también mp4 de laboratorio) |
| `extension/content/skeleton.js` | Dibujo |
| `extension/content/inject-gum.js` | 50/50 + `LSA_AVATAR` |
| `extension/content/meet.js` | `applyAvatarClip` |

---

## 7. Si no se mueve

| Lo que ves | Dónde está |
|------------|------------|
| Cámara entera, sin esqueleto | No es modo oyente, o no es este `extension/` |
| Quietos y “falta el clip” | 404 del stem, o servidor caído |
| Deletrea | Esa palabra no tiene `.pose`; `CH` y `Ñ` son una seña |
| “Esqueleto en reposo” | Error de red o `LSA_AVATAR_URL` |
| Naranja del lado “equivocado” | Recargar extensión y Meet (preview local) |
| Voz / Texto no cambia | Recargar la extensión una vez |
