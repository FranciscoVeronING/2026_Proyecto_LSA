# Proyecto LSA 2026
## Ficha Técnica de Entrega

**Módulo:** Extensión (Chrome + ILSA)  
**Iteración:** Primera entrega de la extensión  
**Fecha estimada:** 13/09/2026  
**Fecha entregado:** 13/09/2026  

---

### Alcance y objetivos de la entrega

Esta ronda une el clasificador y el traductor con Google Meet. La persona sorda seña frente a la cámara de la videollamada. El sistema corta cada seña, la convierte en una glosa y, tras una pausa, arma una oración en español. Esa oración se ve como subtítulo en la cámara. En modo oyente ocurre lo inverso: la voz de esta PC se escribe como subtítulo para que la otra persona la lea.

La entrada es el video de Meet (cámara local). La salida es texto en español sobre el video, más un estado visible en un HUD. El motor no corre dentro de Chrome. Corre en una ventana aparte llamada ILSA. La extensión solo extrae el esqueleto de las manos y el cuerpo, y le manda números al motor.

Hasta esta fecha el flujo completo funciona en la máquina de trabajo. En la laptop objetivo el mismo flujo también corre, pero no de forma óptima. El cuello de botella más claro es el traductor local. Esa parte queda abierta: vamos a evaluar pasar el semántico a la nube.

---

### Especificaciones técnicas y decisiones de diseño

El sistema tiene dos piezas. La extensión de Chrome trabaja en Meet. ILSA es el motor local (ventana con FastAPI en `127.0.0.1:8765`). Chrome no puede cargar la red de señas ni el modelo de lenguaje. Por eso el navegador solo envía landmarks. El video crudo no entra a PyTorch.

MediaPipe Holistic corre en un iframe sandbox. Manifest V3 bloquea el tipo de worker que Holistic necesita. El sandbox relaja esa regla. Un documento offscreen recibe un JPEG chico, corre Holistic y decide cuándo empieza y termina cada seña. Luego hace `POST /sign` al motor.

En Meet no envolvemos la cámara todo el tiempo. Si lo hacemos, Meet cree que la cámara está ocupada o muerta. El gancho de `getUserMedia` actúa solo cuando LSA está habilitado. Meet ve un canvas. El video real queda en un elemento oculto. El español se dibuja abajo. Meet espeja la vista local, así que el texto se dibuja espejado para que se lea bien.

ILSA tiene dos modos, elegidos antes de Encender. Sordo: señas a español. Oyente: voz a subtítulos. Con el motor encendido, Meet arranca solo. El popup muestra estado. No enciende el motor.

El clasificador es el mismo TinySkeleton de las entregas anteriores. Recibe 16 frames de 225 números (pose y dos manos). Corre en CPU dentro de ILSA. El traductor actual es Llama 3.2 1B en formato GGUF, también en CPU. El archivo del modelo no va dentro del exe. La primera vez que se abre ILSA, lo baja desde GitHub (~770 MB) a una carpeta del usuario.

La captura de señas no usa movimiento de todo el cuadro. En Meet hay gente detrás y eso disparaba falsos inicios. Solo se miran las manos. Como Holistic en Meet rinde pocos cuadros por segundo, los umbrales en “cantidad de frames” se sentían muy lentos. Ahora el inicio y el fin también miran el tiempo: una mano usable abre la seña en un instante; unas manos quietas unos 0,9 s la cierran.

---

### Alternativas que se probaron

Primero se pensó seguir solo con la app de escritorio (OpenCV + webcam). Eso ya interpretaba LSA, pero no servía para una videollamada real. Se pasó a una extensión en Chrome más un motor local. Esa es la línea que se entrega ahora.

Después se discutió si el backend debía ver video o solo landmarks. Mandar video saturaba la PC y mezclaba el fondo de Meet. Se eligió extraer el esqueleto en el navegador y mandar JSON. El clasificador nunca ve píxeles.

Para el traductor se probaron varios modelos chicos en local: Qwen 2.5 en 0.5B, 1.5B y 3B, y Llama 3.2 1B. Qwen 3B traduce mejor en las pruebas de laboratorio, pero pide más memoria y más tiempo. En una laptop justa no es un buen compañero de Meet y MediaPipe al mismo tiempo. Se dejó un solo traductor empaquetado: Llama 1B en CPU. Es más liviano. No es el techo de calidad.

También se armó el exe con GPU y con CPU. La versión con CUDA inflaba el zip y fallaba en PCs sin placa NVIDIA. La versión CPU es la que se publica. En el camino faltaron DLLs de Visual C++: en una PC sin Visual Studio el exe no cargaba `shm.dll`. Ahora esas librerías viajan junto al exe.

El modelo dentro del zip se descartó. El archivo pesa cientos de megas y GitHub limita el tamaño. Se separó: el zip trae ILSA; el GGUF se descarga al primer arranque.

En la ventana de ILSA se probaron menús emergentes para elegir modo. En algunas PCs el menú quedaba atrás o se cerraba mal. Se reemplazó por dos botones claros: Sordo y Oyente.

---

### Problemas encontrados

Meet y la extensión pelean por la cámara. Si el gancho de `getUserMedia` queda siempre activo, Meet no arranca el video. La regla pasó a ser: envolver solo con LSA prendido, y al apagar no cortar los tracks reales.

Recargar la extensión con Meet abierto deja el script viejo inválido. Hay que recargar la pestaña de Meet. Es un límite de Chrome, no del modelo.

Holistic a veces marca una mano que no está (un “fantasma” al mover el torso). Si se abre la seña con un solo tick ruidoso, el clasificador recibe basura. Se pide un instante de mano usable, no un único marco suelto.

El movimiento de píxeles del cuadro completo no sirve en una llamada. Una persona que camina atrás alcanzaba el umbral. La captura quedó atada a landmarks de mano.

En la laptop objetivo MediaPipe baja a pocos FPS. Un umbral de “28 frames quietos” o “6 frames para empezar” se volvía una espera de más de un segundo. La seña arrancaba tarde y se cortaba tarde. Se bajaron los umbrales y se midió también en milisegundos.

El puente JPEG entre Meet y el offscreen también pesaba. Se bajó el ancho de 320 a 240 px y la calidad del JPEG. Eso alivia CPU y memoria. No resuelve del todo el atraso del traductor.

Al empaquetar, excluir mal un módulo de PyTorch rompía el arranque. Faltaban rutas de DLL. El hook de runtime ahora agrega las carpetas de Torch al PATH antes de importar.

El traductor local, aunque sea de 1B, sigue siendo lento cuando la misma máquina corre Chrome, Meet, Holistic y el clasificador. La oración llega, pero la espera se nota. Eso no es un fallo de glosas: es carga de la PC.

---

### Resultados y validación

En la máquina de desarrollo el circuito está cerrado. Se abre ILSA, se elige modo Sordo, se entra a Meet y se seña. Aparecen glosas. Tras unos cuatro segundos de pausa aparece el español. El subtítulo se borra solo a los ocho segundos. El modo Oyente pinta la voz como texto en la misma cámara.

El clasificador y las reglas de repetición no se reentrenaron en esta ronda. Se reutiliza el TinySkeleton ya validado (94 clases, 16 frames). Lo nuevo es el canal: Meet, sandbox, ILSA y el exe.

El empaquetado genera `ILSA.exe` y `ILSA.zip` sin el GGUF. En una PC limpia el programa abre, muestra carga, baja el modelo una vez y queda listo. El zip cabe en el límite de GitHub. La extensión está en la versión 1.6.6.

No hay en esta ficha una tabla nueva de BLEU ni de Top-1 en cámara. Esas métricas viven en las entregas del clasificador y del semántico. Acá la prueba fue funcional: punta a punta, con Meet real, en dos clases de máquina.

En la laptop de trabajo el uso es fluido. En la laptop objetivo el sistema **funciona**, pero **no de forma óptima**. Holistic va justos de FPS. El traductor tarda. La captura ya se aceleró. Aun así la oración en español no llega con la frescura que hace falta para una llamada.

---

### Estado al cierre de esta fecha (punto de control)

Hasta acá el producto integrado **funciona**. Hay extensión, hay motor instalable, hay modo sordo y modo oyente, y hay subtítulos en Meet. Este commit deja registro de ese estado.

Lo que no está resuelto es el rendimiento en la laptop objetivo. Seguir agrandando el modelo local no es el camino. Seguir achicándolo también tiene techo: un modelo más chico traduce peor.

**Pendiente (abierto):** evaluar el módulo semántico **en la nube**. La idea es dejar en la PC solo lo que tiene que ser local (cámara, Holistic, clasificador) y mandar la lista de glosas a un servicio remoto que devuelva el español. Esta sección se completa cuando esa prueba exista. Hoy solo se deja el problema planteado y la dirección de trabajo.

No se decide aún proveedor, costo ni arquitectura de red. Primero hay que medir si la nube baja la espera sin romper la privacidad del video. El video no debería salir de la PC. Solo saldrían glosas (texto).

---

### Tecnologías usadas en esta entrega

Extensión Manifest V3, MediaPipe Holistic en WASM, FastAPI en loopback, PyTorch CPU (TinySkeleton), Llama 3.2 1B GGUF por llama.cpp, PyInstaller, Tkinter para la ventana ILSA, pyttsx3 solo en el escritorio (en Meet el sordo ve texto, no se lee en voz alta hacia la llamada).
