# Eval de cámara — 100 señas, 10 tomas (28/10/2026)

Dos sesiones del mismo checkpoint, la misma mano y el mismo encuadre que quedó registrado. Cambia la luz de la habitación. La etiqueta `buena` o `mala` la anota quien corre el script: el programa no mide la luz.

| | Luz buena | Luz mala |
|---|---|---|
| CSV | `src/eval_100senias_x10_buena_20261028_200319.csv` | `src/eval_100senias_x10_mala_20261028_233708.csv` |
| Horario | 20:05–21:06 | 23:37–00:37 |
| Checkpoint | `model_new_2026_10_20` | el mismo |
| Tomas | 100 señas × 10, sin saltos | 100 señas × 10, sin saltos |
| Mano / modo | derecha, automático | derecha, automático |
| Captura | `640x480->pad853x480` | la misma |
| Extractor | `@mediapipe/holistic@0.5.1675471629` | el mismo |

El criterio de cámara del plan (top-1 > 75 % y top-3 > 90 %) se cumple en las dos, con la lectura de abajo.

---

## Cómo se hizo la prueba

El comando es `python camera.py --eval`, desde `src`. Dos argumentos separan las corridas:

```text
python camera.py --eval --light buena
python camera.py --eval --light mala
```

`--takes` vale 10 si no se pasa otro número. `--light` no cambia la cámara ni el modelo: escribe esa palabra en el nombre del CSV y en la columna `light` de cada fila.

Antes de abrir la cámara hay dos cuadros. El primero pide la mano dominante. El segundo lista los checkpoints de `src/model` y de las carpetas `model_new_*`. El que queda marcado es `model_new_2026_10_20`, el de 100 clases. `src/model` sigue siendo el de 97 y no entra en estas dos sesiones.

La cámara se pide a 1280×720. La que se usó entregó 640×480. El script rellena con bandas negras hasta 16:9 (853×480), que es el aspecto de los videos de entrenamiento. Holistic es el de la web, JS/WASM, pose y dos manos, sin cara. Cada frame queda en 225 valores y la secuencia se resamplea a 16 frames, igual que en el entrenamiento.

El modo es automático. Empieza a grabar si hay movimiento o si las manos se quedan visibles unos frames. Corta cuando la seña se queda quieta, cuando las manos salen de cuadro, o cuando el buffer llega a 60 frames. Entre una inferencia y la siguiente hay 1 segundo de espera. Si una captura cae dentro de esa espera, no se guarda y el contador de la pantalla no avanza. Por eso, entre toma y toma, las manos bajan un momento.

En pantalla se lee, por ejemplo, `EVAL 12/100 toma 3/10 GRACIAS`. La seña sigue siendo la misma durante 10 tomas y después pasa a la siguiente. El orden es el alfabético de los nombres de clase del checkpoint. `n` salta la toma actual y la anota como `SKIP`. `r`, o el botón REHACER, borra la toma anterior del CSV y su `.npy`, y vuelve a pedirla. En estas dos sesiones no hubo saltos.

Cada fila del CSV guarda la hora, la seña pedida, el modelo, el top-3 con su confianza, `hit_top1`, `hit_top3`, la mano, el modo, el tamaño de cuadro, la versión de MediaPipe, el número de toma (`rep`, de 1 a 10) y la luz. Los landmarks de la toma quedan en una carpeta `_takes` al lado del CSV. Al cerrar, la consola imprime un resumen con el top-1 literal: ahí `0` predicho como `O` cuenta como error.

## Cómo se leen los aciertos

`0` y `O` son la misma seña. `V` y `viernes` también. Si el modelo dice una y se pidió la otra, la toma cuenta como acierto, en los dos sentidos. Esa regla se aplica al leer el CSV. La columna `hit_top1` del archivo sigue siendo la comparación literal, y el resumen de la consola también.

No se juntan otros pares parecidos. `años` predicho como `G` queda como error.

---

## Luz buena

Lectura literal: **978/1000 = 97,8 %** top-1. Con la regla de `0`/`O` y `V`/`viernes`: **996/1000 = 99,6 %**. Top-3: **1000/1000**.

La regla absorbe 18 tomas: 8 de `0`→`O` y 10 de `V`→`viernes`. `O` y `viernes` ya estaban en 10/10 por su propio nombre, así que en esta sesión el cruce fue en un solo sentido.

95 señas salieron 10/10 en la lectura literal. Con la regla son 97. Quedan 4 errores, y en los cuatro la seña pedida quedó segunda.

| Seña | Aciertos | Salió |
|---|---:|---|
| `A` | 8/10 | `6` en las tomas 7 y 8 (0,62 y 0,59) |
| `ahora_hoy` | 9/10 | `tuyo` en la toma 10 (0,49, con `ahora_hoy` en 0,46) |
| `repetir` | 9/10 | `esposo a` en la toma 10 (0,83) |

Confianza media: 0,979 en los aciertos de la regla y 0,632 en esos 4 errores.

`gracias`, `ayudar` y `policia`: 10/10.

Letras, ñ y dígitos (37 señas, 370 tomas): 368/370 = 99,5 % con la regla, y 350/370 = 94,6 % en literal. Las 20 tomas de diferencia son las de `0` y `V`.

## Luz mala

Lectura literal: **922/1000 = 92,2 %** top-1 y **986/1000 = 98,6 %** top-3 (14 tomas fuera del top-3). Con la regla: **946/1000 = 94,6 %** top-1. Una de esas 14 es `O`→`0`, que pasa a acierto, así que el top-3 de la regla queda en **987/1000 = 98,7 %**. Hay 13 tomas en las que ni la seña pedida ni su equivalente entran al top-3.

La regla absorbe 24 tomas: `0`→`O` (6), `O`→`0` (8) y `V`→`viernes` (10). Acá el par `0`/`O` se cruza en los dos sentidos. `V` otra vez sale siempre como `viernes`, y `viernes` sigue en 10/10.

77 señas quedan en 10/10. La confianza media es 0,967 en los aciertos y 0,771 en los 54 errores: en luz mala el modelo sigue bastante seguro cuando se equivoca.

`policia` sigue en 10/10. `gracias` y `ayudar` bajan a 9/10 (`gracias`→`4` en la toma 8, confianza 0,97, y ni siquiera entra al top-3; `ayudar`→`como` en la toma 9).

Las caídas grandes, de señas que en luz buena estaban en 10/10 o en 9/10:

| Seña | Buena | Mala | A dónde fue en la mala |
|---|---:|---:|---|
| `T` | 10/10 | 3/10 | `I` en las 7 |
| `como` | 10/10 | 3/10 | `numero` en 5, `poder` en 2 |
| `años` | 10/10 | 5/10 | `G` en las 5 |
| `hola` | 10/10 | 6/10 | `chau` en las 4 |
| `ahora_hoy` | 9/10 | 6/10 | `quien`, `lunes`, `tuyo`, `dia` |
| `N` | 10/10 | 7/10 | `ahora_hoy` en las 3 |
| `futuro` | 10/10 | 7/10 | `hola` en las 3 |
| `vos` | 10/10 | 7/10 | `K` en 2, `el_ella` en 1 |
| `P` | 10/10 | 8/10 | `lugar`, `M` |
| `cara` | 10/10 | 8/10 | `4`, `pasado` |
| `ellos` | 10/10 | 8/10 | `N`, `tuyo` |

Otras doce señas pierden una sola toma: `2`, `Y`, `ayudar`, `cuchillo`, `donde`, `gracias`, `mal`, `mio`, `nosotros`, `que`, `si`, `vivir`. Dos que habían fallado con luz buena salen enteras con luz mala: `A` (10/10) y `repetir` (10/10).

Las 13 tomas fuera del top-3, ya descontando `O`→`0`:

| Seña | Toma | Salió | Confianza | Top-2 | Top-3 |
|---|---:|---|---:|---|---|
| `2` | 10 | `3` | 1,00 | `6` | `H` |
| `P` | 1 | `lugar` | 0,77 | `ahora_hoy` | `viernes` |
| `Y` | 4 | `tener` | 0,99 | `si` | `domingo` |
| `ahora_hoy` | 2 | `quien` | 0,49 | `tuyo` | `K` |
| `ahora_hoy` | 5 | `lunes` | 0,33 | `K` | `tuyo` |
| `ahora_hoy` | 6 | `tuyo` | 0,63 | `robar` | `martes` |
| `ahora_hoy` | 8 | `dia` | 0,78 | `lunes` | `tuyo` |
| `como` | 6 | `numero` | 0,97 | `poder` | `M` |
| `como` | 7 | `numero` | 0,99 | `poder` | `M` |
| `ellos` | 3 | `tuyo` | 0,27 | `N` | `miercoles` |
| `gracias` | 8 | `4` | 0,97 | `mama` | `que` |
| `vivir` | 1 | `G` | 0,73 | `años` | `quien` |
| `vos` | 9 | `K` | 0,82 | `el_ella` | `T` |

---

## Comparativa

| | Luz buena | Luz mala |
|---|---:|---:|
| Top-1 literal | 978/1000 · 97,8 % | 922/1000 · 92,2 % |
| Top-1 con `0`≡`O` y `V`≡`viernes` | **996/1000 · 99,6 %** | **946/1000 · 94,6 %** |
| Top-3 literal | 1000/1000 · 100 % | 986/1000 · 98,6 % |
| Top-3 con la misma regla | 1000/1000 · 100 % | 987/1000 · 98,7 % |
| Señas en 10/10, con la regla | 97 | 77 |
| Errores que quedan | 4 | 54 |
| Confianza media en aciertos | 0,979 | 0,967 |
| Confianza media en errores | 0,632 | 0,771 |
| `gracias` / `ayudar` / `policia` | 10 · 10 · 10 | 9 · 9 · 10 |

La luz mala baja 5,0 puntos el top-1 de la regla (de 99,6 % a 94,6 %) y saca 20 señas del 10/10. El top-3 casi no se mueve: sigue por encima de 98 %. Las dos sesiones pasan el criterio de 75 % y 90 %.

Lo que se mantiene en las dos luces es el par `V`/`viernes`: las 10 tomas de `V` salen `viernes`, y `viernes` acierta las 10 por su nombre. `0` y `O` también se sostienen si se leen como una sola seña. En luz buena el cruce es solo `0`→`O`. En luz mala van los dos para cada lado, y una toma de `O` ni siquiera tiene la letra en el top-3: el primer puesto es `0`.

Lo que aparece con poca luz, y no estaba en la sesión buena, son unos pocos destinos repetidos: `T`→`I`, `como`→`numero`, `años`→`G`, `hola`→`chau`. `ahora_hoy` es la seña que más se dispersa: en luz mala sus 4 errores van a cuatro clases distintas y las cuatro quedan fuera del top-3.
