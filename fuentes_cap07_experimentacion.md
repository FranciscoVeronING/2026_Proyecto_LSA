# Fuentes para el capítulo 7: Experimentación, Resultados y Discusión

Repositorio: [FranciscoVeronING/2026_Proyecto_LSA](https://github.com/FranciscoVeronING/2026_Proyecto_LSA)

Rutas verificadas contra el remoto el 26/10/2026.

## 1. Clasificador

### Primeras iteraciones (junio y julio), rama `classifier`

- [docs/Entregable_Semana_Clasificador.md](https://github.com/FranciscoVeronING/2026_Proyecto_LSA/blob/classifier/docs/Entregable_Semana_Clasificador.md)
- [docs/documentacion.md](https://github.com/FranciscoVeronING/2026_Proyecto_LSA/blob/classifier/docs/documentacion.md)
- [src/model/entregables_nuevos/](https://github.com/FranciscoVeronING/2026_Proyecto_LSA/tree/classifier/src/model/entregables_nuevos): variantes de 16 fotogramas, 32 fotogramas (a y b) y Optuna del 31/07. Cada carpeta trae `metrics.json`, la evaluación en cámara `eval_91senias_*.csv`, la curva y la matriz.

### Agosto y septiembre (Optuna v2, 94 y 97 señas, refinamiento), rama `scratch-mediapipe-v2`

- [docs/Relevamiento_clasificador_agosto_2026.md](https://github.com/FranciscoVeronING/2026_Proyecto_LSA/blob/scratch-mediapipe-v2/docs/Relevamiento_clasificador_agosto_2026.md)
- [docs/cambios_scratch-mediapipe-v2.md](https://github.com/FranciscoVeronING/2026_Proyecto_LSA/blob/scratch-mediapipe-v2/docs/cambios_scratch-mediapipe-v2.md)
- [docs/ficha_tecnica_refinamiento_feature_engineering.md](https://github.com/FranciscoVeronING/2026_Proyecto_LSA/blob/scratch-mediapipe-v2/docs/ficha_tecnica_refinamiento_feature_engineering.md)
- [docs/inspeccion_features_mano_rostro.md](https://github.com/FranciscoVeronING/2026_Proyecto_LSA/blob/scratch-mediapipe-v2/docs/inspeccion_features_mano_rostro.md)
- [docs/eval_97senias_20260917.md](https://github.com/FranciscoVeronING/2026_Proyecto_LSA/blob/scratch-mediapipe-v2/docs/eval_97senias_20260917.md)
- [src/model/](https://github.com/FranciscoVeronING/2026_Proyecto_LSA/tree/scratch-mediapipe-v2/src/model): carpetas `2026_08_18_model_no_opt`, `2026_08_26_model_opt` y `2026_08_29_model_opt`, y `eval_36letras_numeros_20260915_141001.csv`.

### Holistic en navegador y reentrenamientos de octubre, rama `scratch-mediapipe-v3`

- [docs/reentrenamiento_100_senas_20261015.md](https://github.com/FranciscoVeronING/2026_Proyecto_LSA/blob/scratch-mediapipe-v3/docs/reentrenamiento_100_senas_20261015.md) (todavía no incorporado al capítulo)
- [src/eval_97senias_20260921_201459.csv](https://github.com/FranciscoVeronING/2026_Proyecto_LSA/blob/scratch-mediapipe-v3/src/eval_97senias_20260921_201459.csv)
- [src/model/](https://github.com/FranciscoVeronING/2026_Proyecto_LSA/tree/scratch-mediapipe-v3/src/model): checkpoint del 21/09, con curva y matriz.
- [src/model_new_2026_10_15/](https://github.com/FranciscoVeronING/2026_Proyecto_LSA/tree/scratch-mediapipe-v3/src/model_new_2026_10_15) y [src/model_new_2026_10_20/](https://github.com/FranciscoVeronING/2026_Proyecto_LSA/tree/scratch-mediapipe-v3/src/model_new_2026_10_20): reentrenamientos de 100 clases.

### Transfer learning (línea descartada)

- [README de experiment/transfer-mediapipe-16-frames-no-face](https://github.com/FranciscoVeronING/2026_Proyecto_LSA/blob/experiment/transfer-mediapipe-16-frames-no-face/README.md)
- [README de experiment/transfer-mediapipe-filtered](https://github.com/FranciscoVeronING/2026_Proyecto_LSA/blob/experiment/transfer-mediapipe-filtered/README.md)

## 2. Traducción semántica

### Primera ronda de modelos (julio), rama `deprecated/semantic`

- [docs/entregables_mvp_modulo_semantico.md](https://github.com/FranciscoVeronING/2026_Proyecto_LSA/blob/deprecated/semantic/docs/entregables_mvp_modulo_semantico.md): tabla de BLEU, METEOR y ROUGE-L de julio y descarte de DeepSeek.
- [docs/comparativa_llm_modulo_semantico.md](https://github.com/FranciscoVeronING/2026_Proyecto_LSA/blob/deprecated/semantic/docs/comparativa_llm_modulo_semantico.md): niveles de hardware. Sus latencias son estimaciones de diseño, no mediciones.

### Canal directo (glosas a español), rama `semantic-v2`

- [docs/entrenamiento_llm.md](https://github.com/FranciscoVeronING/2026_Proyecto_LSA/blob/semantic-v2/docs/entrenamiento_llm.md): protocolo de entrenamiento y de evaluación.
- [src/semantic/outputs/training_summary.json](https://github.com/FranciscoVeronING/2026_Proyecto_LSA/blob/semantic-v2/src/semantic/outputs/training_summary.json): todas las corridas sobre el 15 % reservado.
- Comparativas del conjunto de prueba fijo:
  - [07/09](https://github.com/FranciscoVeronING/2026_Proyecto_LSA/blob/semantic-v2/src/semantic/outputs/eval_trial/comparativa_trial_20260907_204012.md)
  - [16/09](https://github.com/FranciscoVeronING/2026_Proyecto_LSA/blob/semantic-v2/src/semantic/outputs/eval_trial/comparativa_trial_20260916_182300.md)
  - [05/10](https://github.com/FranciscoVeronING/2026_Proyecto_LSA/blob/semantic-v2/src/semantic/outputs/eval_trial/comparativa_trial_20261005_094906.md)
  - [13/10](https://github.com/FranciscoVeronING/2026_Proyecto_LSA/blob/semantic-v2/src/semantic/outputs/eval_trial/comparativa_trial_20261013_232706.md)
- [src/semantic/eval_compare_models.py](https://github.com/FranciscoVeronING/2026_Proyecto_LSA/blob/semantic-v2/src/semantic/eval_compare_models.py): parámetros de evaluación (temperatura 0, 24 tokens, todas las capas en GPU).

### Canal inverso (español a glosas), rama `semantic-reverse`

- [docs/entrenamiento.md](https://github.com/FranciscoVeronING/2026_Proyecto_LSA/blob/semantic-reverse/docs/entrenamiento.md): describe el protocolo, pero no hay resultados archivados.

### Servicio en Cloud Run, rama `semantic-cloud`

- [README.md](https://github.com/FranciscoVeronING/2026_Proyecto_LSA/blob/semantic-cloud/README.md): configuración (4 vCPU y 8 GiB), arranque en frío, caché del prompt, modelo principal y respaldo.

## 3. Sistema integrado

### Primera integración y latencia del clasificador, rama `integration`

- [docs/rama_integration.md](https://github.com/FranciscoVeronING/2026_Proyecto_LSA/blob/integration/docs/rama_integration.md): error de 8 contra 4 cabezas, 0,78 y 1,08 ms del clasificador, evaluación del 06/08 con 10 % de coincidencia exacta y batería de la lógica.
- [docs/ficha_tecnica_integrador.md](https://github.com/FranciscoVeronING/2026_Proyecto_LSA/blob/integration/docs/ficha_tecnica_integrador.md)

### Extensión y motor ILSA, rama `extension`

- [docs/archivo/entregable_extension_2026-09.md](https://github.com/FranciscoVeronING/2026_Proyecto_LSA/blob/extension/docs/archivo/entregable_extension_2026-09.md): validación de punta a punta del 13/09 y comparación entre la máquina de desarrollo y la laptop objetivo.
- [docs/ARQUITECTURA.md](https://github.com/FranciscoVeronING/2026_Proyecto_LSA/blob/extension/docs/ARQUITECTURA.md): umbrales de captura, lista de comprobaciones y consulta de mantenimiento.
- [docs/EXTENSION.md](https://github.com/FranciscoVeronING/2026_Proyecto_LSA/blob/extension/docs/EXTENSION.md)
- [docs/AVATAR.md](https://github.com/FranciscoVeronING/2026_Proyecto_LSA/blob/extension/docs/AVATAR.md)
- [README.md](https://github.com/FranciscoVeronING/2026_Proyecto_LSA/blob/extension/README.md)
- [src/backend/metrics.py](https://github.com/FranciscoVeronING/2026_Proyecto_LSA/blob/extension/src/backend/metrics.py): latencias que ILSA registra en cada sesión.

### Servicio de poses, rama `avatar-cloud`

- [README.md](https://github.com/FranciscoVeronING/2026_Proyecto_LSA/blob/avatar-cloud/README.md)

## Lo que no está en GitHub

- **Registros de latencia de las sesiones.** ILSA los guarda en cada computadora, en `%LOCALAPPDATA%\ILSA\metrics\live-AAAAMMDD.jsonl`. Con esos archivos se puede armar la tabla de latencia de punta a punta con mediana y percentil 95.
- **Métricas del canal inverso.** Habría que buscarlas en la carpeta `outputs` de la máquina donde se entrenó.
