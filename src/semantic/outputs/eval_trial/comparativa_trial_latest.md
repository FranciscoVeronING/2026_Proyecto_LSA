# Comparativa de modelos — dataset_trial

- Dataset: `D:\Mis Archivos\Proyectos\2026_Proyecto_LSA\src\semantic\dataset_trial.json`
- Ejemplos: 18
- Fecha: 2026-09-16T21:23:00.269610+00:00

## Ranking

| # | Modelo | Backend | Acc. norm. % | Acc. estricta % | BLEU | ROUGE-L | METEOR | Latencia ms |
|---|--------|---------|--------------|-----------------|------|---------|--------|-------------|
| 1 | Qwen2.5 1.5B | gguf | 100.00 | 100.00 | 100.00 | 100.00 | 99.47 | 413.34 |
| 2 | Qwen2.5 3B (mayor capacidad) | gguf | 100.00 | 100.00 | 100.00 | 100.00 | 99.47 | 801.29 |
| 3 | Llama 3.2 1B (Meta) | gguf | 94.44 | 94.44 | 97.07 | 98.89 | 98.14 | 335.89 |
| 4 | Qwen2.5 0.5B (ultra liviano) | gguf | 88.89 | 88.89 | 92.29 | 96.51 | 95.91 | 228.48 |
| 5 | SmolLM2 1.7B | gguf | 66.67 | 66.67 | 83.89 | 91.02 | 90.29 | 676.05 |

## Predicciones por ejemplo

### pronombre_objeto_segunda_persona_en_pasado

- Glosas: `PASADO YO VOS LLAMAR`
- Referencias:
  - Te llamé.
  - Yo te llamé.
  - Te llamé a vos.
  - Yo te llamé a vos.

| Modelo | Predicción | Exacta (norm.) |
|--------|------------|----------------|
| Qwen2.5 1.5B | Yo te llamé. | sí |
| Qwen2.5 3B (mayor capacidad) | Yo te llamé. | sí |
| Llama 3.2 1B (Meta) | Yo te llamé a vos. | sí |
| Qwen2.5 0.5B (ultra liviano) | Te llamé. | sí |
| SmolLM2 1.7B | Yo te llamé. | sí |

### pronombre_objeto_segunda_persona_en_pasado_percepcion

- Glosas: `PASADO YO VOS VER`
- Referencias:
  - Te vi.
  - Yo te vi.
  - Te vi a vos.
  - Yo te vi a vos.

| Modelo | Predicción | Exacta (norm.) |
|--------|------------|----------------|
| Qwen2.5 1.5B | Yo te vi a vos. | sí |
| Qwen2.5 3B (mayor capacidad) | Yo te vi. | sí |
| Llama 3.2 1B (Meta) | Yo te vi a vos. | sí |
| Qwen2.5 0.5B (ultra liviano) | Te vi. | sí |
| SmolLM2 1.7B | Yo te vi. | sí |

### pronombre_objeto_segunda_persona_futuro

- Glosas: `FUTURO YO VOS TENER`
- Referencias:
  - Te voy a tener.
  - Yo te voy a tener.
  - Te tendré.
  - Yo te tendré.
  - Te voy a tener a vos.
  - Yo te voy a tener a vos.

| Modelo | Predicción | Exacta (norm.) |
|--------|------------|----------------|
| Qwen2.5 1.5B | Te voy a tener. | sí |
| Qwen2.5 3B (mayor capacidad) | Te voy a tener. | sí |
| Llama 3.2 1B (Meta) | Te voy a tener. | sí |
| Qwen2.5 0.5B (ultra liviano) | Te voy a tener. | sí |
| SmolLM2 1.7B | Te voy a tener. | sí |

### pregunta_wh_ubicacion_tiempo_pasado_con_locativo

- Glosas: `PASADO PLAZA ELLOS MUJER VIVIR DONDE`
- Referencias:
  - ¿Dónde vivían ellas en la plaza?
  - ¿Dónde vivieron ellas en la plaza?

| Modelo | Predicción | Exacta (norm.) |
|--------|------------|----------------|
| Qwen2.5 1.5B | ¿Dónde vivieron ellas en la plaza? | sí |
| Qwen2.5 3B (mayor capacidad) | ¿Dónde vivían ellas en la plaza? | sí |
| Llama 3.2 1B (Meta) | ¿Dónde vivían ellas en la plaza? | sí |
| Qwen2.5 0.5B (ultra liviano) | ¿Dónde vivieron ellas en la plaza? | sí |
| SmolLM2 1.7B | ¿Dónde vivían ellas en la plaza? | sí |

### pregunta_wh_ubicacion_tiempo_pasado_segunda_persona

- Glosas: `PASADO CALLE VOS VIVIR DONDE`
- Referencias:
  - ¿Dónde vivías en la calle?
  - ¿Dónde vivías vos en la calle?
  - ¿Dónde viviste en la calle?
  - ¿Dónde viviste vos en la calle?

| Modelo | Predicción | Exacta (norm.) |
|--------|------------|----------------|
| Qwen2.5 1.5B | ¿Dónde vivías vos en la calle? | sí |
| Qwen2.5 3B (mayor capacidad) | ¿Dónde vivías vos en la calle? | sí |
| Llama 3.2 1B (Meta) | ¿Dónde vivías vos en la calle? | sí |
| Qwen2.5 0.5B (ultra liviano) | ¿Dónde viviste en la calle? | sí |
| SmolLM2 1.7B | ¿Dónde vivías vos en la calle? | sí |

### pregunta_wh_objeto_tiempo_pasado_transitivo

- Glosas: `PASADO ELLOS TENER QUE`
- Referencias:
  - ¿Qué tuvieron?
  - ¿Qué tenían?

| Modelo | Predicción | Exacta (norm.) |
|--------|------------|----------------|
| Qwen2.5 1.5B | ¿Qué tuvieron? | sí |
| Qwen2.5 3B (mayor capacidad) | ¿Qué tuvieron? | sí |
| Llama 3.2 1B (Meta) | ¿Qué tenían? | sí |
| Qwen2.5 0.5B (ultra liviano) | ¿Qué tenían? | sí |
| SmolLM2 1.7B | ¿Qué tenían ellos? | no |

### pregunta_wh_objeto_tiempo_pasado_voseo

- Glosas: `PASADO VOS LLAMAR QUE`
- Referencias:
  - ¿Qué llamaste?
  - ¿Qué llamaste vos?

| Modelo | Predicción | Exacta (norm.) |
|--------|------------|----------------|
| Qwen2.5 1.5B | ¿Qué llamaste? | sí |
| Qwen2.5 3B (mayor capacidad) | ¿Qué llamaste? | sí |
| Llama 3.2 1B (Meta) | ¿Qué llamaste? | sí |
| Qwen2.5 0.5B (ultra liviano) | ¿Qué llamaste? | sí |
| SmolLM2 1.7B | ¿Qué llamaste? | sí |

### pronombre_objeto_en_pasado_tercera_persona

- Glosas: `PASADO ELLOS HOMBRE YO VER`
- Referencias:
  - Ellos me vieron.
  - Ellos me vieron a mí.

| Modelo | Predicción | Exacta (norm.) |
|--------|------------|----------------|
| Qwen2.5 1.5B | Ellos me vieron. | sí |
| Qwen2.5 3B (mayor capacidad) | Ellos me vieron. | sí |
| Llama 3.2 1B (Meta) | Ellos me vieron. | sí |
| Qwen2.5 0.5B (ultra liviano) | Ellos me vi themselves. | no |
| SmolLM2 1.7B | Ellos me vi. | no |

### pregunta_wh_identidad_tiempo_pasado

- Glosas: `PASADO HERMANO TUYO QUIEN`
- Referencias:
  - ¿Quién era tu hermano?
  - ¿Quién fue tu hermano?

| Modelo | Predicción | Exacta (norm.) |
|--------|------------|----------------|
| Qwen2.5 1.5B | ¿Quién era tu hermano? | sí |
| Qwen2.5 3B (mayor capacidad) | ¿Quién era tu hermano? | sí |
| Llama 3.2 1B (Meta) | ¿Quién era tu hermano? | sí |
| Qwen2.5 0.5B (ultra liviano) | ¿Quién era tu hermano? | sí |
| SmolLM2 1.7B | ¿Quién era tu hermano? | sí |

### pregunta_wh_identidad_tiempo_futuro

- Glosas: `FUTURO ESPOSO MUJER TUYA QUIEN`
- Referencias:
  - ¿Quién va a ser tu esposa?
  - ¿Quién será tu esposa?

| Modelo | Predicción | Exacta (norm.) |
|--------|------------|----------------|
| Qwen2.5 1.5B | ¿Quién será tu esposa? | sí |
| Qwen2.5 3B (mayor capacidad) | ¿Quién será tu esposa? | sí |
| Llama 3.2 1B (Meta) | ¿Quién es tu esposa? | no |
| Qwen2.5 0.5B (ultra liviano) | ¿Quién es tu esposa? | no |
| SmolLM2 1.7B | ¿Quién es tu esposa? | no |

### pregunta_wh_ubicacion_futuro_transitivo

- Glosas: `FUTURO VOS DOCUMENTO LLEVAR DONDE`
- Referencias:
  - ¿A dónde vas a llevar el documento?
  - ¿Dónde vas a llevar el documento?
  - ¿Dónde vas a llevar el documento vos?
  - ¿A dónde llevarás el documento?
  - ¿Dónde llevarás el documento?

| Modelo | Predicción | Exacta (norm.) |
|--------|------------|----------------|
| Qwen2.5 1.5B | ¿Dónde vas a llevar el documento? | sí |
| Qwen2.5 3B (mayor capacidad) | ¿Dónde vas a llevar el documento? | sí |
| Llama 3.2 1B (Meta) | ¿Dónde vas a llevar el documento? | sí |
| Qwen2.5 0.5B (ultra liviano) | ¿Dónde vas a llevar el documento? | sí |
| SmolLM2 1.7B | ¿Dónde vas a llevar el documento? | sí |

### tiempo_futuro_objeto_y_negacion

- Glosas: `FUTURO YO CUCHILLO LLEVAR NO`
- Referencias:
  - No voy a llevar el cuchillo.
  - Yo no voy a llevar el cuchillo.
  - No llevaré el cuchillo.

| Modelo | Predicción | Exacta (norm.) |
|--------|------------|----------------|
| Qwen2.5 1.5B | No llevaré el cuchillo. | sí |
| Qwen2.5 3B (mayor capacidad) | No voy a llevar el cuchillo. | sí |
| Llama 3.2 1B (Meta) | No llevaré el cuchillo. | sí |
| Qwen2.5 0.5B (ultra liviano) | No llevaré el cuchillo. | sí |
| SmolLM2 1.7B | No llevaré el cuchillo. | sí |

### tiempo_pasado_objeto_y_negacion

- Glosas: `PASADO YO DOCUMENTO LLEVAR NO`
- Referencias:
  - No llevé el documento.
  - Yo no llevé el documento.

| Modelo | Predicción | Exacta (norm.) |
|--------|------------|----------------|
| Qwen2.5 1.5B | No llevé el documento. | sí |
| Qwen2.5 3B (mayor capacidad) | Yo no llevé el documento. | sí |
| Llama 3.2 1B (Meta) | Yo no llevé el documento. | sí |
| Qwen2.5 0.5B (ultra liviano) | Yo no llevé el documento. | sí |
| SmolLM2 1.7B | No llevo el documento. | no |

### flexion_genero_parentesco_transitivo_pasado

- Glosas: `PASADO YO HIJO MIO LLAMAR`
- Referencias:
  - Llamé a mi hijo.
  - Yo llamé a mi hijo.
  - Llamé al hijo.

| Modelo | Predicción | Exacta (norm.) |
|--------|------------|----------------|
| Qwen2.5 1.5B | Yo llamé a mi hijo. | sí |
| Qwen2.5 3B (mayor capacidad) | Yo llamé a mi hijo. | sí |
| Llama 3.2 1B (Meta) | Yo llamé a mi hijo. | sí |
| Qwen2.5 0.5B (ultra liviano) | Yo llamé a mi hijo. | sí |
| SmolLM2 1.7B | Yo me llamé a mi hijo. | no |

### deletreo_nombre_femenino_no_visto

- Glosas: `NOMBRE MIO Micaela`
- Referencias:
  - Me llamo Micaela.
  - Yo me llamo Micaela.
  - Mi nombre es Micaela.
  - Soy Micaela.
  - Yo soy Micaela.

| Modelo | Predicción | Exacta (norm.) |
|--------|------------|----------------|
| Qwen2.5 1.5B | Mi nombre es Micaela. | sí |
| Qwen2.5 3B (mayor capacidad) | Yo me llamo Micaela. | sí |
| Llama 3.2 1B (Meta) | Yo me llamo Micaela. | sí |
| Qwen2.5 0.5B (ultra liviano) | Mi nombre es Micaela. | sí |
| SmolLM2 1.7B | Mi nombre es Micaela. | sí |

### deletreo_apellido_con_grafemas_complejos

- Glosas: `APELLIDO MIO Guerra`
- Referencias:
  - Mi apellido es Guerra.
  - Me apellido Guerra.
  - Yo me apellido Guerra.

| Modelo | Predicción | Exacta (norm.) |
|--------|------------|----------------|
| Qwen2.5 1.5B | Mi apellido es Guerra. | sí |
| Qwen2.5 3B (mayor capacidad) | Mi apellido es Guerra. | sí |
| Llama 3.2 1B (Meta) | Mi apellido es Guerra. | sí |
| Qwen2.5 0.5B (ultra liviano) | Mi apellido es Guerra. | sí |
| SmolLM2 1.7B | Guerra apellido. | no |

### deletreo_calle_no_vista

- Glosas: `CALLE Moreno YO VIVIR_EN`
- Referencias:
  - Vivo en la calle Moreno.
  - Yo vivo en la calle Moreno.
  - Mi domicilio es en la calle Moreno.
  - Mi casa está en la calle Moreno.

| Modelo | Predicción | Exacta (norm.) |
|--------|------------|----------------|
| Qwen2.5 1.5B | Vivo en la calle Moreno. | sí |
| Qwen2.5 3B (mayor capacidad) | Vivo en la calle Moreno. | sí |
| Llama 3.2 1B (Meta) | Vivo en la calle Moreno. | sí |
| Qwen2.5 0.5B (ultra liviano) | Vivo en la calle Moreno. | sí |
| SmolLM2 1.7B | Vivo en la calle Moreno. | sí |

### secuencia_numerica_dni_arbitraria

- Glosas: `DOCUMENTO MIO NUMERO 18333940`
- Referencias:
  - Mi número de documento es 18333940.
  - Mi documento es 18333940.
  - Mi número de DNI es 18333940.
  - Mi DNI es 18333940.
  - El número de mi documento es 18333940.

| Modelo | Predicción | Exacta (norm.) |
|--------|------------|----------------|
| Qwen2.5 1.5B | Mi DNI es 18333940. | sí |
| Qwen2.5 3B (mayor capacidad) | Mi número de DNI es 18333940. | sí |
| Llama 3.2 1B (Meta) | Mi DNI es 18333940. | sí |
| Qwen2.5 0.5B (ultra liviano) | Mi DNI es 18333940. | sí |
| SmolLM2 1.7B | Mi DNI es 18333940. | sí |


## Métricas por categoría

| Categoría | qwen2.5-1.5b | qwen2.5-3b | llama-3.2-1b | qwen2.5-0.5b | smollm2-1.7b |
|-----------|--------|--------|--------|--------|--------|
| `deletreo_apellido_con_grafemas_complejos` | 100% | 100% | 100% | 100% | 0% |
| `deletreo_calle_no_vista` | 100% | 100% | 100% | 100% | 100% |
| `deletreo_nombre_femenino_no_visto` | 100% | 100% | 100% | 100% | 100% |
| `flexion_genero_parentesco_transitivo_pasado` | 100% | 100% | 100% | 100% | 0% |
| `pregunta_wh_identidad_tiempo_futuro` | 100% | 100% | 0% | 0% | 0% |
| `pregunta_wh_identidad_tiempo_pasado` | 100% | 100% | 100% | 100% | 100% |
| `pregunta_wh_objeto_tiempo_pasado_transitivo` | 100% | 100% | 100% | 100% | 0% |
| `pregunta_wh_objeto_tiempo_pasado_voseo` | 100% | 100% | 100% | 100% | 100% |
| `pregunta_wh_ubicacion_futuro_transitivo` | 100% | 100% | 100% | 100% | 100% |
| `pregunta_wh_ubicacion_tiempo_pasado_con_locativo` | 100% | 100% | 100% | 100% | 100% |
| `pregunta_wh_ubicacion_tiempo_pasado_segunda_persona` | 100% | 100% | 100% | 100% | 100% |
| `pronombre_objeto_en_pasado_tercera_persona` | 100% | 100% | 100% | 0% | 0% |
| `pronombre_objeto_segunda_persona_en_pasado` | 100% | 100% | 100% | 100% | 100% |
| `pronombre_objeto_segunda_persona_en_pasado_percepcion` | 100% | 100% | 100% | 100% | 100% |
| `pronombre_objeto_segunda_persona_futuro` | 100% | 100% | 100% | 100% | 100% |
| `secuencia_numerica_dni_arbitraria` | 100% | 100% | 100% | 100% | 100% |
| `tiempo_futuro_objeto_y_negacion` | 100% | 100% | 100% | 100% | 100% |
| `tiempo_pasado_objeto_y_negacion` | 100% | 100% | 100% | 100% | 0% |
