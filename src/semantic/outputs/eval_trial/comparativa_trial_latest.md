# Comparativa de modelos — dataset_trial

- Dataset: `D:\Mis Archivos\Proyectos\2026_Proyecto_LSA_rama2\src\semantic\dataset_trial.json`
- Ejemplos: 21
- Fecha: 2026-10-14T02:27:06.871643+00:00

## Ranking

| # | Modelo | Backend | Acc. norm. % | Acc. estricta % | BLEU | ROUGE-L | METEOR | Latencia ms |
|---|--------|---------|--------------|-----------------|------|---------|--------|-------------|
| 1 | Llama 3.2 1B (Meta) | gguf | 100.00 | 100.00 | 100.00 | 100.00 | 99.74 | 343.41 |
| 2 | Qwen2.5 3B (mayor capacidad) | gguf | 100.00 | 100.00 | 100.00 | 100.00 | 99.71 | 790.53 |
| 3 | Qwen2.5 1.5B | gguf | 100.00 | 100.00 | 100.00 | 100.00 | 99.69 | 412.99 |
| 4 | Qwen2.5 0.5B (ultra liviano) | gguf | 100.00 | 100.00 | 100.00 | 100.00 | 99.67 | 225.64 |
| 5 | SmolLM2 1.7B | gguf | 95.24 | 95.24 | 97.73 | 99.21 | 98.74 | 655.4 |

## Predicciones por ejemplo

### pronombre_objeto_en_pasado_tercera_persona_masculino

- Glosas: `PASADO ELLOS O YO VER`
- Referencias:
  - Ellos me vieron.
  - Ellos me vieron a mí.

| Modelo | Predicción | Exacta (norm.) |
|--------|------------|----------------|
| Llama 3.2 1B (Meta) | Ellos me vieron. | sí |
| Qwen2.5 3B (mayor capacidad) | Ellos me vieron. | sí |
| Qwen2.5 1.5B | Ellos me vieron. | sí |
| Qwen2.5 0.5B (ultra liviano) | Ellos me vieron. | sí |
| SmolLM2 1.7B | Ellos me vieron. | sí |

### pregunta_wh_identidad_tiempo_futuro_femenino

- Glosas: `FUTURO ESPOSO A TUYO QUIEN`
- Referencias:
  - ¿Quién va a ser tu esposa?
  - ¿Quién será tu esposa?

| Modelo | Predicción | Exacta (norm.) |
|--------|------------|----------------|
| Llama 3.2 1B (Meta) | ¿Quién va a ser tu esposa? | sí |
| Qwen2.5 3B (mayor capacidad) | ¿Quién va a ser tu esposa? | sí |
| Qwen2.5 1.5B | ¿Quién va a ser tu esposa? | sí |
| Qwen2.5 0.5B (ultra liviano) | ¿Quién va a ser tu esposa? | sí |
| SmolLM2 1.7B | ¿Quién va a ser tu esposa? | sí |

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
| Llama 3.2 1B (Meta) | ¿Dónde vas a llevar el documento? | sí |
| Qwen2.5 3B (mayor capacidad) | ¿Dónde vas a llevar el documento? | sí |
| Qwen2.5 1.5B | ¿Dónde vas a llevar el documento? | sí |
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
| Llama 3.2 1B (Meta) | No voy a llevar el cuchillo. | sí |
| Qwen2.5 3B (mayor capacidad) | No voy a llevar el cuchillo. | sí |
| Qwen2.5 1.5B | No voy a llevar el cuchillo. | sí |
| Qwen2.5 0.5B (ultra liviano) | Yo no voy a llevar el cuchillo. | sí |
| SmolLM2 1.7B | No voy a llevar el cuchillo. | sí |

### tiempo_pasado_objeto_y_negacion

- Glosas: `PASADO YO DOCUMENTO LLEVAR NO`
- Referencias:
  - No llevé el documento.
  - Yo no llevé el documento.

| Modelo | Predicción | Exacta (norm.) |
|--------|------------|----------------|
| Llama 3.2 1B (Meta) | Yo no llevé el documento. | sí |
| Qwen2.5 3B (mayor capacidad) | Yo no llevé el documento. | sí |
| Qwen2.5 1.5B | No llevé el documento. | sí |
| Qwen2.5 0.5B (ultra liviano) | No llevé el documento. | sí |
| SmolLM2 1.7B | No llevé el documento. | sí |

### flexion_genero_parentesco_transitivo_pasado_defecto

- Glosas: `PASADO YO HIJO MIO LLAMAR`
- Referencias:
  - Llamé a mi hijo.
  - Yo llamé a mi hijo.
  - Llamé al hijo.

| Modelo | Predicción | Exacta (norm.) |
|--------|------------|----------------|
| Llama 3.2 1B (Meta) | Yo llamé a mi hijo. | sí |
| Qwen2.5 3B (mayor capacidad) | Yo llamé a mi hijo. | sí |
| Qwen2.5 1.5B | Yo llamé a mi hijo. | sí |
| Qwen2.5 0.5B (ultra liviano) | Llamé a mi hijo. | sí |
| SmolLM2 1.7B | Llamé a mi hijo. | sí |

### flexion_genero_parentesco_transitivo_pasado_masculino_explicito

- Glosas: `PASADO YO HIJO O MIO LLAMAR`
- Referencias:
  - Llamé a mi hijo.
  - Yo llamé a mi hijo.
  - Llamé al hijo.

| Modelo | Predicción | Exacta (norm.) |
|--------|------------|----------------|
| Llama 3.2 1B (Meta) | Yo llamé a mi hijo. | sí |
| Qwen2.5 3B (mayor capacidad) | Yo llamé a mi hijo. | sí |
| Qwen2.5 1.5B | Llamé a mi hijo. | sí |
| Qwen2.5 0.5B (ultra liviano) | Llamé a mi hijo. | sí |
| SmolLM2 1.7B | Llamé a mi hijo. | sí |

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
| Llama 3.2 1B (Meta) | Yo me llamo Micaela. | sí |
| Qwen2.5 3B (mayor capacidad) | Me llamo Micaela. | sí |
| Qwen2.5 1.5B | Mi nombre es Micaela. | sí |
| Qwen2.5 0.5B (ultra liviano) | Mi nombre es Micaela. | sí |
| SmolLM2 1.7B | Me llamo Micaela. | sí |

### deletreo_apellido_con_grafemas_complejos

- Glosas: `APELLIDO MIO Guerra`
- Referencias:
  - Mi apellido es Guerra.
  - Me apellido Guerra.
  - Yo me apellido Guerra.

| Modelo | Predicción | Exacta (norm.) |
|--------|------------|----------------|
| Llama 3.2 1B (Meta) | Mi apellido es Guerra. | sí |
| Qwen2.5 3B (mayor capacidad) | Mi apellido es Guerra. | sí |
| Qwen2.5 1.5B | Mi apellido es Guerra. | sí |
| Qwen2.5 0.5B (ultra liviano) | Mi apellido es Guerra. | sí |
| SmolLM2 1.7B | Me apellido Guerra. | sí |

### deletreo_calle_no_vista

- Glosas: `CALLE Moreno YO VIVIR_EN`
- Referencias:
  - Vivo en la calle Moreno.
  - Yo vivo en la calle Moreno.
  - Mi domicilio es en la calle Moreno.
  - Mi casa está en la calle Moreno.

| Modelo | Predicción | Exacta (norm.) |
|--------|------------|----------------|
| Llama 3.2 1B (Meta) | Yo vivo en la calle Moreno. | sí |
| Qwen2.5 3B (mayor capacidad) | Yo vivo en la calle Moreno. | sí |
| Qwen2.5 1.5B | Vivo en la calle Moreno. | sí |
| Qwen2.5 0.5B (ultra liviano) | Vivo en la calle Moreno. | sí |
| SmolLM2 1.7B | Yo vivo en la calle Moreno. | sí |

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
| Llama 3.2 1B (Meta) | Mi número de DNI es 18333940. | sí |
| Qwen2.5 3B (mayor capacidad) | Mi DNI es 18333940. | sí |
| Qwen2.5 1.5B | Mi DNI es 18333940. | sí |
| Qwen2.5 0.5B (ultra liviano) | Mi DNI es 18333940. | sí |
| SmolLM2 1.7B | Mi número de DNI es 18333940. | sí |

### policia_transitivo_objeto_presente

- Glosas: `YO POLICIA LLAMAR`
- Referencias:
  - Llamo a la policía.
  - Yo llamo a la policía.

| Modelo | Predicción | Exacta (norm.) |
|--------|------------|----------------|
| Llama 3.2 1B (Meta) | Yo llamo a la policía. | sí |
| Qwen2.5 3B (mayor capacidad) | Yo llamo a la policía. | sí |
| Qwen2.5 1.5B | Llamo a la policía. | sí |
| Qwen2.5 0.5B (ultra liviano) | Llamo a la policía. | sí |
| SmolLM2 1.7B | Llamo a la policía. | sí |

### policia_transitivo_objeto_futuro

- Glosas: `FUTURO YO POLICIA LLAMAR`
- Referencias:
  - Voy a llamar a la policía.
  - Yo voy a llamar a la policía.
  - Llamaré a la policía.
  - Yo llamaré a la policía.

| Modelo | Predicción | Exacta (norm.) |
|--------|------------|----------------|
| Llama 3.2 1B (Meta) | Yo voy a llamar a la policía. | sí |
| Qwen2.5 3B (mayor capacidad) | Yo voy a llamar a la policía. | sí |
| Qwen2.5 1.5B | Voy a llamar a la policía. | sí |
| Qwen2.5 0.5B (ultra liviano) | Llamaré a la policía. | sí |
| SmolLM2 1.7B | Llamaré a la policía. | sí |

### policia_transitivo_objeto_pasado

- Glosas: `PASADO YO POLICIA LLAMAR`
- Referencias:
  - Llamé a la policía.
  - Yo llamé a la policía.

| Modelo | Predicción | Exacta (norm.) |
|--------|------------|----------------|
| Llama 3.2 1B (Meta) | Llamé a la policía. | sí |
| Qwen2.5 3B (mayor capacidad) | Llamé a la policía. | sí |
| Qwen2.5 1.5B | Llamé a la policía. | sí |
| Qwen2.5 0.5B (ultra liviano) | Llamé a la policía. | sí |
| SmolLM2 1.7B | Llamé a la policía. | sí |

### policia_sujeto_agente_ayuda_presente

- Glosas: `POLICIA YO AYUDAR`
- Referencias:
  - La policía me ayuda.
  - La policía me está ayudando.

| Modelo | Predicción | Exacta (norm.) |
|--------|------------|----------------|
| Llama 3.2 1B (Meta) | La policía me ayuda. | sí |
| Qwen2.5 3B (mayor capacidad) | La policía me ayuda. | sí |
| Qwen2.5 1.5B | La policía me ayuda. | sí |
| Qwen2.5 0.5B (ultra liviano) | La policía me ayuda. | sí |
| SmolLM2 1.7B | La policía me ayuda. | sí |

### policia_sujeto_agente_ayuda_modal_poder

- Glosas: `POLICIA YO AYUDAR PODER`
- Referencias:
  - La policía me puede ayudar.
  - La policía es capaz de ayudarme.

| Modelo | Predicción | Exacta (norm.) |
|--------|------------|----------------|
| Llama 3.2 1B (Meta) | La policía me puede ayudar. | sí |
| Qwen2.5 3B (mayor capacidad) | La policía me puede ayudar. | sí |
| Qwen2.5 1.5B | La policía me puede ayudar. | sí |
| Qwen2.5 0.5B (ultra liviano) | La policía me puede ayudar. | sí |
| SmolLM2 1.7B | La policía me puede ayudar. | sí |

### policia_percepcion_locativo_calle

- Glosas: `CALLE YO POLICIA VER`
- Referencias:
  - Veo a la policía en la calle.
  - Yo veo a la policía en la calle.

| Modelo | Predicción | Exacta (norm.) |
|--------|------------|----------------|
| Llama 3.2 1B (Meta) | Veo a la policía en la calle. | sí |
| Qwen2.5 3B (mayor capacidad) | Veo a la policía en la calle. | sí |
| Qwen2.5 1.5B | Veo a la policía en la calle. | sí |
| Qwen2.5 0.5B (ultra liviano) | Veo a la policía en la calle. | sí |
| SmolLM2 1.7B | Veo a la policía en la calle. | sí |

### policia_pregunta_wh_ubicacion

- Glosas: `POLICIA DONDE`
- Referencias:
  - ¿Dónde está la policía?
  - ¿Dónde queda la policía?

| Modelo | Predicción | Exacta (norm.) |
|--------|------------|----------------|
| Llama 3.2 1B (Meta) | ¿Dónde queda la policía? | sí |
| Qwen2.5 3B (mayor capacidad) | ¿Dónde queda la policía? | sí |
| Qwen2.5 1.5B | ¿Dónde queda la policía? | sí |
| Qwen2.5 0.5B (ultra liviano) | ¿Dónde queda la policía? | sí |
| SmolLM2 1.7B | ¿Dónde está la policía? | sí |

### policia_pregunta_wh_agente_pasado

- Glosas: `PASADO POLICIA LLAMAR QUIEN`
- Referencias:
  - ¿Quién llamó a la policía?

| Modelo | Predicción | Exacta (norm.) |
|--------|------------|----------------|
| Llama 3.2 1B (Meta) | ¿Quién llamó a la policía? | sí |
| Qwen2.5 3B (mayor capacidad) | ¿Quién llamó a la policía? | sí |
| Qwen2.5 1.5B | ¿Quién llamó a la policía? | sí |
| Qwen2.5 0.5B (ultra liviano) | ¿Quién llamó a la policía? | sí |
| SmolLM2 1.7B | ¿Quién llamó a la policía? | sí |

### policia_objeto_pasado_negacion

- Glosas: `PASADO YO POLICIA LLAMAR NO`
- Referencias:
  - No llamé a la policía.
  - Yo no llamé a la policía.

| Modelo | Predicción | Exacta (norm.) |
|--------|------------|----------------|
| Llama 3.2 1B (Meta) | Yo no llamé a la policía. | sí |
| Qwen2.5 3B (mayor capacidad) | Yo no llamé a la policía. | sí |
| Qwen2.5 1.5B | No llamé a la policía. | sí |
| Qwen2.5 0.5B (ultra liviano) | No llamé a la policía. | sí |
| SmolLM2 1.7B | No llamé a la policía. | sí |

### policia_predicado_parentesco_femenino_explicito

- Glosas: `POLICIA HERMANO A MIO`
- Referencias:
  - La policía es mi hermana.

| Modelo | Predicción | Exacta (norm.) |
|--------|------------|----------------|
| Llama 3.2 1B (Meta) | La policía es mi hermana. | sí |
| Qwen2.5 3B (mayor capacidad) | La policía es mi hermana. | sí |
| Qwen2.5 1.5B | La policía es mi hermana. | sí |
| Qwen2.5 0.5B (ultra liviano) | La policía es mi hermana. | sí |
| SmolLM2 1.7B | La policía es mi hermano. | no |


## Métricas por categoría

| Categoría | llama-3.2-1b | qwen2.5-3b | qwen2.5-1.5b | qwen2.5-0.5b | smollm2-1.7b |
|-----------|--------|--------|--------|--------|--------|
| `deletreo_apellido_con_grafemas_complejos` | 100% | 100% | 100% | 100% | 100% |
| `deletreo_calle_no_vista` | 100% | 100% | 100% | 100% | 100% |
| `deletreo_nombre_femenino_no_visto` | 100% | 100% | 100% | 100% | 100% |
| `flexion_genero_parentesco_transitivo_pasado_defecto` | 100% | 100% | 100% | 100% | 100% |
| `flexion_genero_parentesco_transitivo_pasado_masculino_explicito` | 100% | 100% | 100% | 100% | 100% |
| `policia_objeto_pasado_negacion` | 100% | 100% | 100% | 100% | 100% |
| `policia_percepcion_locativo_calle` | 100% | 100% | 100% | 100% | 100% |
| `policia_predicado_parentesco_femenino_explicito` | 100% | 100% | 100% | 100% | 0% |
| `policia_pregunta_wh_agente_pasado` | 100% | 100% | 100% | 100% | 100% |
| `policia_pregunta_wh_ubicacion` | 100% | 100% | 100% | 100% | 100% |
| `policia_sujeto_agente_ayuda_modal_poder` | 100% | 100% | 100% | 100% | 100% |
| `policia_sujeto_agente_ayuda_presente` | 100% | 100% | 100% | 100% | 100% |
| `policia_transitivo_objeto_futuro` | 100% | 100% | 100% | 100% | 100% |
| `policia_transitivo_objeto_pasado` | 100% | 100% | 100% | 100% | 100% |
| `policia_transitivo_objeto_presente` | 100% | 100% | 100% | 100% | 100% |
| `pregunta_wh_identidad_tiempo_futuro_femenino` | 100% | 100% | 100% | 100% | 100% |
| `pregunta_wh_ubicacion_futuro_transitivo` | 100% | 100% | 100% | 100% | 100% |
| `pronombre_objeto_en_pasado_tercera_persona_masculino` | 100% | 100% | 100% | 100% | 100% |
| `secuencia_numerica_dni_arbitraria` | 100% | 100% | 100% | 100% | 100% |
| `tiempo_futuro_objeto_y_negacion` | 100% | 100% | 100% | 100% | 100% |
| `tiempo_pasado_objeto_y_negacion` | 100% | 100% | 100% | 100% | 100% |
