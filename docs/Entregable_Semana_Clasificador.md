# Entregable semana — Clasificador LSA

> **Rama:** `scratch-mediapipe`  
> **Autores:** Maite Nigro, Francisco Veron  
> **Fecha entrega:** _[completar viernes]_  
> **Estado:** borrador — modificar libremente antes de exportar a PDF

---

## 1. Resumen ejecutivo

_[1 párrafo: qué se entrega esta semana, métricas top-1/top-3 en prueba de 91 señas, modelo local funcionando en `camera.py`]_

**Objetivo de precisión:** ~90% ±10% en general (reportar top-1 y top-3 por separado).

---

## 2. Trabajo previo (antes de esta semana)

### 2.1 Proyecto y protocolo

- Prototipo de traducción LSA con procesamiento en tiempo cuasi real (UNMDP, Ing. Informática, Tipo 1).
- Directores: Esp. Ing. Gonzalo Avalos Ribas, Ing. Bruno Constanzo.
- Repositorio activo: [2026_Proyecto_LSA](https://github.com/FranciscoVeronING/2026_Proyecto_LSA), rama `scratch-mediapipe`.
- Repositorio anterior descartado por orden: `2026---Trabajo-Final-Informatica`.

### 2.2 Investigación y dataset

- Evaluación LSTM vs Transformer; landmarks vs frames → se eligió **MediaPipe + landmarks + Tiny Transformer**.
- Dataset ASL de prueba **descartado** (etiquetas incorrectas, dialectos, videos espejados).
- Dataset **LSA propio**: ~100 señas, vocabulario validado por experta (Miriam Rolls), contexto policial/viajero.
- Grabaciones iniciales con variación de sujetos, ropa, lentes; **ampliación con +7 personas pendiente** (próximas semanas).

### 2.3 Arquitectura del clasificador

| Componente | Detalle |
|-----------|---------|
| Extracción | MediaPipe Holistic (pose + manos, 225 features/frame) |
| Normalización | Ancla en hombros, escala inter-hombros |
| Secuencia | 16 frames (trim + subsampleo uniforme) |
| Modelo | Conv1D + Tiny Transformer (128 dim, 4 heads, 2 layers) |
| Pooling | Attention pooling |
| Clases activas | 91 (mínimo 50 videos/clase) |
| Regularización | Dropout 0.5, label smoothing, AdamW, early stopping, data augmentation 3D |

### 2.4 Iteraciones relevantes

- Reducción de frames 100 → 16 (menos oversampling).
- Eliminación malla facial; interpolación de ceros; augmentation matemática (ruido + escala, sin mirroring).
- Archivo de experimentos en `src/model/16_Frames/` y `32_Frames/`.
- Modelo activo previo al retrain de esta semana: commit `4738611` (13/07/2026).

---

## 3. Trabajo realizado esta semana

### 3.1 Mejoras de pipeline (train ↔ cámara)

| ID | Mejora | Archivo | Estado |
|----|--------|---------|--------|
| 1.1 | Trim + subsampleo alineado con `camera.py` | `preprocessing.py`, `utils.py` | Implementado |
| 1.5 | Interpolación de frames con ceros (MediaPipe) | `utils.py` | Implementado |
| 1.6 | Modal diestro/zurdo + mirror landmarks | `camera.py`, `utils.py` | Implementado |
| 3.2 | Top-3 siempre visible | `camera.py` | Implementado |
| 3.3 | Umbral confianza 0.75 | `config.py` | Implementado |
| 3.4 | Modo eval → CSV (`--eval`) | `camera.py` | Implementado |
| 3.5 | Cooldown 1 s post-inferencia | `config.py`, `camera.py` | Implementado |
| 2.3 | Export `metrics.json` al entrenar | `train.py` | Implementado |

### 3.2 Reentrenamiento

_[Completar después del jueves]_

```bash
cd src
python preprocessing.py --force   # regenerar .npy con nuevo pipeline
python train.py                   # nuevo tinyskeleton_best.pth + metrics.json
```

| Métrica | Valor |
|---------|-------|
| Val loss | _[completar]_ |
| Val acc top-1 | _[completar]_ % |
| Épocas | _[completar]_ |
| Fecha train | _[completar]_ |

### 3.3 Trabajo paralelo — Maite (transfer learning)

_[Breve párrafo: rama/enfoque de Maite, no incluido en el modelo de este entregable]_

---

## 4. Prueba en vivo — 91 señas

### 4.1 Protocolo

1. Ejecutar: `python camera.py --eval`
2. Elegir diestro/zurdo en modal inicial.
3. Por cada seña (1/91 … 91/91): realizar gesto → sostener ~1 s → bajar manos.
4. Tecla `n` = saltar seña; `q` = salir.

### 4.2 Resultados

| Métrica | Resultado | Objetivo |
|---------|-----------|----------|
| Top-1 | _[X]/91 = __%_ | 80–100% |
| Top-3 | _[Y]/91 = __%_ | 80–100% |
| CSV | `src/eval_91senias_YYYYMMDD.csv` | Adjunto / referenciado |

### 4.3 Señas con error (completar)

| Seña esperada | Predicho top-1 | En top-3 | Notas |
|---------------|----------------|----------|-------|
| _ej. mal_ | _0_ | _sí/no_ | _[completar]_ |

---

## 5. Uso del sistema

### Instalación (entorno conda)

```bash
conda activate <tu_entorno>
cd src
```

### Inferencia normal

```bash
python camera.py
```

### Evaluación formal

```bash
python camera.py --eval
```

### Controles

| Control | Acción |
|---------|--------|
| Modal inicial | Diestro / Zurdo (botón o tecla D/Z) |
| `m` | Cambiar modo captura (auto / static / dynamic) |
| `n` | Saltar seña (solo `--eval`) |
| Config | Umbral confianza, sensibilidad, silencio |
| CAPTURAR | Inferencia manual del buffer |

---

## 6. Archivos del entregable

| Archivo | Descripción |
|---------|-------------|
| `src/model/tinyskeleton_best.pth` | Pesos del clasificador |
| `src/model/mapeo_clases.json` | 91 clases |
| `src/model/metrics.json` | Métricas del último entrenamiento |
| `src/curva_tinyskeleton.png` | Curva train/val |
| `src/matriz_confusion_tinyskeleton.png` | Matriz de confusión |
| `src/eval_91senias_*.csv` | Prueba en vivo (modo `--eval`) |
| `docs/Entregable_Semana_Clasificador.md` | Este documento |

---

## 7. Limitaciones conocidas

- Dataset ampliado (+7 personas) **pendiente**; métricas en vivo pueden variar con nuevos sujetos.
- Espejado zurdo implementado en inferencia; videos zurdos en train requieren sufijo `_zurdo` en el nombre o metadata futura.
- Módulo semántico (LLM) e integración clasificador→texto: **próxima semana**.
- Extensión navegador: fase posterior.

---

## 8. Próximos pasos

- [ ] Grabaciones con +7 personas nuevas
- [ ] Reentrenar con dataset ampliado
- [ ] Integración `pipeline_local.py`: clasificador + LLM (`modulo-semantico`)
- [ ] Push repo + PDF final para cátedra
- [ ] Exportar este documento a LaTeX/PDF

---

## Anexo A — Comandos git (post-entrega)

```bash
git checkout scratch-mediapipe
git add ...
git commit -m "..."
git push -u origin scratch-mediapipe
```
