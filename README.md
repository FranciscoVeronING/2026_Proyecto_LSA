# Servicio de glosas del avatar

Esta rama publica los archivos de seña del avatar. Un cliente pide una glosa por HTTP y recibe el archivo `.pose` correspondiente. El servicio está pensado para correr en Google Cloud Run y también se puede probar en local.

Los archivos viven en `poses/`. El nombre de cada archivo, sin la extensión, es la glosa.

## Qué responde el servicio

El proceso escucha en `0.0.0.0` y toma el puerto de la variable de entorno `PORT`. Si esa variable no existe, usa el puerto `8080`. Cloud Run inyecta `PORT` al arrancar el contenedor.

Hay dos rutas.

### `GET /health`

Indica que el proceso está arriba. Responde `200` con este JSON:

```json
{"status": "ok"}
```

Sirve para el monitoreo de disponibilidad. No revisa si una glosa existe.

### `GET /tposes/{glosa}.pose`

Devuelve el archivo de esa glosa. El nombre tiene que coincidir exactamente con un archivo de `poses/`, incluida la extensión `.pose`.

Ejemplos:

- `GET /tposes/HOLA.pose`
- `GET /tposes/HOYAHORA.pose`
- `GET /tposes/ENIE.pose`
- `GET /tposes/PADRE.pose`

Si el archivo existe, la respuesta es `200` y el cuerpo es el binario `.pose`, con `Content-Type: application/octet-stream`. Si no existe, la respuesta es `404`.

La misma carpeta también puede servir `.json`, `.gltf`, `.bin` y `.fbx` si más adelante se agregan ahí. Esos formatos salen con su tipo de contenido correspondiente: JSON, `model/gltf+json` para `.gltf`, y binario para `.bin` y `.fbx`.

Al arrancar, el servicio exige que exista la carpeta `poses/`. Si no está, el proceso termina y no abre el puerto.

## CORS

El servicio acepta `GET`, `HEAD` y `OPTIONS` desde cualquier origen. Un cliente que corre en el navegador, en otro dominio, puede pedir las glosas sin que el navegador bloquee la respuesta.

## Formato de las glosas

ILSA tiene que transformar la glosa antes de armar la solicitud. El nombre final es una sola cadena de letras mayúsculas y números, sin tildes, sin espacios y sin otros signos. Después se pide así:

```text
/tposes/{GLOSA}.pose
```

Reglas, en este orden:

1. Pasar toda la glosa a mayúsculas.
2. Dejar la forma masculina. Si trae el sufijo de género `_A`, `_a` o `_RA`, se quita. `ABOGADO_A` queda `ABOGADO`. `PROFESOR_RA` queda `PROFESOR`. `EL_ELLA` queda `EL`.
3. Sacar las tildes de las vocales. `Á É Í Ó Ú Ü` pasan a `A E I O U U`. `CAFÉ` queda `CAFE`.
4. Borrar espacios, `_`, guiones, paréntesis, signos de exclamación y cualquier carácter que no sea letra o número.
5. Si una seña significa más de una cosa, las glosas se concatenan en un solo nombre, sin separador. No se piden como archivos distintos.

Ejemplos de señas con más de una glosa:

| Significado | Nombre del archivo |
| --- | --- |
| Hoy / ahora | `HOYAHORA.pose` |
| Anual / por período | `ANUALPORPERIODO.pose` |
| Sprite / 7Up | `SPRITE7UP.pose` |
| Camarero / mozo | `CAMAREROMOZO.pose` |
| Países Bajos / Holanda | `PAISESBAJOSHOLANDA.pose` |
| Época / período | `EPOCAPERIODO.pose` |

`PERIODO.pose` es otra grabación y se pide aparte de `EPOCAPERIODO.pose`.

Cuando `Ñ` aparece dentro de una palabra, se reemplaza por `N`. `ESPAÑA` queda `ESPANA`. `MAÑANA` queda `MANANA`. `AÑOS` queda `ANOS`.

## Excepciones

Estas dos sustituciones se aplican a la glosa completa. No se buscan adentro de otra palabra.

- La letra `Ñ`, cuando esa es toda la glosa, se envía como `ENIE`. El archivo es `ENIE.pose`. La letra `N` sigue siendo `N.pose`.
- La glosa `PAPA` se envía como `PADRE`. El archivo es `PADRE.pose`.

`PAPEL`, `PAPASFRITAS` y el resto de las palabras que contienen esas letras no cambian por estas excepciones. `PAPA.pose` existe en la carpeta y corresponde a la papa como alimento. La glosa de parentesco `PAPA` que ILSA envía al avatar se transforma a `PADRE` antes de la solicitud.

## Ejecución local

Desde la raíz del repositorio, con Python 3.10 o superior:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

Queda disponible en `http://localhost:8080`. Para comprobarlo:

```text
GET http://localhost:8080/health
GET http://localhost:8080/tposes/HOLA.pose
```

Las dependencias fijas son FastAPI `0.115.12` y Uvicorn `0.34.2`, elegidas porque funcionan con la imagen `python:3.10-slim`.

## Imagen de contenedor

`Dockerfile` parte de `python:3.10-slim`, instala `requirements.txt`, copia `main.py` y la carpeta `poses/`, y arranca con `python main.py`. El directorio de trabajo dentro del contenedor es `/app`.

`.dockerignore` deja afuera el entorno virtual, la caché de Python, `.git`, `.env` y `src/`. El código de `src/` no entra en la imagen: el contenedor solo publica archivos.

Para construirla y levantarla en local:

```powershell
docker build -t tpose-server .
docker run --rm -p 8080:8080 tpose-server
```

## Despliegue en Cloud Run

El despliegue se hace desde la raíz del repositorio, que es donde está el `Dockerfile`. La región usada es `southamerica-east1` y el proyecto es `ilsa-511622`.

```powershell
gcloud run deploy tpose-server --source . --region southamerica-east1 --port 8080 --allow-unauthenticated --project ilsa-511622
```

Ese comando sube el código, construye la imagen y publica el servicio `tpose-server`. `--allow-unauthenticated` deja las glosas accesibles para el cliente del navegador. `--port 8080` le dice a Cloud Run en qué puerto escucha el contenedor. El puerto real del proceso sigue saliendo de `PORT`.

Los nombres de `poses/` tienen que ser ASCII. Un archivo con `Ñ` o con tilde en el nombre hace fallar la importación del contenedor en Cloud Run, aunque el build termine bien. Por eso la letra `Ñ` está en `ENIE.pose` y el resto de las eñes del nombre quedó escrito con `N`.

Si `gcloud` tiene guardado el número de proyecto en lugar del ID, el comando necesita `--project ilsa-511622`, o hay que dejar el ID configurado:

```powershell
gcloud config set project ilsa-511622
```

## Qué más hay en la rama

`src/` es el código anterior de captura y clasificación de señas. No lo usa este servicio y no se copia al contenedor.

- `src/camera.py` abre la cámara, toma landmarks con MediaPipe y clasifica la seña con un modelo pequeño.
- `src/preprocessing.py` convierte videos del dataset en secuencias de landmarks.
- `src/utils.py` normaliza la pose respecto de los hombros y deja todas las secuencias en la misma cantidad de frames.
- `src/config.py` define las clases de seña de ese modelo, el tamaño de frame y los parámetros de entrenamiento.

Esas clases no son el índice de `poses/`. Para pedir una animación al avatar se usa el nombre del archivo en `poses/`, con el formato descrito arriba.
