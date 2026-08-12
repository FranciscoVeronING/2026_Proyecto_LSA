# LSA Meet — arranque con Cloudflare Tunnel

Expone el frontend (Next.js :3000) y el backend (FastAPI :8000) bajo un solo hostname HTTPS.

## Requisitos

- [cloudflared](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/) instalado
- GPU NVIDIA con drivers CUDA (para clasificador + LLM)
- Node.js 20+ y Python 3.9+ con el entorno `lsa_gpu`

## 1. Backend

```bash
conda activate lsa_gpu
pip install -r requirements.txt
python run_server.py
```

Verificar: http://localhost:8000/api/health

## 2. Frontend

```bash
cd web
npm install          # copia MediaPipe a public/mediapipe/holistic (postinstall)
cp .env.local.example .env.local
npm run dev
```

Si MediaPipe falla al cargar, ejecutá manualmente: `node scripts/copy-mediapipe.mjs`

Verificar: http://localhost:3000

## 3. Túnel (demo entre dos máquinas)

Opción rápida (URL temporal):

```bash
# Terminal A — backend
cloudflared tunnel --url http://localhost:8000

# Terminal B — frontend (ajustar .env.local con la URL del túnel del backend)
# NEXT_PUBLIC_API_URL=https://xxxx.trycloudflare.com
# NEXT_PUBLIC_WS_URL=wss://xxxx.trycloudflare.com/ws
cloudflared tunnel --url http://localhost:3000
```

Opción con config fija ([cloudflared.yml](cloudflared.yml)):

```bash
cloudflared tunnel --config cloudflared.yml run
```

## 4. Flujo de uso

1. Usuario A crea una sala en la home → comparte el código/URL
2. Usuario B se une con el código
3. Cada uno completa: nombre → ¿señante? → mano hábil (si aplica)
4. Entran a la llamada: video P2P + interpretación LSA en el chat

## Validación de landmarks

Antes de la demo final, comparar precisión del navegador vs OpenCV:

```bash
python run.py --eval
# Señar las mismas señas en la webapp y comparar top-1/top-3
```

Si la caída es grande, activar la ruta de respaldo enviando frames JPEG (`type: "frame"` en el WebSocket) en lugar de landmarks.

## Notas

- Usar **Chrome** en ambas máquinas (MediaPipe JS + SpeechRecognition)
- Auriculares recomendados para evitar que el STT transcriba la voz del otro
- WebRTC usa STUN público; si falla la conexión de video, puede hacer falta un servidor TURN
