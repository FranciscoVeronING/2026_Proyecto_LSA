# Expone LSA Meet por HTTPS en la tailnet (Tailscale Serve).
# Requisitos: MagicDNS + HTTPS activados en https://login.tailscale.com/admin/dns
# Backend en :8000 y frontend (npm run dev) en :3000 deben estar corriendo ANTES.

$ts = "C:\Program Files\Tailscale\tailscale.exe"
if (-not (Test-Path $ts)) {
    Write-Error "No se encontró Tailscale en $ts"
    exit 1
}

Write-Host "[*] Comprobando backend local..."
try {
    $health = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/health" -TimeoutSec 3
    Write-Host "[+] Backend OK: $($health.status)"
} catch {
    Write-Error "Backend no responde en http://127.0.0.1:8000 — ejecutá: python run_server.py"
    exit 1
}

Write-Host "[*] Comprobando frontend local..."
try {
    Invoke-WebRequest -Uri "http://127.0.0.1:3000" -TimeoutSec 3 -UseBasicParsing | Out-Null
    Write-Host "[+] Frontend OK en :3000"
} catch {
    Write-Error "Frontend no responde en http://127.0.0.1:3000 — ejecutá: cd web && npm run dev"
    exit 1
}

Write-Host "[*] Configurando Tailscale Serve..."
& $ts serve reset
# Frontend en la raíz HTTPS
& $ts serve --bg http://127.0.0.1:3000
# WebSocket: el target termina en /ws para que /ws/ROOM llegue al backend como /ws/ROOM
& $ts serve --bg --set-path /ws http://127.0.0.1:8000/ws

Write-Host ""
Write-Host "=== Listo ==="
& $ts serve status
Write-Host ""
Write-Host "Abrí en Chrome la URL https://...ts.net de arriba."
Write-Host "Probá también: https://...ts.net/api/health"
Write-Host "WebSocket: wss://...ts.net/ws/CODIGO-SALA (Tailscale reenvía sin prefijo /ws al backend)"
Write-Host ""
Write-Host "En web/.env.local dejá NEXT_PUBLIC_API_URL vacío (o comentado)."
