# Sube el GGUF de Llama 3.2 1B a GitHub Releases (tag ilsa-llama-1b).
# Requisitos: gh autenticado (`gh auth login`) y el .gguf en disco.
# Uso (desde la raíz del repo):
#   powershell -File packaging\upload_llama_gguf.ps1

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
if (-not $repoRoot) { $repoRoot = (Get-Location).Path }
$repo = "FranciscoVeronING/2026_Proyecto_LSA"
$tag = "ilsa-llama-1b"
$asset = "llama-3.2-1b-instruct.Q4_K_M.gguf"
$search = @(
  (Join-Path $repoRoot "src\semantic\outputs\unsloth_Llama-3.2-1B-Instruct_gguf\$asset"),
  (Join-Path $env:LOCALAPPDATA "ILSA\models\$asset")
)
$gguf = $search | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $gguf) {
  Write-Error "No está $asset. Esperaba src\semantic\outputs\unsloth_Llama-3.2-1B-Instruct_gguf\"
}

$gh = Get-Command gh -ErrorAction SilentlyContinue
if (-not $gh) {
  $ghExe = "C:\Program Files\GitHub CLI\gh.exe"
  if (Test-Path $ghExe) { $ghBin = $ghExe } else { $ghBin = $null }
} else {
  $ghBin = $gh.Source
}
if (-not $ghBin) {
  Write-Error "Falta GitHub CLI (gh). Instalá con winget install GitHub.cli y hacé gh auth login"
}

Write-Host "Subiendo $gguf"
Write-Host "  tag    $tag"
Write-Host "  repo   $repo"

$prevEap = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $ghBin release view $tag --repo $repo 2>$null | Out-Null
$exists = ($LASTEXITCODE -eq 0)
$ErrorActionPreference = $prevEap

if (-not $exists) {
  & $ghBin release create $tag $gguf `
    --repo $repo `
    --title "Traductor Llama 1B (ILSA)" `
    --notes "GGUF Q4 de Llama 3.2 1B para ILSA. Lo baja el exe la primera vez que se abre." `
    --prerelease
} else {
  & $ghBin release upload $tag $gguf --repo $repo --clobber
}

if ($LASTEXITCODE -ne 0) {
  Write-Error "gh falló. ¿Estás autenticado? gh auth login"
}

Write-Host "Listo. URL:"
Write-Host "https://github.com/FranciscoVeronING/2026_Proyecto_LSA/releases/download/$tag/$asset"
