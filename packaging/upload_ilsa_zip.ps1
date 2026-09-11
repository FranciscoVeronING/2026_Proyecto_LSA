# Sube extension\bin\ILSA.zip al mismo GitHub Release que el GGUF (tag ilsa-llama-1b).
# GitHub admite como máximo 2 GB por archivo. El zip NO debe incluir el .gguf.
# Uso (después de packaging\build_exe.bat):
#   powershell -File packaging\upload_ilsa_zip.ps1

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
if (-not $repoRoot) { $repoRoot = (Get-Location).Path }
$repo = "FranciscoVeronING/2026_Proyecto_LSA"
$tag = "ilsa-llama-1b"
$zip = Join-Path $repoRoot "extension\bin\ILSA.zip"

if (-not (Test-Path $zip)) {
  Write-Error "No está extension\bin\ILSA.zip. Primero: packaging\build_exe.bat"
}

$zi = Get-Item $zip
$gb = [math]::Round($zi.Length / 1GB, 2)
Write-Host "Zip: $zip ($gb GB)"
if ($zi.Length -gt 2GB) {
  Write-Error "GitHub no acepta archivos de más de 2 GB. Este zip pesa $gb GB. Regeneralo con packaging\build_exe.bat (sin GGUF)."
}

$gh = Get-Command gh -ErrorAction SilentlyContinue
if (-not $gh) {
  $ghExe = "C:\Program Files\GitHub CLI\gh.exe"
  if (Test-Path $ghExe) { $ghBin = $ghExe } else { $ghBin = $null }
} else {
  $ghBin = $gh.Source
}
if (-not $ghBin) {
  Write-Error "Falta GitHub CLI (gh)."
}

$prevEap = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $ghBin release view $tag --repo $repo 2>$null | Out-Null
$exists = ($LASTEXITCODE -eq 0)
$ErrorActionPreference = $prevEap

if (-not $exists) {
  Write-Error "No existe el release $tag. Subí antes el GGUF: packaging\upload_llama_gguf.ps1"
}

Write-Host "Subiendo ILSA.zip a $tag ..."
& $ghBin release upload $tag $zip --repo $repo --clobber
if ($LASTEXITCODE -ne 0) {
  Write-Error "gh falló al subir ILSA.zip"
}

Write-Host "Listo. El botón de la extensión baja:"
Write-Host "https://github.com/$repo/releases/download/$tag/ILSA.zip"
