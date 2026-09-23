# Sube packaging\semantic_url.txt al release ilsa-llama-1b.
# ILSA.exe lo baja solo al arrancar (no hace falta set LSA_SEMANTIC_URL).
#
#   powershell -File packaging\upload_semantic_url.ps1
#   powershell -File packaging\upload_semantic_url.ps1 -Url https://otra.ngrok-free.dev

param(
  [string]$Url = ""
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
if (-not $repoRoot) { $repoRoot = (Get-Location).Path }
$repo = "FranciscoVeronING/2026_Proyecto_LSA"
$tag = "ilsa-llama-1b"
$file = Join-Path $PSScriptRoot "semantic_url.txt"

if ($Url) {
  [System.IO.File]::WriteAllText($file, ($Url.Trim() + "`n"))
}

if (-not (Test-Path $file)) {
  Write-Error "No está packaging\semantic_url.txt"
}

$line = (Get-Content $file | Where-Object { $_.Trim() -and -not $_.Trim().StartsWith("#") } | Select-Object -First 1)
if (-not $line) {
  Write-Error "semantic_url.txt está vacío"
}
Write-Host "URL: $($line.Trim())"

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
  Write-Error "No existe el release $tag."
}

Write-Host "Subiendo semantic_url.txt a $tag ..."
& $ghBin release upload $tag $file --repo $repo --clobber
if ($LASTEXITCODE -ne 0) {
  Write-Error "gh falló al subir semantic_url.txt"
}
Write-Host "https://github.com/$repo/releases/download/$tag/semantic_url.txt"
