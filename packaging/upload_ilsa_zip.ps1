# Upload extension\bin\ILSA.zip to the GitHub Release (tag ilsa-llama-1b).
# GitHub allows at most 2 GB per file.
# Usage (after packaging\build_exe.bat):
#   powershell -File packaging\upload_ilsa_zip.ps1

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
if (-not $repoRoot) { $repoRoot = (Get-Location).Path }
$repo = "FranciscoVeronING/2026_Proyecto_LSA"
$tag = "ilsa-llama-1b"
$zip = Join-Path $repoRoot "extension\bin\ILSA.zip"

if (-not (Test-Path $zip)) {
  Write-Error "extension\bin\ILSA.zip is missing. First: packaging\build_exe.bat"
}

$zi = Get-Item $zip
$gb = [math]::Round($zi.Length / 1GB, 2)
Write-Host "Zip: $zip ($gb GB)"
if ($zi.Length -gt 2GB) {
  Write-Error "GitHub does not accept files over 2 GB. This zip is $gb GB. Rebuild with packaging\build_exe.bat."
}

$gh = Get-Command gh -ErrorAction SilentlyContinue
if (-not $gh) {
  $ghExe = "C:\Program Files\GitHub CLI\gh.exe"
  if (Test-Path $ghExe) { $ghBin = $ghExe } else { $ghBin = $null }
} else {
  $ghBin = $gh.Source
}
if (-not $ghBin) {
  Write-Error "GitHub CLI (gh) is missing."
}

$prevEap = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $ghBin release view $tag --repo $repo 2>$null | Out-Null
$exists = ($LASTEXITCODE -eq 0)
$ErrorActionPreference = $prevEap

if (-not $exists) {
  Write-Host "Creating release $tag ..."
  & $ghBin release create $tag $zip --repo $repo --title "ILSA" --notes "ILSA engine (CPU classifier). Translates against Cloud Run."
  if ($LASTEXITCODE -ne 0) {
    Write-Error "gh failed to create the release"
  }
} else {
  Write-Host "Uploading ILSA.zip to $tag ..."
  & $ghBin release upload $tag $zip --repo $repo --clobber
  if ($LASTEXITCODE -ne 0) {
    Write-Error "gh failed to upload ILSA.zip"
  }
}

Write-Host "Done. The extension button downloads:"
Write-Host "https://github.com/$repo/releases/download/$tag/ILSA.zip"
