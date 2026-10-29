"""Voz en off en WAV, con la síntesis de Windows (System.Speech).

`speechSynthesis` del navegador no se puede mezclar con la pista de Meet.
Este WAV sí: la extensión lo suma al audio que sale a la llamada.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

_PS = r"""
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$picked = $false
foreach ($voice in $synth.GetInstalledVoices()) {
  if (-not $voice.Enabled) { continue }
  $culture = $voice.VoiceInfo.Culture.Name
  if ($culture -like 'es*') {
    $synth.SelectVoice($voice.VoiceInfo.Name)
    $picked = $true
    break
  }
}
$text = [System.IO.File]::ReadAllText($env:LSA_TTS_TEXT, [System.Text.Encoding]::UTF8)
$synth.SetOutputToWaveFile($env:LSA_TTS_WAV)
$synth.Speak($text)
$synth.Dispose()
"""


def synthesize_wav(text: str) -> bytes:
    """Devuelve un WAV PCM. Falla si Windows no puede sintetizar."""
    cleaned = " ".join((text or "").split())
    if not cleaned:
        raise ValueError("No hay texto para leer.")
    if len(cleaned) > 500:
        cleaned = cleaned[:500]
    with tempfile.TemporaryDirectory(prefix="lsa-tts-") as folder:
        root = Path(folder)
        src = root / "line.txt"
        wav = root / "out.wav"
        script = root / "speak.ps1"
        src.write_text(cleaned, encoding="utf-8")
        script.write_text(_PS, encoding="utf-8")
        try:
            done = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-NonInteractive",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(script),
                ],
                check=False,
                capture_output=True,
                timeout=25,
                env={**os.environ, "LSA_TTS_TEXT": str(src), "LSA_TTS_WAV": str(wav)},
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("La voz en off tardó demasiado.") from exc
        if done.returncode != 0 or not wav.is_file() or wav.stat().st_size < 44:
            err = (done.stderr or done.stdout or b"").decode("utf-8", "replace").strip()
            raise RuntimeError(err or "Windows no pudo generar la voz.")
        return wav.read_bytes()
