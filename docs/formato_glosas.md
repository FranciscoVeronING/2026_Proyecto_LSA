# Formato de glosas

En el JSON los nombres y el DNI siguen letra a letra. Al entrenar se compactan.

| En el JSON | Target |
|------------|--------|
| `YO NOMBRE M A I T E` | `YO NOMBRE Maite` |
| `YO DOCUMENTO NUMERO 1 8 3 3 3 9 4 0` | `YO DOCUMENTO NUMERO 18333940` |
| `X` | `X` |

Una oración en español tiene una sola lista de glosas.

- Posesivo detrás del sustantivo: `CUCHILLO MIO`, `HIJO HOMBRE MIO`
- hijo/hermano/esposo → `HOMBRE`; hija/hermana/esposa → `MUJER`
- `Ellos` → `ELLOS HOMBRE`; `Ellas` → `ELLOS MUJER`; sujeto tácito sin marca
- `calle` no se traduce como `CASA`

```powershell
cd src\semantic
python test_gloss_format.py
```
