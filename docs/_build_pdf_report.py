#!/usr/bin/env python3
"""Convierte entregables_mvp_modulo_semantico.md a PDF."""
from __future__ import annotations

import re
from pathlib import Path

import markdown
from xhtml2pdf import pisa

ROOT = Path(__file__).resolve().parent
MD_PATH = ROOT / "entregables_mvp_modulo_semantico.md"
PDF_PATH = ROOT / "Entregables_MVP_Modulo_Semantico.pdf"

CSS = """
@page {
    size: A4;
    margin: 2cm 2.2cm 2.2cm 2.2cm;
    @frame footer {
        -pdf-frame-content: footerContent;
        bottom: 0.8cm;
        margin-left: 2.2cm;
        margin-right: 2.2cm;
        height: 1cm;
    }
}
body {
    font-family: DejaVuSans, Helvetica, Arial, sans-serif;
    font-size: 10.5pt;
    line-height: 1.45;
    color: #1a1a1a;
}
h1 {
    font-size: 20pt;
    color: #0f3d5c;
    border-bottom: 2px solid #0f3d5c;
    padding-bottom: 6px;
    margin-top: 0;
}
h2 {
    font-size: 14pt;
    color: #0f3d5c;
    margin-top: 18px;
    border-bottom: 1px solid #cbd5e1;
    padding-bottom: 4px;
}
h3 {
    font-size: 11.5pt;
    color: #1e4a66;
    margin-top: 14px;
}
p, li {
    text-align: justify;
}
code, pre {
    font-family: DejaVuSansMono, Courier, monospace;
    font-size: 8.5pt;
    background-color: #f4f6f8;
}
pre {
    padding: 8px;
    border: 1px solid #dde3ea;
    white-space: pre-wrap;
    word-wrap: break-word;
}
table {
    width: 100%;
    border-collapse: collapse;
    margin: 10px 0 14px 0;
    font-size: 9.5pt;
}
th {
    background-color: #0f3d5c;
    color: white;
    padding: 6px 8px;
    text-align: left;
}
td {
    border: 1px solid #cbd5e1;
    padding: 5px 8px;
    vertical-align: top;
}
tr:nth-child(even) td {
    background-color: #f8fafc;
}
blockquote {
    border-left: 3px solid #0f3d5c;
    margin: 10px 0;
    padding: 6px 12px;
    background: #f8fafc;
    color: #334155;
}
hr {
    border: none;
    border-top: 1px solid #cbd5e1;
    margin: 16px 0;
}
a {
    color: #0f3d5c;
    text-decoration: none;
}
.footer {
    font-size: 8pt;
    color: #64748b;
    text-align: center;
}
.diagram {
    border: 1px solid #cbd5e1;
    background: #f8fafc;
    padding: 10px;
    font-size: 9pt;
    margin: 10px 0;
}
"""


def preprocess_markdown(text: str) -> str:
    text = re.sub(
        r"```mermaid\n.*?```",
        (
            "\n> **Diagrama de arquitectura:** Clasificador → Glosas → "
            "(System prompt + few-shot + LLM fine-tuned) → Oración en español.\n"
        ),
        text,
        flags=re.DOTALL,
    )
    return text


def md_to_html(md_text: str) -> str:
    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "nl2br", "sane_lists"],
    )
    return f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="utf-8"/>
    <title>Entregables MVP - Módulo Semántico</title>
    <style>
    @font-face {{
        font-family: DejaVuSans;
        src: url("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf");
    }}
    @font-face {{
        font-family: DejaVuSansMono;
        src: url("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf");
    }}
    {CSS}
    </style>
</head>
<body>
{html_body}
<div id="footerContent" class="footer">
    Proyecto LSA 2026 | Módulo Semántico | MVP | Rama modulo-semantico
</div>
</body>
</html>"""


def html_to_pdf(html: str, pdf_path: Path) -> None:
    with open(pdf_path, "wb") as pdf_file:
        status = pisa.CreatePDF(html, dest=pdf_file, encoding="utf-8")
    if status.err:
        raise RuntimeError(f"Error generando PDF: {status.err}")


def main() -> None:
    md_text = preprocess_markdown(MD_PATH.read_text(encoding="utf-8"))
    html = md_to_html(md_text)
    html_to_pdf(html, PDF_PATH)
    print(f"PDF generado: {PDF_PATH}")


if __name__ == "__main__":
    main()
