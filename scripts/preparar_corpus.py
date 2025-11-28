import os
import re
import fitz  # PyMuPDF
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch
from reportlab.lib import colors



# === CONFIGURACIÓN DE RUTAS ===
BASEPATH = r"C:\Users\felipe\Documents\Universidad\Inteligentes1\Documentos ChatBot IA"
PDFFOLDER = os.path.join(BASEPATH, "datasets", "corpus_pdfs")
OUTPUTFOLDER = os.path.join(BASEPATH, "datasets", "corpus_pdf_estructurado_v2")
METADATAFILE = os.path.join(BASEPATH, "datasets", "metadata_pdf_v2.json")

os.makedirs(PDFFOLDER, exist_ok=True)
os.makedirs(OUTPUTFOLDER, exist_ok=True)

import unicodedata
import re

def limpiar_texto(texto):
    # Elimina caracteres de control
    texto = re.sub(r"[\x00-\x1F\x7F-\x9F]", "", texto)
    # Quita tildes y acentos
    texto = unicodedata.normalize('NFD', texto)
    texto = texto.encode('ascii', 'ignore').decode('utf-8')
    # Opcional: eliminar otros símbolos extraños
    texto = re.sub(r'[^\x20-\x7E]', '', texto)
    return texto


def extraer_texto_pdf(pdf_path):
    """
    Extrae el texto completo del PDF usando PyMuPDF y lo limpia de tildes/símbolos raros.
    """
    texto = ""
    doc = fitz.open(pdf_path)
    for pagina in doc:
        # Extrae y limpia
        pag_texto = pagina.get_text()
        pag_texto = limpiar_texto(pag_texto)
        texto += pag_texto
    return texto

def estructurar_texto(text):
    """
    Detecta títulos y subtítulos y los marca con encabezados Markdown (#, ##).
    Retorna una lista de tuplas (tipo, contenido), limpiando tildes y caracteres.
    """
    lines = text.split("\n")
    bullet_pattern = r'^\s*(?:[*-•]|[0-9]{1,2}[\.\)-]|[a-zA-Z][\.\)-])\s+.*:\s*$'
    resultado = []
    for line in lines:
        clean = limpiar_texto(line.strip())
        if not clean:
            continue
        if re.match(r"^(Capitulo|Chapter|Seccion|[0-9]+\.)", clean, re.IGNORECASE):
            resultado.append(("h1", f"# {clean}"))
        elif re.match(bullet_pattern, clean):
            resultado.append(("h2", f"## {clean}"))
        elif len(clean.split()) <= 6 and clean.isupper():
            resultado.append(("h2", f"## {clean}"))
        else:
            resultado.append(("p", clean))
    return resultado

def crear_pdf_estructurado(estructura, output_path):
    estilos = {
        "h1": ParagraphStyle(
            "h1",
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#003366"),
            fontName="Helvetica-Bold",
            spaceBefore=12,
            spaceAfter=10,
        ),
        "h2": ParagraphStyle(
            "h2",
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#006633"),
            fontName="Helvetica-Bold",
            spaceBefore=10,
            spaceAfter=6,
        ),
        "p": ParagraphStyle(
            "p", fontSize=11, leading=14, alignment=4, spaceBefore=4, spaceAfter=4
        ),
    }

    doc = SimpleDocTemplate(output_path, pagesize=LETTER)
    story = []
    for tipo, contenido in estructura:
        estilo = estilos.get(tipo, estilos["p"])
        if tipo in ["h1", "h2"]:
            contenido_limpio = contenido.strip()
        else:
            contenido_limpio = contenido

        story.append(Paragraph(contenido_limpio, estilo))
        story.append(Spacer(1, 0.15 * inch))
    doc.build(story)

def procesar_pdfs():
    pdf_files = [f for f in os.listdir(PDFFOLDER) if f.endswith(".pdf")]
    print(f"🔍 Se encontraron {len(pdf_files)} PDFs. Iniciando conversión...")

    for pdf_file in pdf_files:
        input_path = os.path.join(PDFFOLDER, pdf_file)
        print(f"Procesando {pdf_file}...")
        texto_extraido = extraer_texto_pdf(input_path)
        estructura = estructurar_texto(texto_extraido)

        output_pdf_path = os.path.join(
            OUTPUTFOLDER, f"estructurado_{pdf_file.replace('.pdf', '.pdf')}"
        )
        crear_pdf_estructurado(estructura, output_pdf_path)
        print(f"PDF estructurado creado en {output_pdf_path}")

if __name__ == "__main__":
    procesar_pdfs()

