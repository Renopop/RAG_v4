# docx_processing.py
import docx
import re

def docx_to_text(docx_path):
    """
    Extrait le texte brut d'un fichier .docx en conservant les sauts de ligne entre paragraphes.
    """
    doc = docx.Document(docx_path)
    full_text = [para.text for para in doc.paragraphs]
    # On garde un \n entre chaque paragraphe
    return "\n".join(full_text)

def normalize_whitespace(text: str) -> str:
    """
    Normalise les espaces dans le texte EN CONSERVANT les sauts de ligne.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    norm_lines = []
    for line in lines:
        # Réduire les suites d'espaces/tabulations
        line = re.sub(r"[ \t]+", " ", line)
        norm_lines.append(line.strip())
    return "\n".join(norm_lines).strip()

def extract_text_from_docx(docx_path):
    """
    Extrait et nettoie le texte d'un fichier DOCX, compatible avec la détection EASA.
    """
    raw_text = docx_to_text(docx_path)
    return normalize_whitespace(raw_text)

def extract_paragraphs_from_docx(docx_path):
    """
    Extrait les paragraphes individuels d'un fichier DOCX.
    """
    doc = docx.Document(docx_path)
    paragraphs = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
    return paragraphs

def extract_sections_from_docx(docx_path, heading_styles=None):
    """
    Extrait les sections d'un fichier DOCX en se basant sur les styles de titre.
    heading_styles est une liste de noms de styles considérés comme des titres de sections.
    """
    if heading_styles is None:
        heading_styles = ["Heading 1", "Heading 2", "Titre 1", "Titre 2"]

    doc = docx.Document(docx_path)
    sections = []
    current_section_title = None
    current_section_content = []

    for para in doc.paragraphs:
        style_name = para.style.name if para.style else ""

        if style_name in heading_styles:
            # Sauvegarder la section précédente
            if current_section_title or current_section_content:
                sections.append({
                    "title": current_section_title,
                    "content": "\n".join(current_section_content).strip(),
                })
                current_section_content = []

            # Nouveau titre de section
            current_section_title = para.text.strip()
        else:
            # Contenu de la section en cours
            if para.text.strip():
                current_section_content.append(para.text.strip())

    # Ajouter la dernière section si présente
    if current_section_title or current_section_content:
        sections.append({
            "title": current_section_title,
            "content": "\n".join(current_section_content).strip(),
        })

    return sections

def extract_text_from_tables(docx_path):
    """
    Extrait le texte contenu dans les tableaux d'un fichier DOCX.
    """
    doc = docx.Document(docx_path)
    table_texts = []

    for table in doc.tables:
        for row in table.rows:
            row_text = []
            for cell in row.cells:
                cell_text = cell.text.strip()
                if cell_text:
                    row_text.append(cell_text)
            if row_text:
                table_texts.append(" | ".join(row_text))

    return "\n".join(table_texts)
