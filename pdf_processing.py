import io
import os
import re
import logging
import traceback
from typing import Optional, Callable, List, Dict, Any

from unidecode import unidecode
import chardet  # conservé pour compatibilité / heuristiques

from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFPasswordIncorrect
import pdfminer

from packaging import version

# Pour l'extraction via PyMuPDF
try:
    import pymupdf as fitz  # PyMuPDF / pymupdf
    logging.info("PyMuPDF importé avec succès")
except ImportError as e:
    fitz = None
    logging.error(f"Erreur lors de l'importation de PyMuPDF : {e}")

# Pour l'extraction améliorée des tableaux
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
    logging.info("pdfplumber importé avec succès")
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logging.warning("pdfplumber non disponible - extraction des tableaux basique")

# -----------------------------------------------------------------------------
# Types
# -----------------------------------------------------------------------------

ProgressFn = Callable[[float, str], None]

# -----------------------------------------------------------------------------
# Logging de base
# -----------------------------------------------------------------------------

logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")

# -----------------------------------------------------------------------------
# Utilitaires génériques
# -----------------------------------------------------------------------------

def clean_filename(filename: str) -> str:
    """
    Nettoie un nom de fichier pour le système de fichiers :
      - suppression des caractères surrogates (Unicode invalides)
      - translittération unicode -> ASCII
      - suppression/remplacement des caractères spéciaux
      - PRÉSERVE l'extension du fichier (.pdf, .docx, etc.)
    """
    # Séparer le nom de base et l'extension
    name_part, ext_part = os.path.splitext(filename)

    # Si pas de nom (juste extension), utiliser un nom par défaut
    if not name_part.strip():
        name_part = "attachment"

    # Supprimer les surrogates et caractères invalides du nom
    try:
        # Nettoyer les surrogates en les ignorant
        clean = name_part.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
        if not clean.strip():
            clean = "attachment"
    except Exception:
        clean = "attachment"

    # Translittération Unicode -> ASCII (seulement pour le nom)
    safe_name = unidecode(clean)
    safe_name = re.sub(r"[^\w\s-]", "_", safe_name)
    safe_name = re.sub(r"\s+", "_", safe_name)
    safe_name = re.sub(r"_{2,}", "_", safe_name)
    safe_name = safe_name.strip("_")

    # Nettoyer l'extension aussi (mais garder le point)
    if ext_part:
        # Nettoyer l'extension des surrogates
        try:
            clean_ext = ext_part.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
        except Exception:
            clean_ext = ext_part
        ext_part = clean_ext.lower()  # Extensions en minuscules

    # Recombiner nom + extension
    final_name = (safe_name or "attachment") + ext_part
    return final_name


def detect_and_clean_surrogates(data: bytes) -> bytes:
    """
    Tente de décoder puis ré-encoder des données binaires texte en éliminant
    les surrogates / caractères invalides.
    (Non utilisée dans l'orchestrateur, mais dispo si tu veux traiter des PJ texte.)
    """
    encodings_to_try = ["utf-8", "utf-16", "latin-1", "iso-8859-1", "cp1252"]

    # Tentative de détection via chardet (optionnel)
    try:
        guess = chardet.detect(data)
        if guess and guess.get("encoding"):
            encodings_to_try.insert(0, guess["encoding"])
    except Exception:
        pass

    seen = set()
    ordered_encodings: List[str] = []
    for enc in encodings_to_try:
        if enc and enc.lower() not in seen:
            ordered_encodings.append(enc)
            seen.add(enc.lower())

    for encoding in ordered_encodings:
        try:
            text = data.decode(encoding, errors="replace")
            logging.info(f"[pdf_processing] Décodage pièce jointe réussi avec {encoding}")
            return text.encode(encoding, errors="replace")
        except UnicodeDecodeError:
            logging.debug(f"[pdf_processing] Échec du décodage avec {encoding}")

    logging.warning("[pdf_processing] Aucun encodage concluant, fallback latin-1")
    return data.decode("latin-1", errors="replace").encode("latin-1", errors="replace")

# -----------------------------------------------------------------------------
# Fallback PyMuPDF pour l'extraction texte
# -----------------------------------------------------------------------------

def _extract_text_with_pymupdf(path: str) -> str:
    """
    Fallback d'extraction texte via PyMuPDF si pdfminer échoue.
    """
    if fitz is None:
        raise RuntimeError("PyMuPDF non installé, impossible d'utiliser le fallback.")

    logging.info(f"[pdf_processing] Fallback PyMuPDF pour {path}")
    doc = fitz.open(path)
    try:
        pages_text: List[str] = []
        for page in doc:
            try:
                page_text = page.get_text("text")
            except TypeError:
                # compat anciennes versions
                page_text = page.get_text()
            pages_text.append(page_text)
        return "\n".join(pages_text)
    finally:
        doc.close()


def _extract_with_pdfplumber(path: str) -> str:
    """
    Extraction PDF avec pdfplumber - meilleure gestion des tableaux.
    Extrait le texte normal ET formate les tableaux en markdown.
    """
    if not PDFPLUMBER_AVAILABLE:
        raise RuntimeError("pdfplumber non disponible")

    logging.info(f"[pdf_processing] Extraction pdfplumber (tableaux) pour {path}")
    all_content: List[str] = []

    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            page_content: List[str] = []

            # Détecter les tableaux sur la page
            tables = page.extract_tables()

            if tables:
                # Page avec tableaux - extraire texte hors tableaux + tableaux formatés
                # Récupérer les bounding boxes des tableaux
                table_bboxes = []
                for table in page.find_tables():
                    table_bboxes.append(table.bbox)

                # Extraire le texte en excluant les zones de tableaux
                text_outside_tables = page.filter(
                    lambda obj: not any(
                        _is_inside_bbox(obj, bbox) for bbox in table_bboxes
                    ) if hasattr(obj, 'x0') else True
                ).extract_text() or ""

                if text_outside_tables.strip():
                    page_content.append(text_outside_tables.strip())

                # Formater chaque tableau en markdown
                for idx, table in enumerate(tables):
                    if table and len(table) > 0:
                        formatted_table = _format_table_as_markdown(table)
                        if formatted_table:
                            page_content.append(f"\n[TABLEAU {idx + 1}]\n{formatted_table}")
            else:
                # Pas de tableau - extraction normale
                text = page.extract_text() or ""
                if text.strip():
                    page_content.append(text.strip())

            if page_content:
                all_content.append("\n".join(page_content))

    result = "\n\n".join(all_content)
    logging.info(f"[pdf_processing] pdfplumber: {len(result)} caractères extraits")
    return result


def _is_inside_bbox(obj, bbox) -> bool:
    """Vérifie si un objet PDF est à l'intérieur d'une bounding box."""
    if not hasattr(obj, 'x0') or not hasattr(obj, 'top'):
        return False
    x0, top, x1, bottom = bbox
    return (obj.x0 >= x0 and getattr(obj, 'x1', obj.x0) <= x1 and
            obj.top >= top and getattr(obj, 'bottom', obj.top) <= bottom)


def _format_table_as_markdown(table: List[List]) -> str:
    """
    Formate un tableau extrait en format markdown/texte lisible.
    """
    if not table or len(table) == 0:
        return ""

    # Nettoyer les cellules (remplacer None par "", nettoyer les espaces)
    cleaned_table = []
    for row in table:
        if row:
            cleaned_row = []
            for cell in row:
                if cell is None:
                    cleaned_row.append("")
                else:
                    # Nettoyer les retours à la ligne dans les cellules
                    cleaned_row.append(str(cell).replace("\n", " ").strip())
            cleaned_table.append(cleaned_row)

    if not cleaned_table:
        return ""

    # Calculer la largeur max de chaque colonne
    num_cols = max(len(row) for row in cleaned_table)
    col_widths = [0] * num_cols

    for row in cleaned_table:
        for i, cell in enumerate(row):
            if i < num_cols:
                col_widths[i] = max(col_widths[i], len(cell))

    # Formater en texte tabulé
    lines = []
    for row_idx, row in enumerate(cleaned_table):
        # Compléter les colonnes manquantes
        while len(row) < num_cols:
            row.append("")

        # Formater la ligne
        formatted_cells = []
        for i, cell in enumerate(row):
            formatted_cells.append(cell.ljust(col_widths[i]))
        line = " | ".join(formatted_cells)
        lines.append(line)

        # Ajouter une ligne de séparation après l'en-tête (première ligne)
        if row_idx == 0:
            separator = "-+-".join("-" * w for w in col_widths)
            lines.append(separator)

    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Heuristiques de qualité du texte
# -----------------------------------------------------------------------------

def _is_text_suspect(text: str, min_chars: int = 500, min_ratio_printable: float = 0.85) -> bool:
    """
    Détecte un texte "suspect" : trop court, rempli de caractères non imprimables, etc.
    Permet de décider si on tente un fallback (PyMuPDF) même si pdfminer n'a pas levé d'erreur.
    """
    if not text:
        return True

    # Trop peu de texte -> suspect
    if len(text) < min_chars:
        logging.warning(f"[pdf_processing] Texte très court ({len(text)} caractères) — suspect.")
        return True

    # Ratio de caractères imprimables
    printable = sum(1 for c in text if c.isprintable() or c in "\n\r\t")
    total = len(text)
    if total == 0:
        return True
    ratio = printable / total
    if ratio < min_ratio_printable:
        logging.warning(
            f"[pdf_processing] Texte avec beaucoup de caractères non imprimables "
            f"(ratio imprimables={ratio:.2f}) — suspect."
        )
        return True

    return False


def _split_pages(text: str) -> List[str]:
    """
    Sépare le texte en pages.
    pdfminer insère généralement un caractère form feed '\f' entre les pages.
    Si aucun form feed n'est trouvé, on considère qu'il n'y a qu'une page.
    """
    if "\f" in text:
        pages = text.split("\f")
    else:
        pages = [text]
    return [p for p in pages]


def _remove_repeated_headers_footers(
    pages: List[str],
    min_pages: int = 3,
    min_occ_ratio: float = 0.7
) -> List[str]:
    """
    Supprime les entêtes / pieds de page répétés sur la majorité des pages.
    """
    if len(pages) < min_pages:
        return pages

    first_lines: List[str] = []
    last_lines: List[str] = []

    for p in pages:
        lines = [l.rstrip() for l in p.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
        lines = [l for l in lines if l.strip() != ""]
        if not lines:
            first_lines.append("")
            last_lines.append("")
            continue
        first_lines.append(lines[0].strip())
        last_lines.append(lines[-1].strip())

    def _find_repeated(candidates: List[str]) -> List[str]:
        freq: Dict[str, int] = {}
        for l in candidates:
            if not l:
                continue
            freq[l] = freq.get(l, 0) + 1

        res: List[str] = []
        n_pages = len(pages)
        for line, count in freq.items():
            if count >= int(min_occ_ratio * n_pages):
                if len(line) > 2:
                    res.append(line)
        return res

    repeated_first = _find_repeated(first_lines)
    repeated_last = _find_repeated(last_lines)

    if repeated_first:
        logging.info(f"[pdf_processing] Headers répétés détectés et supprimés : {repeated_first}")
    if repeated_last:
        logging.info(f"[pdf_processing] Footers répétés détectés et supprimés : {repeated_last}")

    new_pages: List[str] = []

    for p in pages:
        lines = p.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        cleaned: List[str] = []
        for idx, raw in enumerate(lines):
            line = raw.strip()
            if idx == 0 and line in repeated_first:
                continue
            if idx == len(lines) - 1 and line in repeated_last:
                continue
            cleaned.append(raw)
        new_pages.append("\n".join(cleaned))

    return new_pages

# -----------------------------------------------------------------------------
# Nettoyage du bruit / watermarks PDF récurrents
# -----------------------------------------------------------------------------

NOISE_PATTERNS = [
    r"Country of Export: US EAR 9E991",
    r"Extraterritorial Reach: US EAR 9E991",
    r"Use of this Document Subject to Restrictions on Title Page",
    r"SFT1005/09/14",
    r"CAGE 55820",
    r"FROZEN\s+docum?ent issued from Dassault Aviation repository",
    r"FROZEN\s+document issued from Dassault Aviation repository",
]

NOISE_REGEXES = [re.compile(p, re.IGNORECASE) for p in NOISE_PATTERNS]


def _clean_pdf_text_noise(text: str) -> str:
    """
    Supprime quelques lignes de bruit très répétitives (watermarks, mentions export)
    et certaines lignes ultra-courtes typiques de texte vertical bruité.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    cleaned_lines: List[str] = []

    for raw_line in lines:
        line = raw_line.strip()

        if not line:
            cleaned_lines.append("")
            continue

        # Watermarks / disclaimers connus
        if any(rx.search(line) for rx in NOISE_REGEXES):
            continue

        # Lignes très courtes, majuscules/chiffres, souvent issues de colonnes verticales
        if len(line) <= 3 and re.fullmatch(r"[A-Z0-9\-]+", line):
            continue

        cleaned_lines.append(raw_line)

    return "\n".join(cleaned_lines).strip()

# -----------------------------------------------------------------------------
#  EXTRACTION TEXTE PDF (robuste, utilisée partout)
# -----------------------------------------------------------------------------

def extract_text_from_pdf(
    path: str,
    progress_cb: Optional[ProgressFn] = None,
) -> str:
    """
    Extraction texte robuste depuis un PDF en conservant les sauts de ligne.

    Stratégie :
      1) tentative avec pdfplumber (meilleure gestion des tableaux)
      2) si erreur → fallback pdfminer
      3) si texte suspect → fallback PyMuPDF et choix du meilleur
      4) nettoyage des headers/footers répétés
      5) nettoyage léger du bruit (watermarks récurrents, texte vertical très bruité)
    """
    if progress_cb:
        progress_cb(0.05, f"Ouverture PDF : {os.path.basename(path)}")

    text_final: str = ""
    extraction_method: str = ""

    # 1) Tentative pdfplumber (meilleure gestion des tableaux)
    if PDFPLUMBER_AVAILABLE:
        try:
            text_final = _extract_with_pdfplumber(path)
            extraction_method = "pdfplumber"
            logging.info(f"[pdf_processing] Extraction pdfplumber OK pour {path}")
        except Exception as e:
            logging.warning(f"[pdf_processing] Erreur pdfplumber sur {path}: {e}")
            text_final = ""

    # 2) Fallback pdfminer si pdfplumber a échoué, non utilisé ou non disponible
    if not text_final or _is_text_suspect(text_final):
        try:
            laparams = LAParams()
            with open(path, "rb") as f:
                text_pdfminer = extract_text(f, laparams=laparams)

            if text_pdfminer and len(text_pdfminer) > len(text_final or ""):
                text_final = text_pdfminer
                extraction_method = "pdfminer"
                logging.info(f"[pdf_processing] Extraction pdfminer OK pour {path}")
        except PDFPasswordIncorrect:
            logging.error(f"[pdf_processing] PDF protégé par mot de passe : {path}")
            raise
        except Exception as e:
            logging.warning(f"[pdf_processing] Erreur pdfminer sur {path}: {e}")

    # 3) Fallback PyMuPDF si texte toujours suspect ou vide
    if not text_final or _is_text_suspect(text_final):
        try:
            text_pymupdf = _extract_text_with_pymupdf(path)

            if text_pymupdf and len(text_pymupdf) > len(text_final or "") * 1.1:
                logging.info(
                    f"[pdf_processing] Texte PyMuPDF choisi (len={len(text_pymupdf)}) "
                    f"vs {extraction_method} (len={len(text_final or '')})."
                )
                text_final = text_pymupdf
                extraction_method = "pymupdf"
            elif not text_final:
                text_final = text_pymupdf
                extraction_method = "pymupdf"
        except Exception as e2:
            logging.error(f"[pdf_processing] Fallback PyMuPDF échoué pour {path}: {e2}")
            if not text_final:
                raise

    # Normalisation minimale des fins de ligne
    text_final = text_final.replace("\r\n", "\n").replace("\r", "\n")

    # 4) Nettoyage headers/footers répétés page par page
    try:
        pages = _split_pages(text_final)
        pages_clean = _remove_repeated_headers_footers(pages)
        text_final = "\f".join(pages_clean)
    except Exception as e:
        logging.warning(f"[pdf_processing] Erreur durant la suppression headers/footers: {e}")

    # 5) Nettoyage du bruit / watermarks
    try:
        text_final = _clean_pdf_text_noise(text_final)
    except Exception as e:
        logging.warning(f"[pdf_processing] Erreur durant le nettoyage du texte: {e}")
        text_final = text_final.strip()

    if progress_cb:
        method_info = f" (via {extraction_method})" if extraction_method else ""
        progress_cb(1.0, f"Lecture PDF terminée{method_info}")

    return text_final

# -----------------------------------------------------------------------------
#  EXTRACTION DES PIECES JOINTES PDF (logique inspirée de test.py)
# -----------------------------------------------------------------------------

def _unique_path(base_dir: str, filename: str) -> str:
    """Retourne un chemin dans base_dir qui n'existe pas encore (suffixe _1, _2, ...)."""
    filename = os.path.basename(filename) or "attachment"
    root, ext = os.path.splitext(filename)
    candidate = os.path.join(base_dir, filename)
    i = 1
    while os.path.exists(candidate):
        candidate = os.path.join(base_dir, f"{root}_{i}{ext}")
        i += 1
    return candidate


def _extract_embedded_files(doc, out_dir: str) -> List[str]:
    """
    Extrait les fichiers intégrés (/EmbeddedFiles) du PDF.
    Retourne la liste des chemins de fichiers extraits.
    """
    extracted: List[str] = []

    try:
        # Nouvelle API (PyMuPDF récent)
        if hasattr(doc, "embfile_names"):
            names = list(doc.embfile_names())
            if names:
                logging.info(f"[pdf_processing] Found {len(names)} embedded file(s) via embfile_names().")
            for name in names:
                try:
                    data = doc.embfile_get(name)

                    filename = None
                    if hasattr(doc, "embfile_info"):
                        try:
                            info = doc.embfile_info(name)
                            if isinstance(info, dict):
                                filename = (
                                    info.get("filename")
                                    or info.get("name")
                                    or None
                                )
                        except Exception as e_info:
                            logging.warning(
                                f"[pdf_processing] embfile_info() failed for '{name}' "
                                f"({type(e_info).__name__}: {e_info}). Using raw name."
                            )

                    if not filename:
                        filename = name or "embedded_file"

                    # Nettoyer le nom de fichier (surrogates, caractères invalides)
                    filename = clean_filename(filename)
                    out_path = _unique_path(out_dir, filename)
                    with open(out_path, "wb") as f:
                        f.write(data)
                    extracted.append(out_path)
                    logging.info(f"[pdf_processing] Extracted embedded file: {out_path}")

                except Exception as e:
                    logging.warning(
                        f"[pdf_processing] Failed to extract embedded file '{name}': {e}"
                    )
                    traceback.print_exc()

        # Ancienne API (fallback)
        elif hasattr(doc, "embeddedFileNames"):
            names = list(doc.embeddedFileNames())
            if names:
                logging.info(
                    f"[pdf_processing] Found {len(names)} embedded file(s) via embeddedFileNames()."
                )
            for name in names:
                try:
                    data = doc.embeddedFileGet(name)

                    filename = None
                    if hasattr(doc, "embeddedFileInfo"):
                        try:
                            info = doc.embeddedFileInfo(name)
                            if isinstance(info, dict):
                                filename = (
                                    info.get("filename")
                                    or info.get("name")
                                    or None
                                )
                        except Exception as e_info:
                            logging.warning(
                                f"[pdf_processing] embeddedFileInfo() failed for '{name}' "
                                f"({type(e_info).__name__}: {e_info}). Using raw name."
                            )

                    if not filename:
                        filename = name or "embedded_file"

                    # Nettoyer le nom de fichier (surrogates, caractères invalides)
                    filename = clean_filename(filename)
                    out_path = _unique_path(out_dir, filename)
                    with open(out_path, "wb") as f:
                        f.write(data)
                    extracted.append(out_path)
                    logging.info(f"[pdf_processing] Extracted embedded file (old API): {out_path}")

                except Exception as e:
                    logging.warning(
                        f"[pdf_processing] Failed to extract embedded file '{name}' (old API): {e}"
                    )
                    traceback.print_exc()
        else:
            logging.info(
                "[pdf_processing] This PyMuPDF version exposes no embedded-file API on Document."
            )

    except Exception as e:
        logging.error(f"[pdf_processing] Error while extracting embedded files: {e}")
        traceback.print_exc()

    return extracted


def _extract_file_attachment_annots(doc, out_dir: str) -> List[str]:
    """
    Extrait les fichiers attachés via des annotations de type 'FileAttachment'
    (icône trombone / punaise sur une page).
    """
    extracted: List[str] = []

    try:
        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            annots = page.annots()
            if annots is None:
                continue

            for annot in annots:
                try:
                    annot_type = annot.type
                    is_file_annot = (
                        isinstance(annot_type, (list, tuple))
                        and len(annot_type) >= 2
                        and annot_type[1] == "FileAttachment"
                    )

                    if not is_file_annot:
                        continue

                    info = getattr(annot, "file_info", None)
                    filename = None
                    if isinstance(info, dict):
                        filename = info.get("filename")

                    if not filename:
                        filename = f"page{page_index+1}_xref{annot.xref}.bin"

                    if hasattr(annot, "get_file"):
                        data = annot.get_file()
                    elif hasattr(annot, "fileGet"):
                        data = annot.fileGet()
                    else:
                        logging.warning(
                            "[pdf_processing] Annotation looks like FileAttachment "
                            "but has no get_file/fileGet()."
                        )
                        continue

                    # Nettoyer le nom de fichier (surrogates, caractères invalides)
                    filename = clean_filename(filename)
                    out_path = _unique_path(out_dir, filename)
                    with open(out_path, "wb") as f:
                        f.write(data)
                    extracted.append(out_path)
                    logging.info(
                        f"[pdf_processing] Extracted file attachment from page "
                        f"{page_index+1}: {out_path}"
                    )

                except Exception as e:
                    logging.warning(
                        f"[pdf_processing] Failed to extract file attachment on page "
                        f"{page_index+1}: {e}"
                    )
                    traceback.print_exc()

    except Exception as e:
        logging.error(
            f"[pdf_processing] Error while scanning attachment annotations: {e}"
        )
        traceback.print_exc()

    return extracted


def _create_pdf_without_attachments(doc, out_path: str) -> None:
    """
    Crée un nouveau PDF sans pièces jointes.
    (On recopie les pages dans un nouveau document.)
    """
    try:
        new_doc = fitz.open()
        for page in doc:
            new_doc.insert_pdf(doc, from_page=page.number, to_page=page.number)
        new_doc.save(out_path)
        new_doc.close()
        logging.info(f"[pdf_processing] Created PDF without attachments: {out_path}")
    except Exception as e:
        logging.error(
            f"[pdf_processing] Failed to create PDF without attachments: {e}"
        )
        traceback.print_exc()


def extract_attachments_from_pdf(
    pdf_path: str,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Extrait les pièces jointes (Embedded Files + annotations FileAttachment)
    et crée un PDF sans pièces jointes dans output_dir.

    Retourne un dict :
    {
        "attachments_paths": [ ... chemins des PJ ... ],
        "pdf_without_attachments_path": "/.../monPDF_sansPJ.pdf" ou None,
        "temp_dir": "/tmp/pdf_attachments_xyz" ou None (si créé automatiquement)
    }
    
    Si output_dir est None, un répertoire temporaire est créé et son chemin
    est retourné dans "temp_dir" pour permettre le nettoyage par l'appelant.
    """
    if fitz is None:
        logging.error("[pdf_processing] PyMuPDF (pymupdf) n'est pas installé.")
        return {
            "attachments_paths": [],
            "pdf_without_attachments_path": None,
            "temp_dir": None
        }

    pdf_path = os.path.abspath(pdf_path)

    if not os.path.exists(pdf_path):
        logging.error(f"[pdf_processing] PDF not found: {pdf_path}")
        return {
            "attachments_paths": [],
            "pdf_without_attachments_path": None,
            "temp_dir": None
        }

    # Garder trace si on a créé le répertoire temporaire
    created_temp_dir = None
    if output_dir is None:
        import tempfile
        output_dir = tempfile.mkdtemp(prefix="pdf_attachments_")
        created_temp_dir = output_dir
        logging.info(f"[pdf_processing] Répertoire temporaire créé : {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    logging.info(f"[pdf_processing] Target PDF for attachment extraction: {pdf_path}")
    logging.info(f"[pdf_processing] Attachments will be extracted into: {output_dir}")

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logging.error(f"[pdf_processing] Cannot open PDF with PyMuPDF: {e}")
        traceback.print_exc()
        return {"attachments_paths": [], "pdf_without_attachments_path": None, "temp_dir": created_temp_dir}

    attachments_paths: List[str] = []
    pdf_without_path: Optional[str] = None

    try:
        # 1) Créer le PDF sans pièces jointes
        pdf_basename = os.path.basename(pdf_path)
        base_root, base_ext = os.path.splitext(pdf_basename)
        pdf_without_path = os.path.join(output_dir, f"{base_root}_sans_pj{base_ext}")
        _create_pdf_without_attachments(doc, pdf_without_path)

        # 2) Extraction des fichiers embarqués + annotations FileAttachment
        attachments_paths.extend(_extract_embedded_files(doc, output_dir))
        attachments_paths.extend(_extract_file_attachment_annots(doc, output_dir))

    finally:
        doc.close()

    if not attachments_paths:
        logging.info(
            "[pdf_processing] No attachments found (embedded files or file-attachment annotations)."
        )

    return {
        "attachments_paths": attachments_paths,
        "pdf_without_attachments_path": pdf_without_path,
        "temp_dir": created_temp_dir,  # Retourner le chemin du répertoire temporaire créé
    }

# -----------------------------------------------------------------------------
# ORCHESTRATEUR GLOBAL
# -----------------------------------------------------------------------------

def extract_all_texts_from_pdf(
    pdf_path: str,
    output_dir: Optional[str] = None,
    progress_cb: Optional[ProgressFn] = None,
) -> Dict[str, Any]:
    """
    Pipeline complet :
      1) Extraction robuste du texte du PDF principal
      2) Extraction des pièces jointes (Embedded + FileAttachment)
      3) Création d'un PDF sans pièces jointes + extraction texte robuste
      4) Extraction du texte des pièces jointes qui sont des PDF

    Retourne un dict :
    {
        "main_pdf_text": <str>,
        "main_pdf_no_attachments_text": <str ou None>,
        "attachments": [
            {
                "name": <str>,
                "path": <str>,
                "size": <int>,
                "is_pdf": <bool>,
                "text": <str ou None>,
            },
            ...
        ],
    }
    """
    pdf_path = os.path.abspath(pdf_path)
    if output_dir is None:
        import tempfile
        output_dir = tempfile.mkdtemp(prefix="pdf_all_texts_")
        logging.info(f"[pdf_processing] Répertoire temporaire créé : {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    def _sub_progress(start: float, end: float) -> ProgressFn:
        """Fabrique un wrapper de progress_cb pour une sous-étape."""
        def wrapper(ratio: float, message: str) -> None:
            if progress_cb:
                overall = start + (end - start) * max(0.0, min(1.0, ratio))
                progress_cb(overall, message)
        return wrapper

    # 1) Texte du PDF principal
    if progress_cb:
        progress_cb(0.0, "Début : extraction texte PDF principal")
    main_text = extract_text_from_pdf(
        pdf_path,
        progress_cb=_sub_progress(0.0, 0.4) if progress_cb else None,
    )

    # 2) Extraction pièces jointes + PDF sans PJ
    if progress_cb:
        progress_cb(0.4, "Extraction des pièces jointes et création du PDF sans pièces jointes")
    attachments_info = extract_attachments_from_pdf(pdf_path, output_dir=output_dir)
    attachments_paths = attachments_info.get("attachments_paths", [])
    pdf_without_path = attachments_info.get("pdf_without_attachments_path")

    # 3) Texte du PDF sans pièces jointes
    main_no_pj_text: Optional[str] = None
    if pdf_without_path and os.path.exists(pdf_without_path):
        if progress_cb:
            progress_cb(0.6, "Extraction texte du PDF sans pièces jointes")
        main_no_pj_text = extract_text_from_pdf(
            pdf_without_path,
            progress_cb=_sub_progress(0.6, 0.8) if progress_cb else None,
        )

    # 4) Texte des pièces jointes PDF
    attachments_result: List[Dict[str, Any]] = []
    if progress_cb:
        progress_cb(0.8, "Extraction texte des pièces jointes PDF")

    for path in attachments_paths:
        try:
            name = os.path.basename(path)
            size = os.path.getsize(path)
        except OSError:
            name = os.path.basename(path)
            size = -1

        is_pdf = name.lower().endswith(".pdf")
        att_text: Optional[str] = None

        if is_pdf:
            try:
                att_text = extract_text_from_pdf(path)
            except Exception as e:
                logging.error(f"[pdf_processing] Erreur extraction texte PJ PDF '{path}': {e}")
                traceback.print_exc()
                att_text = None

        attachments_result.append(
            {
                "name": name,
                "path": path,
                "size": size,
                "is_pdf": is_pdf,
                "text": att_text,
            }
        )

    if progress_cb:
        progress_cb(1.0, "Extraction complète terminée")

    return {
        "main_pdf_text": main_text,
        "main_pdf_no_attachments_text": main_no_pj_text,
        "attachments": attachments_result,
    }
