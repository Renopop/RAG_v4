import os
import uuid
import gc
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional

from langdetect import detect

from faiss_store import FaissStore

from models_utils import (
    make_logger,
    EMBED_MODEL,
    BATCH_SIZE,
    DirectOpenAIEmbeddings,
    embed_in_batches,
    SNOWFLAKE_API_KEY,
    SNOWFLAKE_API_BASE,
    create_http_client,
)

from pdf_processing import extract_text_from_pdf, extract_attachments_from_pdf
from docx_processing import extract_text_from_docx
from csv_processing import extract_text_from_csv
from xml_processing import extract_text_from_xml, XMLParseConfig
from chunking import (
    simple_chunk,
    chunk_easa_sections,
    smart_chunk_generic,
    adaptive_chunk_document,
    augment_chunks,
    _calculate_content_density,
    _get_adaptive_chunk_size,
    add_cross_references_to_chunks,
    extract_cross_references,
)
from easa_sections import split_easa_sections

logger = make_logger(debug=False)

# =====================================================================
#  FAISS HELPERS
# =====================================================================

def build_faiss_store(path: str) -> FaissStore:
    """Create (and if needed, initialize) a FAISS store."""
    os.makedirs(path, exist_ok=True)
    return FaissStore(path=path)


def get_or_create_collection(store: FaissStore, name: str):
    """Return an existing collection or create it if missing."""
    return store.get_or_create_collection(name=name, dimension=1024)  # Snowflake Arctic = 1024d

# =====================================================================
#  FILE LOADING
# =====================================================================

def load_file_content(path: str, xml_configs: Optional[Dict[str, XMLParseConfig]] = None) -> str:
    """Load text from a supported file type (PDF, DOCX, CSV, TXT, MD, XML).

    Args:
        path: Chemin vers le fichier
        xml_configs: Dict optionnel {chemin_fichier: XMLParseConfig} pour les fichiers XML
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    if ext in (".doc", ".docx"):
        # Note: legacy .doc may fail depending on content; .docx is fully supported.
        return extract_text_from_docx(path)
    if ext == ".csv":
        return extract_text_from_csv(path)
    if ext == ".xml":
        # Utiliser la config spécifique si fournie, sinon config par défaut
        config = xml_configs.get(path) if xml_configs else None
        return extract_text_from_xml(path, config)
    if ext in (".txt", ".md"):
        # Plain text / Markdown files: read as UTF-8 with tolerant error handling
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    raise ValueError(f"Unsupported file format for ingestion: {ext} ({path})")


def detect_language(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "unk"


# Variable globale pour stocker les configs XML (utilisée par le worker)
_xml_configs_global: Optional[Dict[str, XMLParseConfig]] = None


def _load_single_file_worker(path: str) -> Dict[str, Any]:
    """
    Worker function for parallel file loading.
    Returns a dict with path, text, language, and error (if any).
    Must be at module level for pickling by multiprocessing.
    """
    global _xml_configs_global

    result = {
        "path": path,
        "text": "",
        "language": "",
        "error": None
    }

    try:
        if not os.path.isfile(path):
            result["error"] = f"File not found: {path}"
            return result

        text = load_file_content(path, _xml_configs_global)

        if not text.strip():
            result["error"] = f"No text extracted from {path}"
            return result

        result["text"] = text
        result["language"] = detect_language(text) or ""

    except Exception as e:
        result["error"] = f"Error loading {path}: {str(e)}"

    return result


# =====================================================================
#  INGESTION
# =====================================================================

def ingest_documents(
    file_paths: List[str],
    db_path: str,
    collection_name: str,
    chunk_size: int = 1000,
    use_easa_sections: bool = True,
    rebuild: bool = False,
    log=None,
    logical_paths: Optional[Dict[str, str]] = None,
    progress_callback: Optional[callable] = None,
    xml_configs: Optional[Dict[str, XMLParseConfig]] = None,
) -> Dict[str, Any]:
    """Ingest a list of documents into a FAISS collection.

    Steps:
      - load raw text (PDF / DOCX / CSV / XML)
      - optionally split into EASA sections
      - chunk text
      - compute embeddings with Snowflake
      - add to FAISS

    Args:
        xml_configs: Dict optionnel {chemin_fichier: XMLParseConfig} pour les fichiers XML

    Returns a small report with total_chunks and per-file info.
    """
    global _xml_configs_global
    _xml_configs_global = xml_configs  # Rendre accessible au worker

    _log = log or logger

    _log.info(f"[INGEST] DB={db_path} | collection={collection_name}")
    client = build_faiss_store(db_path)

    # Option rebuild: drop collection if it already exists
    if rebuild:
        try:
            existing = client.list_collections()  # FAISS retourne directement une liste de noms
            if collection_name in existing:
                client.delete_collection(collection_name)
                _log.info(
                    f"[INGEST] Collection '{collection_name}' deleted (rebuild=True)"
                )
        except Exception as e:
            _log.warning(
                f"[INGEST] Could not delete collection '{collection_name}': {e}"
            )

    col = get_or_create_collection(client, collection_name)

    # Embeddings client for Snowflake Arctic
    http_client = create_http_client()
    emb_client = DirectOpenAIEmbeddings(
        model=EMBED_MODEL,
        api_key=SNOWFLAKE_API_KEY,
        base_url=SNOWFLAKE_API_BASE,
        http_client=http_client,
        role_prefix=True,
        logger=_log,
    )

    total_chunks = 0
    file_reports: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Chargement parallèle des fichiers (extraction texte multi-threads)
    # ------------------------------------------------------------------
    # Note: ThreadPoolExecutor au lieu de ProcessPoolExecutor pour éviter
    # les problèmes avec PyMuPDF sur Windows (MemoryError, DLL load failures)
    _log.info(f"[INGEST] Starting parallel file loading with {multiprocessing.cpu_count()} threads")

    # Utiliser ThreadPoolExecutor pour le traitement parallèle
    # Threads au lieu de processus = meilleure compatibilité Windows + PyMuPDF
    max_workers = min(multiprocessing.cpu_count(), len(file_paths))

    # Dictionnaire pour stocker les résultats chargés: path -> {text, language, error}
    loaded_files: Dict[str, Dict[str, Any]] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Soumettre tous les fichiers pour traitement parallèle
        future_to_path = {executor.submit(_load_single_file_worker, path): path for path in file_paths}

        # Compteur pour la barre de progression
        completed_count = 0
        total_count = len(file_paths)

        # Récupérer les résultats au fur et à mesure
        for future in as_completed(future_to_path):
            try:
                result = future.result()
                path = result["path"]
                filename = os.path.basename(path)

                if result["error"]:
                    _log.error(f"[INGEST] {result['error']}")
                else:
                    loaded_files[path] = result
                    _log.info(f"[INGEST] Loaded file: {path} ({len(result['text'])} chars)")

                # Mise à jour de la progression
                completed_count += 1
                if progress_callback:
                    progress_fraction = 0.3 * (completed_count / total_count)  # 0-30% pour le chargement
                    progress_callback(
                        progress_fraction,
                        f"📄 Chargement: {completed_count}/{total_count} - {filename}"
                    )

            except Exception as e:
                path = future_to_path[future]
                _log.error(f"[INGEST] Unexpected error loading {path}: {e}")
                completed_count += 1
                if progress_callback:
                    progress_fraction = 0.3 * (completed_count / total_count)
                    progress_callback(progress_fraction, f"❌ Erreur: {os.path.basename(path)}")

    _log.info(f"[INGEST] Parallel loading complete: {len(loaded_files)}/{len(file_paths)} files loaded")

    # ------------------------------------------------------------------
    # Traitement séquentiel des fichiers chargés (chunking + embedding)
    # ------------------------------------------------------------------
    processed_count = 0
    total_loaded = len(loaded_files)

    for path in file_paths:
        # Récupérer le fichier chargé (peut être None si échec)
        file_data = loaded_files.get(path)
        if not file_data:
            continue  # Déjà logué dans la phase de chargement

        text = file_data["text"]
        language = file_data["language"]

        # --------------------------------------------------------------
        # Try to split into EASA sections (CS / AMC / GM / CS-E / CS-APU)
        # --------------------------------------------------------------
        sections = split_easa_sections(text) if use_easa_sections else []

        chunks: List[str] = []
        metas: List[Dict[str, Any]] = []
        faiss_ids: List[str] = []  # IDs envoyés à FAISS (doivent être uniques)

        base_name = os.path.basename(path)

        # ==============================================================
        # CAS 1 — Sections EASA détectées → Smart Chunking Adaptatif
        # ==============================================================
        if sections:
            _log.info(
                f"[INGEST] {len(sections)} EASA section(s) detected for {path} → Adaptive Smart Chunking"
            )

            # Analyser la densité du document pour adapter la taille des chunks
            density_info = _calculate_content_density(text)
            adapted_chunk_size = _get_adaptive_chunk_size(
                text,
                base_size=chunk_size,
                min_size=600,
                max_size=2000
            )

            _log.info(
                f"[INGEST] Content density: {density_info['density_type']} "
                f"(score={density_info['density_score']:.2f}) → chunk_size={adapted_chunk_size}"
            )

            # Utiliser le smart chunking adaptatif qui:
            # - Préserve le contexte [CS xx.xxx - Title] dans chaque chunk
            # - Ne redécoupe pas les petites sections
            # - Fusionne les sections trop petites
            # - Découpe intelligemment par sous-sections (a), (b), etc.
            # - Adapte la taille selon la densité du contenu
            smart_chunks = chunk_easa_sections(
                sections,
                max_chunk_size=adapted_chunk_size + 500,  # +500 pour le préfixe de contexte
                min_chunk_size=200,
                merge_small_sections=True,
                add_context_prefix=True,
            )

            # Augmenter les chunks avec mots-clés et métadonnées
            smart_chunks = augment_chunks(
                smart_chunks,
                add_keywords=True,
                add_key_phrases=True,
                add_density_info=True,
            )

            # Ajouter les cross-références (liens vers autres sections)
            smart_chunks = add_cross_references_to_chunks(smart_chunks)

            _log.info(f"[INGEST] Adaptive chunking: {len(sections)} sections → {len(smart_chunks)} augmented chunks")

            for smart_chunk in smart_chunks:
                ch = smart_chunk.get("text", "")
                sec_id = smart_chunk.get("section_id", "")
                sec_kind = smart_chunk.get("section_kind", "")
                sec_title = smart_chunk.get("section_title", "")
                chunk_idx = smart_chunk.get("chunk_index", 0)

                # Données d'augmentation
                keywords = smart_chunk.get("keywords", [])
                density_type = smart_chunk.get("density_type", "")
                density_score = smart_chunk.get("density_score", 0)
                references_to = smart_chunk.get("references_to", [])

                if not ch:
                    continue

                # chunk_id lisible / logique
                safe_sec_id = sec_id.replace(" ", "_").replace("|", "_") if sec_id else "no_section"
                chunk_id = f"{base_name}_{safe_sec_id}_{chunk_idx}"

                # ID réellement utilisé par FAISS (unique grâce à un uuid par chunk)
                faiss_id = f"{chunk_id}__{uuid.uuid4().hex[:8]}"

                chunks.append(ch)
                metas.append(
                    {
                        "source_file": base_name,
                        "path": logical_paths.get(path, path) if logical_paths else path,
                        "chunk_id": chunk_id,
                        "section_id": sec_id,
                        "section_kind": sec_kind,
                        "section_title": sec_title,
                        "language": language,
                        "is_complete_section": smart_chunk.get("is_complete_section", False),
                        # Métadonnées d'augmentation
                        "keywords": keywords[:10] if keywords else [],
                        "density_type": density_type,
                        "density_score": density_score,
                        # Cross-références vers d'autres sections
                        "references_to": references_to[:5] if references_to else [],
                    }
                )
                faiss_ids.append(faiss_id)

        # ==============================================================
        # CAS 2 — Aucune section EASA : Chunking Adaptatif Automatique
        # ==============================================================
        else:
            _log.info(
                f"[INGEST] No EASA sections detected → Adaptive Automatic Chunking for {path}"
            )

            # Analyser la densité du document pour adapter la taille des chunks
            density_info = _calculate_content_density(text)
            adapted_chunk_size = _get_adaptive_chunk_size(
                text,
                base_size=chunk_size,
                min_size=600,
                max_size=2000
            )

            _log.info(
                f"[INGEST] Content density: {density_info['density_type']} "
                f"(score={density_info['density_score']:.2f}) → chunk_size={adapted_chunk_size}"
            )

            # Utiliser le smart chunking générique adaptatif qui:
            # - Détecte les titres/headers et les garde avec leur contenu
            # - Préserve les listes (ne coupe pas au milieu)
            # - Coupe aux fins de phrases
            # - Ajoute le contexte source [Source: filename]
            # - Respecte la structure du document
            # - Adapte la taille selon la densité
            smart_chunks = smart_chunk_generic(
                text,
                source_file=base_name,
                chunk_size=adapted_chunk_size + 300,  # +300 pour les préfixes
                min_chunk_size=200,
                overlap=100,
                add_source_prefix=True,
                preserve_lists=True,
                preserve_headers=True,
            )

            # Augmenter les chunks avec mots-clés et métadonnées
            smart_chunks = augment_chunks(
                smart_chunks,
                add_keywords=True,
                add_key_phrases=True,
                add_density_info=True,
            )

            # Ajouter les cross-références (liens vers autres sections)
            smart_chunks = add_cross_references_to_chunks(smart_chunks)

            _log.info(f"[INGEST] Adaptive generic chunking: {len(smart_chunks)} augmented chunks")

            for smart_chunk in smart_chunks:
                ch = smart_chunk.get("text", "")
                chunk_idx = smart_chunk.get("chunk_index", 0)
                header = smart_chunk.get("header", "")

                # Données d'augmentation
                keywords = smart_chunk.get("keywords", [])
                density_type = smart_chunk.get("density_type", "")
                density_score = smart_chunk.get("density_score", 0)
                references_to = smart_chunk.get("references_to", [])

                if not ch:
                    continue

                chunk_id = f"{base_name}_chunk_{chunk_idx}"
                faiss_id = f"{chunk_id}__{uuid.uuid4().hex[:8]}"

                chunks.append(ch)
                metas.append(
                    {
                        "source_file": base_name,
                        "path": logical_paths.get(path, path) if logical_paths else path,
                        "chunk_id": chunk_id,
                        "section_id": header[:50] if header else "",  # Utiliser le header détecté
                        "section_kind": smart_chunk.get("type", ""),
                        "section_title": header if header else "",
                        "language": language,
                        # Métadonnées d'augmentation
                        "keywords": keywords[:10] if keywords else [],
                        "density_type": density_type,
                        "density_score": density_score,
                        # Cross-références vers d'autres sections
                        "references_to": references_to[:5] if references_to else [],
                    }
                )
                faiss_ids.append(faiss_id)

        if not chunks:
            _log.warning(f"[INGEST] No chunks generated for {path}")
            continue

        # --------------------------------------------------------------
        # Compute embeddings
        # --------------------------------------------------------------
        _log.info(f"[INGEST] Embedding {len(chunks)} chunks for {path}")
        embeddings = embed_in_batches(
            texts=chunks,
            role="doc",          # -> embed_documents côté client
            batch_size=BATCH_SIZE,
            emb_client=emb_client,
            log=_log,
            dry_run=False,
        )

        # --------------------------------------------------------------
        # Push to FAISS in batches
        # --------------------------------------------------------------
        max_batch = 4000
        n = len(chunks)
        _log.info(f"[INGEST] Adding {n} embeddings to FAISS in batches of {max_batch}")

        for start in range(0, n, max_batch):
            end = start + max_batch
            _log.debug(f"[INGEST] FAISS add batch {start}:{end}")
            col.add(
                documents=chunks[start:end],
                metadatas=metas[start:end],
                embeddings=embeddings[start:end].tolist(),
                ids=faiss_ids[start:end],
            )

        total_chunks += len(chunks)
        file_reports.append(
            {
                "file": base_name,
                "num_chunks": len(chunks),
                "language": language,
                "sections_detected": bool(sections),
            }
        )

        # Mise à jour de la progression après traitement complet du fichier
        processed_count += 1
        if progress_callback:
            # 30-100% pour le chunking et embedding
            progress_fraction = 0.3 + (0.7 * (processed_count / total_loaded))
            progress_callback(
                progress_fraction,
                f"✅ Indexation: {processed_count}/{total_loaded} - {base_name} ({len(chunks)} chunks)"
            )

    _log.info("[INGEST] Completed")

    # =====================================================================
    # Nettoyage explicite pour assurer la synchronisation sur stockage réseau
    # =====================================================================
    # FAISS sauvegarde automatiquement après chaque add() dans notre implémentation,
    # mais on garde un délai pour que Windows synchronise les fichiers vers le réseau.

    try:
        _log.info("[INGEST] FAISS cleanup for network storage synchronization...")

        # Libérer les ressources Python
        del col
        del client
        gc.collect()

        _log.info("[INGEST] FAISS store closed and resources freed")

        # Délai pour que Windows synchronise les fichiers FAISS vers le réseau
        _log.info("[INGEST] Waiting 2 seconds for OS to flush data to network storage...")
        time.sleep(2)

        _log.info("[INGEST] ✅ Database fully synchronized - safe to shut down PC")

    except Exception as e:
        _log.warning(f"[INGEST] Error during cleanup (database may still be OK): {e}")

    return {"total_chunks": total_chunks, "files": file_reports}
