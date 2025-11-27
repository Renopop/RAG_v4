# rag_query.py

import os
import time
from typing import Any, Dict, List, Optional

from faiss_store import FaissStore

from models_utils import (
    EMBED_MODEL,
    BATCH_SIZE,  # même si non utilisé directement, laissé pour compatibilité
    SNOWFLAKE_API_KEY,
    SNOWFLAKE_API_BASE,
    make_logger,
    create_http_client,
    DirectOpenAIEmbeddings,
    embed_in_batches,
    call_dallem_chat,
)

# Import optionnel du FeedbackStore pour le re-ranking
try:
    from feedback_store import FeedbackStore
    FEEDBACK_AVAILABLE = True
except ImportError:
    FEEDBACK_AVAILABLE = False
    FeedbackStore = None

# Import du module de recherche avancée
try:
    from advanced_search import (
        expand_query_with_llm,
        run_multi_query_search,
        apply_reranking_to_sources,
        extract_keywords,
        filter_sources_by_keywords,
    )
    from models_utils import DALLEM_API_KEY, DALLEM_API_BASE, LLM_MODEL
    ADVANCED_SEARCH_AVAILABLE = True
    BGE_RERANKER_AVAILABLE = True
    KEYWORD_FILTER_AVAILABLE = True
except ImportError:
    ADVANCED_SEARCH_AVAILABLE = False
    BGE_RERANKER_AVAILABLE = False
    KEYWORD_FILTER_AVAILABLE = False

# Import des fonctions de context expansion et cross-references
try:
    from chunking import (
        extract_cross_references,
        add_cross_references_to_chunk,
        get_related_chunks_by_reference,
        expand_chunk_context,
        get_neighboring_chunks,
    )
    CONTEXT_EXPANSION_AVAILABLE = True
except ImportError:
    CONTEXT_EXPANSION_AVAILABLE = False

logger = make_logger(debug=False)


# =====================================================================
#  FAISS STORE
# =====================================================================

def build_store(db_path: str) -> FaissStore:
    """
    Construit un store FAISS sur le répertoire db_path.
    Pas de retry nécessaire: FAISS fonctionne sans problème sur réseau!
    """
    logger.info(f"[QUERY] Creating FAISS store at: {db_path}")
    store = FaissStore(path=db_path)
    logger.info(f"[QUERY] ✅ FAISS store ready (no network issues!)")
    return store


# =====================================================================
#  RAG SUR UNE SEULE COLLECTION
# =====================================================================

def _run_rag_query_single_collection(
    db_path: str,
    collection_name: str,
    question: str,
    top_k: int = 30,
    call_llm: bool = True,
    log=None,
    feedback_store: Optional["FeedbackStore"] = None,
    use_feedback_reranking: bool = False,
    feedback_alpha: float = 0.3,
    use_query_expansion: bool = True,
    num_query_variations: int = 3,
    use_bge_reranker: bool = True,
    use_context_expansion: bool = True,
) -> Dict[str, Any]:
    """
    RAG sur une seule collection.

    - Si call_llm=True : retrieval + appel LLM
    - Si call_llm=False : retrieval uniquement (retourne context_str & sources, answer vide)
    - Si use_feedback_reranking=True et feedback_store fourni : applique le re-ranking
      basé sur les feedbacks utilisateurs
    - Si use_query_expansion=True : génère des variations de la question et fusionne les résultats
    - Si use_bge_reranker=True : applique le reranking BGE après la recherche initiale
    - Si use_context_expansion=True : enrichit les chunks avec contexte voisin et références
    """
    _log = log or logger

    question = (question or "").strip()
    if not question:
        raise ValueError("Question vide")

    _log.info(
        f"[RAG] (single) db={db_path} | collection={collection_name} | top_k={top_k}"
    )

    # 1) FAISS store + collection
    store = build_store(db_path)
    collection = store.get_collection(name=collection_name)

    # 2) Client embeddings Snowflake
    http_client = create_http_client()
    emb_client = DirectOpenAIEmbeddings(
        model=EMBED_MODEL,
        api_key=SNOWFLAKE_API_KEY,
        base_url=SNOWFLAKE_API_BASE,
        http_client=http_client,
        role_prefix=True,
        logger=_log,
    )

    # 3) Recherche avec ou sans Query Expansion
    if use_query_expansion and ADVANCED_SEARCH_AVAILABLE:
        _log.info(f"[RAG] 🔄 Mode Query Expansion activé ({num_query_variations} variations)")

        # Générer les variations de la question
        queries = expand_query_with_llm(
            question=question,
            http_client=http_client,
            api_key=DALLEM_API_KEY,
            api_base=DALLEM_API_BASE,
            model=LLM_MODEL,
            num_variations=num_query_variations,
            log=_log,
        )

        # Fonction d'embedding pour multi-query
        def embed_query(q: str):
            return embed_in_batches(
                texts=[q],
                role="query",
                batch_size=1,
                emb_client=emb_client,
                log=_log,
                dry_run=False,
            )[0]

        # Recherche multi-query
        raw = run_multi_query_search(
            collection=collection,
            queries=queries,
            embed_func=embed_query,
            top_k=top_k,
            log=_log,
        )
        _log.info(f"[RAG] ✅ Multi-query search completed ({len(queries)} queries)")

    else:
        # Mode standard: une seule requête
        q_emb = embed_in_batches(
            texts=[question],
            role="query",
            batch_size=1,
            emb_client=emb_client,
            log=_log,
            dry_run=False,
        )[0]

        _log.info("[RAG] Querying FAISS index...")
        raw = collection.query(
            query_embeddings=[q_emb.tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        _log.info(f"[RAG] ✅ Query successful")

    docs = raw.get("documents", [[]])[0]
    metas = raw.get("metadatas", [[]])[0]
    dists = raw.get("distances", [[]])[0]

    if not docs:
        _log.warning("[RAG] Aucun document retourné par FAISS")
        return {
            "answer": (
                "Aucun contexte trouvé dans la base pour répondre à la question."
                if call_llm
                else ""
            ),
            "context_str": "",
            "raw_results": raw,
            "sources": [],
        }

    # 5) Construction du contexte + liste des sources
    context_blocks: List[str] = []
    sources: List[Dict[str, Any]] = []

    for doc, meta, dist in zip(docs, metas, dists):
        if not isinstance(meta, dict):
            meta = {}

        source_file = meta.get("source_file", "unknown")
        chunk_id = meta.get("chunk_id", "?")
        section_id = meta.get("section_id") or ""
        section_kind = meta.get("section_kind") or ""
        section_title = meta.get("section_title") or ""
        path = meta.get("path") or ""

        # ID de référence pour matcher avec un CSV global
        ref_id = f"{db_path}|{collection_name}|{path or chunk_id}"

        header = (
            f"[source={source_file}, chunk={chunk_id}, "
            f"dist={float(dist):.3f}]"
        )
        context_blocks.append(f"{header}\n{doc}")

        sources.append(
            {
                "collection": collection_name,
                "source_file": source_file,
                "path": path,
                "chunk_id": chunk_id,
                "distance": float(dist),
                "score": 1.0 - float(dist),
                "section_id": section_id,
                "section_kind": section_kind,
                "section_title": section_title,
                "ref_id": ref_id,
                "text": doc,
            }
        )

    # ========== RE-RANKING BASÉ SUR LES FEEDBACKS ==========
    if use_feedback_reranking and feedback_store and FEEDBACK_AVAILABLE:
        _log.info("[RAG] Applying feedback-based re-ranking...")
        try:
            # Extraire le nom de la base depuis db_path
            base_name = os.path.basename(db_path)

            # Appliquer le re-ranking
            sources = feedback_store.compute_reranking_factors(
                sources=sources,
                base_name=base_name,
                collection_name=collection_name,
                question=question,
                alpha=feedback_alpha
            )

            # Reconstruire le contexte avec l'ordre des sources re-rankées
            context_blocks = []
            for src in sources:
                header = (
                    f"[source={src['source_file']}, chunk={src['chunk_id']}, "
                    f"score={src['score']:.3f}, boost={src.get('feedback_boost', 0):.3f}]"
                )
                context_blocks.append(f"{header}\n{src['text']}")

            _log.info(f"[RAG] ✅ Re-ranking applied (alpha={feedback_alpha})")
        except Exception as e:
            _log.warning(f"[RAG] Re-ranking failed, using original order: {e}")

    # ========== RE-RANKING BGE (cross-encoder) ==========
    if use_bge_reranker and BGE_RERANKER_AVAILABLE:
        _log.info("[RAG] 🔄 Applying BGE Reranker...")
        try:
            sources = apply_reranking_to_sources(
                query=question,
                sources=sources,
                top_k=top_k,
                http_client=http_client,
                log=_log
            )

            # Reconstruire le contexte avec l'ordre des sources reranked
            context_blocks = []
            for src in sources:
                header = (
                    f"[source={src['source_file']}, chunk={src['chunk_id']}, "
                    f"rerank_score={src.get('rerank_score', src.get('score', 0)):.3f}]"
                )
                context_blocks.append(f"{header}\n{src['text']}")

            _log.info(f"[RAG] ✅ BGE Reranking applied")
        except Exception as e:
            _log.warning(f"[RAG] BGE Reranking failed, using previous order: {e}")

    # ========== FILTRAGE PAR MOTS-CLÉS ==========
    if KEYWORD_FILTER_AVAILABLE:
        _log.info("[RAG] 🔑 Applying keyword filtering...")
        try:
            # Extraire les mots-clés de la question
            keywords = extract_keywords(question, min_length=3, log=_log)

            if keywords:
                # Filtrer les sources qui contiennent au moins 1 mot-clé
                sources = filter_sources_by_keywords(
                    sources=sources,
                    keywords=keywords,
                    min_matches=1,
                    log=_log
                )

                # Reconstruire le contexte avec les sources filtrées
                context_blocks = []
                for src in sources:
                    kw_info = f", keywords={src.get('keyword_count', 0)}" if 'keyword_count' in src else ""
                    header = (
                        f"[source={src['source_file']}, chunk={src['chunk_id']}, "
                        f"score={src.get('score', 0):.3f}{kw_info}]"
                    )
                    context_blocks.append(f"{header}\n{src['text']}")

                _log.info(f"[RAG] ✅ Keyword filtering applied ({len(sources)} sources)")
        except Exception as e:
            _log.warning(f"[RAG] Keyword filtering failed: {e}")

    # ========== CONTEXT EXPANSION (Cross-References + Neighbors) ==========
    if use_context_expansion and CONTEXT_EXPANSION_AVAILABLE:
        _log.info("[RAG] 🔗 Applying context expansion...")
        try:
            expanded_context_blocks = []
            seen_chunks = set()  # Pour éviter les doublons

            for src in sources[:15]:  # Limiter aux 15 premiers pour performance
                chunk_id = src.get("chunk_id", "")
                if chunk_id in seen_chunks:
                    continue
                seen_chunks.add(chunk_id)

                # Extraire les références croisées du chunk
                refs = extract_cross_references(src.get("text", ""))
                ref_ids = [r["ref_id"] for r in refs if r["ref_type"] in ("CS", "AMC", "GM")]

                # Construire le bloc de contexte principal
                main_header = (
                    f"[source={src['source_file']}, chunk={chunk_id}, "
                    f"score={src.get('score', 0):.3f}]"
                )
                main_block = f"{main_header}\n{src['text']}"

                # Ajouter les références détectées aux métadonnées
                if ref_ids:
                    src["references_to"] = ref_ids[:5]  # Max 5 références
                    _log.debug(f"[RAG] Chunk {chunk_id} references: {ref_ids[:5]}")

                expanded_context_blocks.append(main_block)

            # Chercher les chunks référencés dans les autres sources
            referenced_chunks = []
            all_refs = set()

            for src in sources[:15]:
                refs_to = src.get("references_to", [])
                for ref in refs_to:
                    ref_normalized = ref.upper().replace(" ", "").replace("-", "")
                    all_refs.add(ref_normalized)

            # Trouver les chunks qui correspondent aux références
            if all_refs:
                for src in sources:
                    section_id = src.get("section_id", "")
                    if section_id:
                        section_normalized = section_id.upper().replace(" ", "").replace("-", "")
                        if section_normalized in all_refs and src.get("chunk_id") not in seen_chunks:
                            referenced_chunks.append(src)
                            seen_chunks.add(src.get("chunk_id"))

                # Ajouter les chunks référencés au contexte
                if referenced_chunks:
                    _log.info(f"[RAG] Adding {len(referenced_chunks)} referenced chunk(s) to context")
                    for ref_src in referenced_chunks[:5]:  # Max 5 chunks référencés
                        ref_header = (
                            f"[REFERENCED: source={ref_src['source_file']}, "
                            f"section={ref_src.get('section_id', '?')}]"
                        )
                        expanded_context_blocks.append(f"{ref_header}\n{ref_src['text']}")

            context_blocks = expanded_context_blocks
            _log.info(f"[RAG] ✅ Context expansion applied ({len(context_blocks)} blocks)")

        except Exception as e:
            _log.warning(f"[RAG] Context expansion failed: {e}")

    full_context = "\n\n".join(context_blocks)

    if not call_llm:
        # Mode "retrieval only"
        return {
            "answer": "",
            "context_str": full_context,
            "raw_results": raw,
            "sources": sources,
        }

    # 6) Appel LLM DALLEM
    answer = call_dallem_chat(
        http_client=http_client,
        question=question,
        context=full_context,
        log=_log,
    )

    return {
        "answer": answer,
        "context_str": full_context,
        "raw_results": raw,
        "sources": sources,
    }


# =====================================================================
#  RAG : UNE COLLECTION OU TOUTES LES COLLECTIONS (ALL)
# =====================================================================

def run_rag_query(
    db_path: str,
    collection_name: str,
    question: str,
    top_k: int = 30,
    synthesize_all: bool = False,
    log=None,
    feedback_store: Optional["FeedbackStore"] = None,
    use_feedback_reranking: bool = False,
    feedback_alpha: float = 0.3,
    use_query_expansion: bool = True,
    num_query_variations: int = 3,
    use_bge_reranker: bool = True,
    use_context_expansion: bool = True,
) -> Dict[str, Any]:
    """
    RAG "haut niveau" :

    - Si collection_name != "ALL" :
        → requête sur UNE collection (cf. _run_rag_query_single_collection)

    - Si collection_name == "ALL" :
        - synthesize_all = True :
             → retrieval sur toutes les collections,
               concaténé dans un gros contexte global,
               puis un SEUL appel LLM DALLEM.
        - synthesize_all = False :
             → par sécurité, on se contente de lever une erreur ou
               de déléguer à l'appelant (dans ton streamlit, ce cas
               est géré côté interface, pas ici).

    Options de recherche avancée :
    - use_query_expansion : génère des variations de la question pour améliorer le recall
    - num_query_variations : nombre de variations à générer (défaut: 3)
    - use_bge_reranker : applique le reranking BGE après la recherche (défaut: True)
    - use_context_expansion : enrichit les résultats avec contexte et références (défaut: True)

    Options de re-ranking basé sur les feedbacks :
    - feedback_store : instance de FeedbackStore pour accéder aux feedbacks
    - use_feedback_reranking : activer le re-ranking basé sur les feedbacks
    - feedback_alpha : facteur d'influence (0-1, défaut 0.3)
    """
    _log = log or logger

    # Cas "ALL" (utilisé par streamlit_RAG quand synthesize_all=True)
    if collection_name == "ALL":
        _log.info(
            f"[RAG] Mode ALL collections | db={db_path} | synthesize_all={synthesize_all}"
        )
        store = build_store(db_path)
        collections = store.list_collections()  # FAISS retourne directement une liste de noms

        if not collections:
            return {
                "answer": "Aucune collection disponible dans cette base FAISS.",
                "context_str": "",
                "raw_results": {},
                "sources": [],
            }

        if not synthesize_all:
            # Dans ton streamlit, ce cas est géré côté interface (boucle sur les collections),
            # donc ici on renvoie une erreur explicite pour éviter toute ambiguïté.
            raise ValueError(
                "run_rag_query(collection_name='ALL') appelé avec synthesize_all=False. "
                "Ce mode doit être géré côté interface (une requête par collection)."
            )

        # ---- Mode synthèse globale : un seul appel LLM avec le contexte concaténé ----
        all_sources: List[Dict[str, Any]] = []
        all_context_blocks: List[str] = []

        for col_name in collections:  # FAISS retourne directement les noms (strings)
            _log.info(f"[RAG-ALL-SYNTH] Retrieval sur collection '{col_name}'")

            try:
                res = _run_rag_query_single_collection(
                    db_path=db_path,
                    collection_name=col_name,
                    question=question,
                    top_k=top_k,
                    call_llm=False,  # pas d'appel LLM ici
                    log=_log,
                    feedback_store=feedback_store,
                    use_feedback_reranking=use_feedback_reranking,
                    feedback_alpha=feedback_alpha,
                    use_query_expansion=use_query_expansion,
                    num_query_variations=num_query_variations,
                    use_bge_reranker=use_bge_reranker,
                    use_context_expansion=use_context_expansion,
                )
            except Exception as e:
                _log.error(
                    f"[RAG-ALL-SYNTH] Erreur pendant la récupération sur '{col_name}' : {e}"
                )
                continue

            context_str = res.get("context_str", "")
            sources = res.get("sources", [])

            if context_str:
                all_context_blocks.append(
                    f"=== CONTEXTE {col_name} ===\n{context_str}".strip()
                )

            for s in sources:
                if "collection" not in s:
                    s["collection"] = col_name
                all_sources.append(s)

        global_context = "\n\n".join(all_context_blocks)

        if not global_context.strip():
            return {
                "answer": (
                    "Aucun contexte trouvé dans la base pour répondre à la question (mode ALL)."
                ),
                "context_str": "",
                "raw_results": {},
                "sources": [],
            }

        http_client = create_http_client()
        answer = call_dallem_chat(
            http_client=http_client,
            question=(question or "").strip(),
            context=global_context,
            log=_log,
        )

        return {
            "answer": answer,
            "context_str": global_context,
            "raw_results": {},
            "sources": all_sources,
        }

    # Cas normal : une seule collection
    return _run_rag_query_single_collection(
        db_path=db_path,
        collection_name=collection_name,
        question=question,
        top_k=top_k,
        call_llm=True,
        log=_log,
        feedback_store=feedback_store,
        use_feedback_reranking=use_feedback_reranking,
        feedback_alpha=feedback_alpha,
        use_query_expansion=use_query_expansion,
        num_query_variations=num_query_variations,
        use_bge_reranker=use_bge_reranker,
        use_context_expansion=use_context_expansion,
    )
