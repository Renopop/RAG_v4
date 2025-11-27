"""
Advanced Search Module - Query Expansion & Multi-Query
Améliore la recherche RAG sans dépendances supplémentaires
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


def expand_query_with_llm(
    question: str,
    http_client,
    api_key: str,
    api_base: str,
    model: str,
    num_variations: int = 3,
    log=None
) -> List[str]:
    """
    Génère des variations de la question originale pour améliorer le recall.
    Utilise le LLM local si configuré, sinon l'API.

    Args:
        question: Question originale
        http_client: Client HTTP
        api_key: Clé API du LLM
        api_base: URL de base de l'API
        model: Nom du modèle
        num_variations: Nombre de variations à générer
        log: Logger optionnel

    Returns:
        Liste de questions (originale + variations)
    """
    _log = log or logger

    # Toujours inclure la question originale
    queries = [question]

    system_prompt = """Tu es un expert en reformulation de questions pour la recherche documentaire.
Génère des variations de la question qui pourraient aider à trouver des documents pertinents.
Les variations doivent:
- Utiliser des synonymes
- Reformuler différemment
- Être plus spécifiques ou plus générales
- Garder le même sens

Réponds UNIQUEMENT avec les variations, une par ligne, sans numérotation ni explication."""

    user_prompt = f"""Question originale: {question}

Génère {num_variations} variations de cette question pour améliorer la recherche:"""

    try:
        url = api_base.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.7,
            "max_tokens": 300,
        }

        resp = http_client.post(url, headers=headers, json=payload, timeout=30.0)
        resp.raise_for_status()
        data = resp.json()

        content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

        if content:
            # Parser les variations (une par ligne)
            variations = [v.strip() for v in content.split("\n") if v.strip()]
            # Filtrer les lignes qui ressemblent à des numéros ou sont trop courtes
            variations = [v for v in variations if len(v) > 10 and not v[0].isdigit()]
            queries.extend(variations[:num_variations])

        _log.info(f"[QUERY-EXPAND] {len(queries)} requêtes générées (original + {len(queries)-1} variations)")

    except Exception as e:
        _log.warning(f"[QUERY-EXPAND] Échec de l'expansion: {e}. Utilisation de la question originale uniquement.")

    return queries


def merge_search_results(
    results_list: List[Dict[str, Any]],
    max_results: int = 30,
    log=None
) -> Tuple[List[str], List[Dict], List[float]]:
    """
    Fusionne les résultats de plusieurs requêtes en éliminant les doublons
    et en combinant les scores.

    Args:
        results_list: Liste des résultats de chaque requête
        max_results: Nombre max de résultats à retourner
        log: Logger optionnel

    Returns:
        Tuple (documents, metadatas, scores)
    """
    _log = log or logger

    # Dictionnaire pour agréger les scores par document
    doc_scores: Dict[str, Dict[str, Any]] = {}

    for query_idx, result in enumerate(results_list):
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, dists):
            # Utiliser le chunk_id ou le hash du document comme clé
            doc_key = meta.get("chunk_id") or meta.get("path") or hash(doc[:100])
            doc_key = str(doc_key)

            # Score = 1 - distance (pour L2, plus petit = meilleur)
            score = 1.0 - float(dist) if dist < 1.0 else 0.0

            if doc_key not in doc_scores:
                doc_scores[doc_key] = {
                    "document": doc,
                    "metadata": meta,
                    "scores": [],
                    "best_score": score,
                    "hit_count": 0,
                }

            doc_scores[doc_key]["scores"].append(score)
            doc_scores[doc_key]["hit_count"] += 1
            doc_scores[doc_key]["best_score"] = max(doc_scores[doc_key]["best_score"], score)

    # Calculer le score final combiné
    # Formule: score_final = best_score * (1 + 0.1 * (hit_count - 1))
    # Un document trouvé par plusieurs requêtes est boosté
    for doc_key, data in doc_scores.items():
        hit_bonus = 0.1 * (data["hit_count"] - 1)
        data["final_score"] = data["best_score"] * (1 + hit_bonus)

    # Trier par score final décroissant
    sorted_docs = sorted(doc_scores.values(), key=lambda x: x["final_score"], reverse=True)

    # Extraire les résultats
    documents = [d["document"] for d in sorted_docs[:max_results]]
    metadatas = [d["metadata"] for d in sorted_docs[:max_results]]
    scores = [d["final_score"] for d in sorted_docs[:max_results]]

    _log.info(f"[MERGE] {len(doc_scores)} documents uniques fusionnés, {len(documents)} retenus")

    return documents, metadatas, scores


def run_multi_query_search(
    collection,
    queries: List[str],
    embed_func,
    top_k: int = 20,
    log=None
) -> Dict[str, Any]:
    """
    Exécute plusieurs requêtes et fusionne les résultats.

    Args:
        collection: Collection FAISS
        queries: Liste des requêtes (question originale + variations)
        embed_func: Fonction pour générer les embeddings
        top_k: Nombre de résultats par requête
        log: Logger optionnel

    Returns:
        Résultats fusionnés au format FAISS
    """
    _log = log or logger

    all_results = []

    for i, query in enumerate(queries):
        _log.info(f"[MULTI-QUERY] Requête {i+1}/{len(queries)}: {query[:50]}...")

        try:
            # Générer l'embedding de la requête
            q_emb = embed_func(query)

            # Recherche FAISS
            result = collection.query(
                query_embeddings=[q_emb.tolist()],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
            all_results.append(result)

        except Exception as e:
            _log.warning(f"[MULTI-QUERY] Erreur sur requête {i+1}: {e}")
            continue

    if not all_results:
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    # Fusionner les résultats
    documents, metadatas, scores = merge_search_results(all_results, max_results=top_k * 2, log=_log)

    # Convertir scores en distances pour compatibilité
    distances = [1.0 - s for s in scores]

    return {
        "documents": [documents],
        "metadatas": [metadatas],
        "distances": [distances],
    }


def generate_sub_questions(
    question: str,
    http_client,
    api_key: str,
    api_base: str,
    model: str,
    log=None
) -> List[str]:
    """
    Décompose une question complexe en sous-questions plus simples.
    Utile pour les questions multi-aspects.

    Args:
        question: Question complexe
        http_client: Client HTTP
        api_key: Clé API
        api_base: URL de base
        model: Nom du modèle
        log: Logger

    Returns:
        Liste de sous-questions
    """
    _log = log or logger

    system_prompt = """Tu es un expert en décomposition de questions complexes.
Si la question contient plusieurs aspects ou sous-questions implicites, décompose-la.
Si la question est simple, retourne-la telle quelle.

Réponds UNIQUEMENT avec les questions, une par ligne, sans numérotation ni explication."""

    user_prompt = f"""Question: {question}

Décompose cette question si elle contient plusieurs aspects:"""

    try:
        url = api_base.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 200,
        }

        resp = http_client.post(url, headers=headers, json=payload, timeout=30.0)
        resp.raise_for_status()
        data = resp.json()

        content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

        if content:
            sub_questions = [q.strip() for q in content.split("\n") if q.strip() and len(q.strip()) > 10]
            if sub_questions:
                _log.info(f"[SUB-Q] Question décomposée en {len(sub_questions)} sous-questions")
                return sub_questions

    except Exception as e:
        _log.warning(f"[SUB-Q] Échec de la décomposition: {e}")

    return [question]


# =====================================================================
#  BGE RERANKER (Local ou API)
# =====================================================================

# Configuration du reranker BGE (API)
BGE_RERANKER_API_BASE = "https://api.dev.dassault-aviation.pro/bge-reranker-v2-m3/v1/"
BGE_RERANKER_ENDPOINT = "rerank"
BGE_RERANKER_API_KEY = "EMPTY"  # Peut être configuré si nécessaire


def rerank_with_bge(
    query: str,
    documents: List[str],
    top_k: int = None,
    http_client=None,
    log=None
) -> List[Dict[str, Any]]:
    """
    Rerank les documents en utilisant le modèle BGE Reranker (local ou API).

    Args:
        query: La question/requête
        documents: Liste des documents à reranker
        top_k: Nombre de documents à retourner (None = tous)
        http_client: Client HTTP (optionnel, sinon utilise requests)
        log: Logger optionnel

    Returns:
        Liste de dicts avec index, document, et score de pertinence
    """
    _log = log or logger

    if not documents:
        return []

    _log.info(f"[RERANK] Reranking {len(documents)} documents avec BGE Reranker API...")

    url = f"{BGE_RERANKER_API_BASE}{BGE_RERANKER_ENDPOINT}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {BGE_RERANKER_API_KEY}"
    }
    payload = {
        "query": query,
        "documents": documents
    }

    try:
        if http_client:
            resp = http_client.post(url, headers=headers, json=payload, timeout=60.0)
            resp.raise_for_status()
            response_data = resp.json()
        else:
            import requests
            resp = requests.post(url, headers=headers, json=payload, timeout=60.0)
            resp.raise_for_status()
            response_data = resp.json()

        # Parser la réponse du reranker
        # Format attendu: {"results": [{"index": 0, "relevance_score": 0.95}, ...]}
        results = response_data.get("results", [])

        if not results:
            _log.warning("[RERANK] Pas de résultats du reranker, utilisation de l'ordre original")
            return [{"index": i, "score": 1.0 - (i * 0.01)} for i in range(len(documents))]

        # Trier par score décroissant
        sorted_results = sorted(results, key=lambda x: x.get("relevance_score", 0), reverse=True)

        # Formater les résultats
        reranked = []
        for r in sorted_results:
            idx = r.get("index", 0)
            score = r.get("relevance_score", 0.0)
            reranked.append({
                "index": idx,
                "score": score,
                "document": documents[idx] if idx < len(documents) else ""
            })

        if top_k:
            reranked = reranked[:top_k]

        _log.info(f"[RERANK] ✅ Reranking API terminé. Top score: {reranked[0]['score']:.3f}" if reranked else "[RERANK] ✅ Reranking terminé")

        return reranked

    except Exception as e:
        _log.error(f"[RERANK] ❌ Erreur lors du reranking: {e}")
        # Fallback: retourner l'ordre original
        return [{"index": i, "score": 1.0 - (i * 0.01), "document": documents[i]} for i in range(len(documents))]


def apply_reranking_to_sources(
    query: str,
    sources: List[Dict[str, Any]],
    top_k: int = 30,
    http_client=None,
    log=None
) -> List[Dict[str, Any]]:
    """
    Applique le reranking BGE aux sources RAG.

    Args:
        query: La question
        sources: Liste des sources avec leurs métadonnées
        top_k: Nombre de sources à retourner
        http_client: Client HTTP optionnel
        log: Logger optionnel

    Returns:
        Sources reordonnées avec scores de reranking
    """
    _log = log or logger

    if not sources:
        return sources

    # Extraire les textes des sources
    documents = [src.get("text", "") for src in sources]

    # Appeler le reranker
    reranked = rerank_with_bge(
        query=query,
        documents=documents,
        top_k=top_k,
        http_client=http_client,
        log=_log
    )

    # Réordonner les sources selon le reranking
    reranked_sources = []
    for r in reranked:
        idx = r["index"]
        if idx < len(sources):
            source = sources[idx].copy()
            source["rerank_score"] = r["score"]
            # Mettre à jour le score principal avec le score de reranking
            source["score"] = r["score"]
            reranked_sources.append(source)

    _log.info(f"[RERANK] {len(reranked_sources)} sources reordonnées")

    return reranked_sources


# =====================================================================
#  KEYWORD EXTRACTION & FILTERING
# =====================================================================

# Mots vides français et anglais à ignorer
STOPWORDS = {
    # Français
    "le", "la", "les", "un", "une", "des", "de", "du", "au", "aux",
    "ce", "cette", "ces", "mon", "ma", "mes", "ton", "ta", "tes",
    "son", "sa", "ses", "notre", "nos", "votre", "vos", "leur", "leurs",
    "je", "tu", "il", "elle", "on", "nous", "vous", "ils", "elles",
    "qui", "que", "quoi", "dont", "où", "quel", "quelle", "quels", "quelles",
    "et", "ou", "mais", "donc", "car", "ni", "or",
    "dans", "sur", "sous", "avec", "sans", "pour", "par", "en", "vers",
    "est", "sont", "être", "avoir", "fait", "faire", "peut", "doit",
    "a", "ai", "as", "avons", "avez", "ont",
    "ne", "pas", "plus", "moins", "très", "bien", "aussi",
    "comment", "pourquoi", "quand", "combien",
    # Anglais
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "must", "shall", "can",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their",
    "this", "that", "these", "those",
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
    "and", "or", "but", "if", "then", "else", "so", "because",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "about",
    "not", "no", "yes", "all", "any", "some", "each", "every",
}


def extract_keywords(
    question: str,
    min_length: int = 3,
    log=None
) -> List[str]:
    """
    Extrait les mots-clés importants d'une question.

    Méthode simple sans dépendances:
    - Tokenisation basique
    - Suppression des stopwords
    - Garde les mots de longueur suffisante
    - Conserve les codes techniques (CS 25.xxx, AMC, etc.)

    Args:
        question: La question à analyser
        min_length: Longueur minimum des mots à garder
        log: Logger optionnel

    Returns:
        Liste des mots-clés extraits
    """
    _log = log or logger
    import re

    # Normaliser la question
    text = question.lower()

    # Extraire d'abord les codes techniques (CS 25.xxx, AMC xx.xxx, etc.)
    technical_codes = re.findall(
        r'\b(?:cs|amc|gm|cs-e|cs-apu|cs-25|cs-23|cs-27|cs-29)[\s\-]?\d+(?:\.\d+)*\b',
        text,
        re.IGNORECASE
    )
    # Normaliser les codes techniques
    technical_codes = [code.upper().replace(" ", "-") for code in technical_codes]

    # Tokeniser (garder lettres, chiffres, tirets)
    tokens = re.findall(r'[a-zA-ZÀ-ÿ0-9\-]+', text)

    # Filtrer
    keywords = []
    for token in tokens:
        token_lower = token.lower()
        # Ignorer stopwords
        if token_lower in STOPWORDS:
            continue
        # Ignorer mots trop courts (sauf si c'est un nombre/code)
        if len(token) < min_length and not token.isdigit():
            continue
        # Garder le mot
        keywords.append(token_lower)

    # Ajouter les codes techniques au début (prioritaires)
    all_keywords = technical_codes + [k for k in keywords if k.upper() not in technical_codes]

    # Dédupliquer en préservant l'ordre
    seen = set()
    unique_keywords = []
    for kw in all_keywords:
        kw_lower = kw.lower()
        if kw_lower not in seen:
            seen.add(kw_lower)
            unique_keywords.append(kw)

    _log.info(f"[KEYWORDS] Extraits: {unique_keywords[:10]}{'...' if len(unique_keywords) > 10 else ''}")

    return unique_keywords


def filter_sources_by_keywords(
    sources: List[Dict[str, Any]],
    keywords: List[str],
    min_matches: int = 1,
    log=None
) -> List[Dict[str, Any]]:
    """
    Filtre les sources qui contiennent au moins N mots-clés.

    Args:
        sources: Liste des sources avec leur texte
        keywords: Liste des mots-clés à rechercher
        min_matches: Nombre minimum de mots-clés requis
        log: Logger optionnel

    Returns:
        Sources filtrées avec un champ 'keyword_matches' ajouté
    """
    _log = log or logger

    if not keywords:
        _log.warning("[KEYWORDS] Aucun mot-clé fourni, pas de filtrage")
        return sources

    if not sources:
        return sources

    filtered = []

    for src in sources:
        text = src.get("text", "").lower()

        # Compter les mots-clés trouvés
        matches = []
        for kw in keywords:
            kw_lower = kw.lower()
            if kw_lower in text:
                matches.append(kw)

        if len(matches) >= min_matches:
            src_copy = src.copy()
            src_copy["keyword_matches"] = matches
            src_copy["keyword_count"] = len(matches)
            filtered.append(src_copy)

    _log.info(f"[KEYWORDS] Filtrage: {len(filtered)}/{len(sources)} sources retenues (min {min_matches} mot-clé)")

    # Si le filtrage est trop strict et élimine tout, garder les originaux
    if not filtered and sources:
        _log.warning("[KEYWORDS] Filtrage trop strict, conservation des sources originales")
        return sources

    return filtered


def boost_sources_by_keywords(
    sources: List[Dict[str, Any]],
    keywords: List[str],
    boost_factor: float = 0.1,
    log=None
) -> List[Dict[str, Any]]:
    """
    Booste le score des sources en fonction du nombre de mots-clés trouvés.
    Alternative au filtrage strict.

    Args:
        sources: Liste des sources
        keywords: Liste des mots-clés
        boost_factor: Facteur de boost par mot-clé trouvé
        log: Logger optionnel

    Returns:
        Sources avec scores ajustés et triées par score décroissant
    """
    _log = log or logger

    if not keywords or not sources:
        return sources

    boosted = []

    for src in sources:
        text = src.get("text", "").lower()
        src_copy = src.copy()

        # Compter les mots-clés
        matches = [kw for kw in keywords if kw.lower() in text]
        match_count = len(matches)

        # Appliquer le boost
        original_score = src_copy.get("score", 0.5)
        boost = boost_factor * match_count
        src_copy["score"] = min(1.0, original_score + boost)
        src_copy["keyword_matches"] = matches
        src_copy["keyword_boost"] = boost

        boosted.append(src_copy)

    # Trier par score décroissant
    boosted.sort(key=lambda x: x["score"], reverse=True)

    _log.info(f"[KEYWORDS] Boost appliqué: max {max(s.get('keyword_boost', 0) for s in boosted):.2f}")

    return boosted

