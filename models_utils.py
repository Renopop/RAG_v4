# models_utils.py
import os
import sys
import math
import time
import traceback
from typing import List, Optional

import numpy as np
import httpx
import logging
from logging import Logger

from openai import OpenAI
import openai

# ═══════════════════════════════════════════════════════════════════
# ⚠️ BLOC MODE TEST LOCAL - À SUPPRIMER APRÈS LES TESTS ⚠️
# ═══════════════════════════════════════════════════════════════════
# Variables globales pour le mode test local
USE_LOCAL_MODELS = False
LOCAL_EMBEDDING_PATH = None
LOCAL_LLM_PATH = None
LOCAL_RERANKER_PATH = None
_local_embedding_model = None
_local_llm_model = None
_local_reranker_model = None


def set_local_mode(use_local: bool, embedding_path: str = None, llm_path: str = None, reranker_path: str = None):
    """Configure le mode test local avec les chemins des modèles."""
    global USE_LOCAL_MODELS, LOCAL_EMBEDDING_PATH, LOCAL_LLM_PATH, LOCAL_RERANKER_PATH
    USE_LOCAL_MODELS = use_local
    LOCAL_EMBEDDING_PATH = embedding_path
    LOCAL_LLM_PATH = llm_path
    LOCAL_RERANKER_PATH = reranker_path


def get_local_embedding_model():
    """Charge et retourne le modèle d'embedding local (sentence-transformers)."""
    global _local_embedding_model
    if _local_embedding_model is None and LOCAL_EMBEDDING_PATH:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise RuntimeError(
                "sentence-transformers non installé. Installez-le avec:\n"
                "  pip install sentence-transformers transformers torch"
            )
        try:
            import torch
        except ImportError:
            raise RuntimeError(
                "torch non installé. Installez-le avec:\n"
                "  pip install torch\n"
                "Pour CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu121"
            )
        try:
            # Détection automatique du device (CUDA si disponible)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _local_embedding_model = SentenceTransformer(LOCAL_EMBEDDING_PATH, device=device)
            print(f"✅ Modèle d'embedding local chargé: {LOCAL_EMBEDDING_PATH} (device: {device})")
        except Exception as e:
            raise RuntimeError(f"Erreur lors du chargement du modèle d'embedding: {e}")
    return _local_embedding_model


def get_local_reranker_model():
    """Charge et retourne le modèle de reranking local (cross-encoder)."""
    global _local_reranker_model
    if _local_reranker_model is None and LOCAL_RERANKER_PATH:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise RuntimeError(
                "sentence-transformers non installé. Installez-le avec:\n"
                "  pip install sentence-transformers"
            )
        try:
            import torch
        except ImportError:
            raise RuntimeError(
                "torch non installé. Installez-le avec:\n"
                "  pip install torch"
            )
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _local_reranker_model = CrossEncoder(LOCAL_RERANKER_PATH, device=device)
            print(f"✅ Modèle reranker local chargé: {LOCAL_RERANKER_PATH} (device: {device})")
        except Exception as e:
            raise RuntimeError(f"Erreur lors du chargement du modèle reranker: {e}")
    return _local_reranker_model


def rerank_local(query: str, documents: List[str], top_k: int = None) -> List[dict]:
    """
    Rerank des documents avec le modèle local.

    Args:
        query: La requête de recherche
        documents: Liste des textes de documents à reranker
        top_k: Nombre de résultats à retourner (None = tous)

    Returns:
        Liste de dicts avec 'index', 'score', 'text' triés par score décroissant
    """
    model = get_local_reranker_model()
    if model is None:
        raise RuntimeError("Aucun modèle reranker local configuré")

    # Créer les paires query-document
    pairs = [(query, doc) for doc in documents]

    # Calculer les scores
    scores = model.predict(pairs)

    # Créer la liste des résultats
    results = []
    for idx, (score, doc) in enumerate(zip(scores, documents)):
        results.append({
            "index": idx,
            "score": float(score),
            "text": doc
        })

    # Trier par score décroissant
    results.sort(key=lambda x: x["score"], reverse=True)

    # Limiter si top_k spécifié
    if top_k is not None:
        results = results[:top_k]

    return results


def get_local_llm_model():
    """Charge et retourne le modèle LLM local."""
    global _local_llm_model
    if _local_llm_model is None and LOCAL_LLM_PATH:
        try:
            # Détection du type de modèle (GGUF ou HuggingFace)
            if LOCAL_LLM_PATH.endswith('.gguf'):
                # Utiliser llama-cpp-python pour les modèles GGUF
                try:
                    from llama_cpp import Llama

                    # Tenter d'activer le GPU (n_gpu_layers=-1 = toutes les couches sur GPU)
                    # Si CUDA n'est pas disponible, llama-cpp utilisera le CPU
                    try:
                        _local_llm_model = Llama(
                            model_path=LOCAL_LLM_PATH,
                            n_ctx=2048,
                            n_threads=4,
                            n_gpu_layers=-1,  # Utiliser GPU si disponible
                            verbose=False
                        )
                        print(f"✅ Modèle LLM GGUF chargé: {LOCAL_LLM_PATH} (GPU activé)")
                    except Exception:
                        # Fallback CPU si GPU non disponible
                        _local_llm_model = Llama(
                            model_path=LOCAL_LLM_PATH,
                            n_ctx=2048,
                            n_threads=4,
                            n_gpu_layers=0,
                            verbose=False
                        )
                        print(f"✅ Modèle LLM GGUF chargé: {LOCAL_LLM_PATH} (CPU)")
                except ImportError:
                    raise RuntimeError(
                        "llama-cpp-python non installé. Installez-le avec: pip install llama-cpp-python"
                    )
            else:
                # Utiliser transformers pour les modèles HuggingFace
                try:
                    from transformers import AutoModelForCausalLM, AutoTokenizer
                except ImportError:
                    raise RuntimeError(
                        "transformers non installé. Installez-le avec:\n"
                        "  pip install transformers torch accelerate"
                    )
                try:
                    import torch
                except ImportError:
                    raise RuntimeError(
                        "torch non installé. Installez-le avec:\n"
                        "  pip install torch"
                    )
                try:
                    import accelerate
                except ImportError:
                    raise RuntimeError(
                        "accelerate non installé (requis pour device_map='auto'). Installez-le avec:\n"
                        "  pip install accelerate"
                    )
                try:
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    tokenizer = AutoTokenizer.from_pretrained(LOCAL_LLM_PATH)
                    model = AutoModelForCausalLM.from_pretrained(
                        LOCAL_LLM_PATH,
                        device_map="auto" if device == "cuda" else None,
                        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                        low_cpu_mem_usage=True
                    )
                    _local_llm_model = {"model": model, "tokenizer": tokenizer, "device": device}
                    print(f"✅ Modèle LLM HuggingFace chargé: {LOCAL_LLM_PATH} (device: {device})")
                except Exception as e:
                    raise RuntimeError(f"Erreur lors du chargement du modèle LLM HuggingFace: {e}")
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"Erreur lors du chargement du modèle LLM: {e}")
    return _local_llm_model


def call_local_llm(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.7
) -> str:
    """
    Appelle le LLM local avec un prompt système et utilisateur.

    Args:
        system_prompt: Instructions système
        user_prompt: Prompt utilisateur
        max_tokens: Nombre maximum de tokens à générer
        temperature: Température de génération

    Returns:
        Texte généré par le modèle
    """
    llm = get_local_llm_model()
    if llm is None:
        raise RuntimeError("Aucun modèle LLM local configuré")

    if isinstance(llm, dict):  # HuggingFace model
        model = llm["model"]
        tokenizer = llm["tokenizer"]
        device = llm["device"]

        prompt = f"{system_prompt}\n\n{user_prompt}"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)

        if device == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items()}

        import torch
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        input_length = inputs['input_ids'].shape[1]
        new_tokens = outputs[0][input_length:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    else:  # GGUF model (llama-cpp-python)
        prompt = f"{system_prompt}\n\n{user_prompt}\n\nAssistant:"
        output = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["User:", "\n\n"],
            echo=False
        )
        response = output["choices"][0]["text"].strip()

    return response
# ═══════════════════════════════════════════════════════════════════
# FIN BLOC MODE TEST LOCAL
# ═══════════════════════════════════════════════════════════════════


# ---------------------------------------------------------------------
#  CONFIG réseau / modèles
# ---------------------------------------------------------------------

LLM_MODEL = "dallem-val"
EMBED_MODEL = "snowflake-arctic-embed-l-v2.0"
BATCH_SIZE = 16  # taille batch embeddings

HARDCODE = {
    "DALLEM_API_BASE": "https://api.dev.dassault-aviation.pro/dallem-pilote/v1",
    "SNOWFLAKE_API_BASE": "https://api.dev.dassault-aviation.pro/snowflake-arctic-embed-l-v2.0/v1",
    "DALLEM_API_KEY": "EMPTY",     # à surcharger par l'env
    "SNOWFLAKE_API_KEY": "token",  # à surcharger par l'env
    "DISABLE_SSL_VERIFY": "true",
}

DALLEM_API_BASE = os.getenv("DALLEM_API_BASE", HARDCODE["DALLEM_API_BASE"]).rstrip("/")
SNOWFLAKE_API_BASE = os.getenv("SNOWFLAKE_API_BASE", HARDCODE["SNOWFLAKE_API_BASE"]).rstrip("/")
DALLEM_API_KEY = os.getenv("DALLEM_API_KEY", HARDCODE["DALLEM_API_KEY"])
SNOWFLAKE_API_KEY = os.getenv("SNOWFLAKE_API_KEY", HARDCODE["SNOWFLAKE_API_KEY"])

VERIFY_SSL = not (
    os.getenv("DISABLE_SSL_VERIFY", HARDCODE["DISABLE_SSL_VERIFY"])
    .lower()
    in ("1", "true", "yes", "on")
)


def _mask(s: Optional[str]) -> str:
    if not s:
        return "<vide>"
    if len(s) <= 6:
        return "***"
    return s[:3] + "…" + s[-3:]


def make_logger(debug: bool) -> Logger:
    log = logging.getLogger("rag_da")

    # Choix des niveaux : console silencieuse en mode non-debug
    if debug:
        level_console = logging.DEBUG
        level_logger = logging.DEBUG
    else:
        level_console = logging.WARNING
        level_logger = logging.WARNING

    log.setLevel(level_logger)

    # Si le logger a déjà des handlers, on met juste à jour leurs niveaux
    if log.handlers:
        for h in log.handlers:
            if isinstance(h, logging.StreamHandler):
                h.setLevel(level_console)
        return log

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level_console)
    ch.setFormatter(fmt)

    # Fichier : on garde tout en DEBUG pour analyse détaillée
    fh = logging.FileHandler("rag_da_debug.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    log.addHandler(ch)
    log.addHandler(fh)

    # Ces logs seront visibles au moins dans le fichier
    log.info("=== Configuration RAG_DA ===")
    log.info(f"SNOWFLAKE_API_BASE = {SNOWFLAKE_API_BASE}")
    log.info(f"DALLEM_API_BASE    = {DALLEM_API_BASE}")
    log.info(f"VERIFY_SSL         = {VERIFY_SSL}")
    log.info(f"EMBED_MODEL        = {EMBED_MODEL}")
    log.info(f"BATCH_SIZE         = {BATCH_SIZE}")
    log.info(
        "API_KEYS           = snowflake={} | dallem={}".format(
            _mask(SNOWFLAKE_API_KEY),
            _mask(DALLEM_API_KEY),
        )
    )
    return log

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if debug else logging.INFO)
    ch.setFormatter(fmt)

    fh = logging.FileHandler("rag_da_debug.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    log.addHandler(ch)
    log.addHandler(fh)

    log.info("=== Configuration RAG_DA ===")
    log.info(f"SNOWFLAKE_API_BASE = {SNOWFLAKE_API_BASE}")
    log.info(f"DALLEM_API_BASE    = {DALLEM_API_BASE}")
    log.info(f"VERIFY_SSL         = {VERIFY_SSL}")
    log.info(f"EMBED_MODEL        = {EMBED_MODEL}")
    log.info(f"BATCH_SIZE         = {BATCH_SIZE}")
    log.info(
        "API_KEYS           = snowflake={} | dallem={}".format(
            _mask(SNOWFLAKE_API_KEY),
            _mask(DALLEM_API_KEY),
        )
    )
    return log


def create_http_client() -> httpx.Client:
    """
    Client HTTP configuré (timeout, SSL) pour Snowflake + DALLEM.
    """
    return httpx.Client(
        verify=VERIFY_SSL,
        timeout=httpx.Timeout(300.0),
    )


# ---------------------------------------------------------------------
#  Client embeddings Snowflake (OpenAI v1-compatible)
# ---------------------------------------------------------------------


class DirectOpenAIEmbeddings:
    """
    Client embeddings minimal (OpenAI v1-compatible).
    role_prefix=True -> "passage:" / "query:".
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        http_client: Optional[httpx.Client] = None,
        role_prefix: bool = True,
        logger: Optional[Logger] = None,
    ):
        self.model = model
        self.role_prefix = role_prefix
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=http_client,
        )
        self.log = logger or logging.getLogger("rag_da")

    def _apply_prefix(self, items: List[str], role: str) -> List[str]:
        if not self.role_prefix:
            return items
        pref = "query: " if role == "query" else "passage: "
        return [pref + (x or "") for x in items]

    def _retry_request(self, func, max_retries: int = 5, base_delay: float = 1.0):
        """
        Exécute func() avec retry exponentiel sur les erreurs transitoires.
        """
        for attempt in range(max_retries):
            try:
                return func()
            except (openai.APIConnectionError, openai.RateLimitError, openai.APIError) as e:
                if attempt == max_retries - 1:
                    self.log.error(
                        f"[embeddings] Échec après {max_retries} tentatives — {type(e).__name__}: {e}"
                    )
                    raise
                wait_time = base_delay * (2 ** attempt)
                self.log.warning(
                    f"[embeddings] Tentative {attempt + 1}/{max_retries} échouée "
                    f"({type(e).__name__}: {e}) — retry dans {wait_time:.1f}s"
                )
                time.sleep(wait_time)

    def _create_embeddings(self, inputs: List[str]) -> List[List[float]]:
        t0 = time.time()
        self.log.debug(
            f"[embeddings] POST {self.client.base_url} | model={self.model} "
            f"| n_inputs={len(inputs)} | len0={len(inputs[0]) if inputs else 0}"
        )

        def _do_request():
            return self.client.embeddings.create(model=self.model, input=inputs)

        try:
            resp = self._retry_request(_do_request)
            dur = (time.time() - t0) * 1000
            self.log.debug(
                f"[embeddings] OK in {dur:.1f} ms | items={len(resp.data)} "
                f"| dim≈{len(resp.data[0].embedding) if resp.data else 'n/a'}"
            )
            return [d.embedding for d in resp.data]

        except openai.NotFoundError as e:
            self.log.error(f"[embeddings] NotFoundError (modèle='{self.model}' ?) : {e}")
            self.log.debug(traceback.format_exc())
            raise
        except openai.AuthenticationError as e:
            self.log.error("[embeddings] AuthenticationError — clé invalide ?")
            self.log.debug(traceback.format_exc())
            raise
        except Exception as e:
            self.log.error(f"[embeddings] Exception — {e}")
            self.log.debug(traceback.format_exc())
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        inputs = self._apply_prefix(list(texts or []), role="passage")
        return self._create_embeddings(inputs)

    def embed_queries(self, texts: List[str]) -> List[List[float]]:
        inputs = self._apply_prefix(list(texts or []), role="query")
        return self._create_embeddings(inputs)


def embed_in_batches(
    texts: List[str],
    role: str,
    batch_size: int,
    emb_client: DirectOpenAIEmbeddings,
    log: Logger,
    dry_run: bool = False,
) -> np.ndarray:
    """
    Découpe en batches, appelle le client embeddings, normalise les vecteurs (L2).
    """
    # ═══════════════════════════════════════════════════════════════════
    # ⚠️ BLOC MODE TEST LOCAL - À SUPPRIMER APRÈS LES TESTS ⚠️
    # ═══════════════════════════════════════════════════════════════════
    if USE_LOCAL_MODELS and LOCAL_EMBEDDING_PATH:
        log.info(f"[emb] MODE LOCAL activé | modèle={LOCAL_EMBEDDING_PATH} | n={len(texts)}")
        try:
            model = get_local_embedding_model()
            embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            M = np.asarray(embeddings, dtype=np.float32)

            # Normalisation L2
            denom = np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
            if np.any(np.isnan(denom)):
                log.warning("[emb] NaN détecté dans la norme, correction appliquée.")
                denom = np.nan_to_num(denom, nan=1.0)
            M = M / denom

            log.info(f"[emb] MODE LOCAL terminé | shape={M.shape}")
            return M
        except Exception as e:
            log.error(f"[emb] Erreur mode local: {e}")
            raise
    # ═══════════════════════════════════════════════════════════════════
    # FIN BLOC MODE TEST LOCAL
    # ═══════════════════════════════════════════════════════════════════

    out: List[List[float]] = []
    n = len(texts)
    log.info(
        f"[emb] start role={role} | n={n} | batch_size={batch_size} | dry_run={dry_run}"
    )
    for i in range(0, n, batch_size):
        chunk = texts[i: i + batch_size]
        log.debug(
            f"[emb] chunk {i // batch_size + 1}/{math.ceil(n / max(1, batch_size))} "
            f"| size={len(chunk)} | first='{(chunk[0][:120] if chunk else '')}'"
        )
        try:
            if dry_run:
                dim = 1024
                fake = np.random.rand(len(chunk), dim).astype(np.float32) - 0.5
                out.extend(fake.tolist())
            else:
                if role == "query":
                    out.extend(emb_client.embed_queries(chunk))
                else:
                    out.extend(emb_client.embed_documents(chunk))
        except Exception as e:
            log.error(f"[emb] échec sur le chunk (i={i}) — {e}")
            log.debug(traceback.format_exc())
            raise

    M = np.asarray(out, dtype=np.float32)
    if M.ndim != 2 or M.shape[0] != n:
        log.error(f"[emb] shape inattendue: {M.shape} (attendu ({n}, d))")

    denom = np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
    if np.any(np.isnan(denom)):
        log.warning("[emb] NaN détecté dans la norme, correction appliquée.")
        denom = np.nan_to_num(denom, nan=1.0)
    M = M / denom
    log.info(
        f"[emb] terminé | shape={M.shape} | d={M.shape[1] if M.ndim == 2 else 'n/a'}"
    )
    return M


# ---------------------------------------------------------------------
#  Appel LLM DALLEM
# ---------------------------------------------------------------------


def call_dallem_chat(
    http_client: httpx.Client,
    question: str,
    context: str,
    log: Logger,
) -> str:
    """
    Appel simple au LLM DALLEM via /chat/completions.
    """
    # ═══════════════════════════════════════════════════════════════════
    # ⚠️ BLOC MODE TEST LOCAL - À SUPPRIMER APRÈS LES TESTS ⚠️
    # ═══════════════════════════════════════════════════════════════════
    if USE_LOCAL_MODELS and LOCAL_LLM_PATH:
        log.info(f"[LLM] MODE LOCAL activé | modèle={LOCAL_LLM_PATH}")
        try:
            llm = get_local_llm_model()

            system_msg = (
                "You are an expert assistant in aeronautical regulations. "
                "You must answer based on the CONTEXT provided below. "
                "The context contains extracts from normative documents (CS, AMC, GM). "
                "Always cite the references (CS xx.xxx, AMC, etc.) present in the context."
            )

            import textwrap
            user_msg = textwrap.dedent(f"""
            === DOCUMENTARY CONTEXT ===
            {context}
            === END OF CONTEXT ===

            QUESTION: {question}

            INSTRUCTIONS:
            - Answer in English based on the information from the context above
            - Cite the normative references (CS, AMC, GM) mentioned in the context
            - If the context contains relevant information, use it to answer
            - Only if the context contains NO relevant information, respond: "I do not have the information to answer your question."
            """)

            # Génération selon le type de modèle
            if isinstance(llm, dict):  # HuggingFace model
                model = llm["model"]
                tokenizer = llm["tokenizer"]
                device = llm["device"]

                prompt = f"{system_msg}\n\n{user_msg}"
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)

                if device == "cuda":
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                import torch
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=512,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )

                # Extraire uniquement les nouveaux tokens générés (sans le prompt)
                input_length = inputs['input_ids'].shape[1]
                new_tokens = outputs[0][input_length:]
                response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            else:  # GGUF model (llama-cpp-python)
                prompt = f"{system_msg}\n\n{user_msg}\n\nAssistant:"
                output = llm(
                    prompt,
                    max_tokens=512,
                    temperature=0.7,
                    stop=["User:", "\n\n"],
                    echo=False
                )
                response = output["choices"][0]["text"].strip()

            log.info(f"[LLM] MODE LOCAL terminé | longueur réponse={len(response)}")
            return response

        except Exception as e:
            log.error(f"[LLM] Erreur mode local: {e}")
            log.debug(traceback.format_exc())
            raise
    # ═══════════════════════════════════════════════════════════════════
    # FIN BLOC MODE TEST LOCAL
    # ═══════════════════════════════════════════════════════════════════

    if not DALLEM_API_KEY or DALLEM_API_KEY == "toto":
        raise RuntimeError("DALLEM_API_KEY manquant ou de test. Impossible d'utiliser le LLM.")

    system_msg = (
        "Tu es un assistant expert en réglementation aéronautique. "
        "Tu dois répondre en te basant sur le CONTEXTE fourni ci-dessous. "
        "Le contexte contient des extraits de documents normatifs (CS, AMC, GM). "
        "Cite toujours les références (CS xx.xxx, AMC, etc.) présentes dans le contexte."
    )

    import textwrap
    user_msg = textwrap.dedent(f"""
    === CONTEXTE DOCUMENTAIRE ===
    {context}
    === FIN DU CONTEXTE ===

    QUESTION : {question}

    INSTRUCTIONS :
    - Réponds en anglais en te basant sur les informations du contexte ci-dessus
    - Cite les références normatives (CS, AMC, GM) mentionnées dans le contexte
    - Si le contexte contient des informations pertinentes, utilise-les pour répondre
    - Seulement si le contexte ne contient AUCUNE information pertinente, réponds : "I do not have the information to answer your question."
    """)

    url = DALLEM_API_BASE.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {DALLEM_API_KEY}",
        "Content-Type": "application/json",
    }

    # Log du contexte pour diagnostic
    log.info(f"[RAG] Contexte: {len(context)} chars, {context.count('[source=')} sources")
    if not context.strip():
        log.warning("[RAG] ⚠️ CONTEXTE VIDE - pas de chunks trouvés!")

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": 2000,
        "temperature": 0.3,
    }

    log.info("[RAG] Appel DALLEM /chat/completions pour réponse RAG")

    # Retry logic avec backoff exponentiel
    max_retries = 4
    base_delay = 2  # secondes

    last_error = None
    for attempt in range(max_retries):
        try:
            resp = http_client.post(url, headers=headers, json=payload, timeout=180.0)
            resp.raise_for_status()
            data = resp.json()

            # Vérifier que la réponse contient bien du contenu
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            if not content:
                raise ValueError("Réponse LLM vide")

            log.info(f"[RAG] ✅ Réponse DALLEM reçue (attempt {attempt + 1}/{max_retries})")
            return content

        except Exception as e:
            last_error = e
            delay = base_delay * (2 ** attempt)  # 2, 4, 8, 16 secondes

            if attempt < max_retries - 1:
                log.warning(f"[RAG] ⚠️ Erreur DALLEM (attempt {attempt + 1}/{max_retries}): {e}")
                log.info(f"[RAG] Retry dans {delay}s...")
                time.sleep(delay)
            else:
                log.error(f"[RAG] ❌ Échec DALLEM après {max_retries} tentatives: {e}")

    # Toutes les tentatives ont échoué - retourner un message d'erreur spécial
    error_msg = (
        "⚠️ **ERREUR DE COMMUNICATION AVEC LE LLM**\n\n"
        f"Le serveur n'a pas pu répondre après {max_retries} tentatives.\n\n"
        f"**Erreur technique:** {str(last_error)[:200]}\n\n"
        "👉 **Veuillez reposer votre question** ou réessayer dans quelques instants."
    )
    return error_msg
