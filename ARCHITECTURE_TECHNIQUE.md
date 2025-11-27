# Architecture Technique - RaGME_UP PROP

Documentation technique complète du système RAG pour développeurs et mainteneurs.

---

## Table des matières

1. [Vue d'ensemble](#vue-densemble)
2. [Structure du projet](#structure-du-projet)
3. [Pipeline d'ingestion](#pipeline-dingestion)
4. [Système de Chunking](#système-de-chunking)
5. [Parsers de documents](#parsers-de-documents)
6. [Stockage vectoriel FAISS](#stockage-vectoriel-faiss)
7. [Pipeline de requête](#pipeline-de-requête)
8. [Configuration](#configuration)
9. [API et modèles](#api-et-modèles)
10. [Dépendances](#dépendances)

---

## Vue d'ensemble

RaGME_UP PROP est un système RAG (Retrieval-Augmented Generation) conçu pour les documents techniques aéronautiques, avec support spécialisé pour les réglementations EASA (CS, AMC, GM).

### Architecture globale

```
┌─────────────────────────────────────────────────────────────────┐
│                        STREAMLIT UI                              │
│  (streamlit_RAG.py, csv_generator_gui.py)                       │
└─────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   INGESTION     │  │    REQUÊTE      │  │   FEEDBACK      │
│ rag_ingestion.py│  │  rag_query.py   │  │ feedback_store  │
└────────┬────────┘  └────────┬────────┘  └─────────────────┘
         │                    │
         ▼                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                      TRAITEMENT DOCUMENTS                        │
│  pdf_processing.py | docx_processing.py | xml_processing.py     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        CHUNKING                                  │
│  chunking.py | easa_sections.py                                 │
│  - Analyse densité adaptative                                   │
│  - Détection sections EASA                                      │
│  - Augmentation sémantique                                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EMBEDDINGS & STOCKAGE                         │
│  models_utils.py | faiss_store.py                               │
│  - Snowflake Arctic API (1024 dims)                             │
│  - FAISS index par collection                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Structure du projet

```
RAG_v3/
├── streamlit_RAG.py          # Application Streamlit principale (98 KB)
├── csv_generator_gui.py      # GUI CustomTkinter pour CSV (33 KB)
│
├── # === INGESTION ===
├── rag_ingestion.py          # Orchestration ingestion (22 KB)
├── pdf_processing.py         # Parser PDF (31 KB)
├── docx_processing.py        # Parser DOCX (3.4 KB)
├── xml_processing.py         # Parser XML EASA (11 KB)
├── csv_processing.py         # Parser CSV (0.7 KB)
│
├── # === CHUNKING ===
├── chunking.py               # Chunking adaptatif (64 KB, 1865 lignes)
├── easa_sections.py          # Détection sections EASA (4.3 KB)
│
├── # === REQUÊTE ===
├── rag_query.py              # Pipeline de requête (22 KB)
├── advanced_search.py        # Recherche avancée (23 KB)
│
├── # === STOCKAGE ===
├── faiss_store.py            # Gestion FAISS (12 KB)
├── feedback_store.py         # Stockage feedbacks (24 KB)
│
├── # === CONFIGURATION ===
├── config_manager.py         # Gestion chemins (14 KB)
├── models_utils.py           # Embeddings/LLM (31 KB)
│
├── # === SCRIPTS ===
├── install.bat               # Installation Windows
├── launch.bat                # Lancement Windows
│
├── # === DOCUMENTATION ===
├── README.md
├── GUIDE_UTILISATEUR.md
├── INSTALLATION_RESEAU.md
├── ARCHITECTURE_TECHNIQUE.md # Ce document
│
├── # === CONFIGURATION (générés) ===
├── config.json               # Configuration utilisateur
├── requirements.txt          # Dépendances Python
└── .gitignore
```

---

## Pipeline d'ingestion

### Flux de données

```
Fichiers sources (PDF, DOCX, XML, TXT, CSV)
          │
          ▼
┌─────────────────────────────────────────┐
│  1. CHARGEMENT PARALLÈLE               │
│  ThreadPoolExecutor (CPU count workers) │
│  load_file_content()                    │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  2. PARSING PAR FORMAT                  │
│  - PDF: pdfplumber (tableaux) + fallbacks│
│  - DOCX: python-docx                    │
│  - XML: ElementTree + patterns EASA     │
│  - TXT/MD/CSV: lecture native           │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  3. DÉTECTION TYPE DOCUMENT             │
│  detect_sections() → EASA ou Générique  │
└────────────────┬────────────────────────┘
                 │
       ┌─────────┴─────────┐
       ▼                   ▼
┌─────────────┐     ┌─────────────┐
│ EASA Parser │     │ Smart Chunk │
│ sections    │     │ générique   │
└──────┬──────┘     └──────┬──────┘
       │                   │
       └─────────┬─────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  4. ANALYSE DENSITÉ                     │
│  _calculate_content_density()           │
│  → very_dense, dense, normal, sparse    │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  5. CHUNKING ADAPTATIF                  │
│  _get_adaptive_chunk_size()             │
│  chunk_easa_sections() ou               │
│  smart_chunk_generic()                  │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  6. AUGMENTATION                        │
│  augment_chunks()                       │
│  - Mots-clés (TF scoring)               │
│  - Phrases clés (shall/must/defined)    │
│  - Densité (type + score)               │
│  add_cross_references_to_chunks()       │
│  - Références CS/AMC/GM/FAR/JAR         │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  7. EMBEDDING                           │
│  Snowflake Arctic API (1024 dims)       │
│  Batch size: 32, max 28000 chars/texte  │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  8. STOCKAGE FAISS                      │
│  Batch de 4000 chunks                   │
│  index.faiss + metadata.json            │
└─────────────────────────────────────────┘
```

### Fonction principale

```python
# rag_ingestion.py
def ingest_documents(
    file_paths: List[str],
    db_path: str,
    collection_name: str,
    chunk_size: int = 1000,         # Taille de base
    use_easa_sections: bool = True, # Détection EASA
    xml_configs: dict = None,       # Config XML optionnelle
    progress_callback = None        # Callback progression
) -> dict:
    """
    Retourne:
    {
        'ingested': [...],      # Fichiers ingérés
        'skipped': [...],       # Fichiers déjà présents
        'missing': [...],       # Fichiers introuvables
        'attachments': [...],   # Pièces jointes extraites
        'errors': [...]         # Erreurs rencontrées
    }
    """
```

---

## Système de Chunking

### Architecture du chunking (`chunking.py`)

Le fichier `chunking.py` (1865 lignes) implémente un système de chunking sophistiqué.

### 1. Analyse de densité

```python
# Dictionnaire des tailles par densité
CHUNK_SIZES = {
    "very_dense": 800,   # Code, formules, tableaux
    "dense": 1200,       # Spécifications, listes
    "normal": 1500,      # Prose technique
    "sparse": 2000       # Narratif, introductions
}

def _calculate_content_density(text: str) -> dict:
    """
    Analyse le contenu et retourne:
    {
        'technical_term_ratio': float,  # Ratio termes techniques
        'numeric_ratio': float,         # Ratio nombres/formules
        'avg_sentence_length': float,   # Longueur moyenne phrases
        'list_density': float,          # Densité listes/tableaux
        'reference_density': float,     # Densité références
        'acronym_ratio': float,         # Ratio acronymes
        'bracket_density': float,       # Densité parenthèses
        'density_type': str,            # Type résultant
        'density_score': float          # Score 0-1
    }
    """
```

**Mots-clés techniques (80+)** :
```python
TECHNICAL_TERMS = {
    'aircraft', 'airplane', 'aeroplane', 'structure', 'load', 'stress',
    'fatigue', 'damage', 'tolerance', 'inspection', 'maintenance',
    'certification', 'compliance', 'requirement', 'airworthiness',
    'fuel', 'engine', 'propeller', 'system', 'electrical', 'hydraulic',
    'pneumatic', 'avionics', 'flight', 'control', 'landing', 'gear',
    'wing', 'fuselage', 'empennage', 'nacelle', 'pylon', ...
}
```

### 2. Chunking EASA spécialisé

```python
def chunk_easa_sections(
    sections: List[dict],
    max_chunk_size: int = 1500,
    min_chunk_size: int = 200,
    merge_small_sections: bool = True,
    add_context_prefix: bool = True
) -> List[dict]:
    """
    Chunks les sections EASA détectées.

    Format entrée:
    {
        'section_id': 'CS 25.571',
        'section_kind': 'CS',
        'title': 'Damage tolerance and fatigue evaluation',
        'content': '...'
    }

    Format sortie (chunk):
    {
        'text': '[CS 25.571 - Damage tolerance...]\nContenu...',
        'source_file': 'CS-25.pdf',
        'chunk_id': 'CS 25.571__chunk_0',
        'section_id': 'CS 25.571',
        'section_kind': 'CS',
        'section_title': 'Damage tolerance...',
        'is_complete_section': True/False
    }
    """
```

**Patterns de détection** (`easa_sections.py`) :
```python
EASA_PATTERNS = {
    'CS': r'CS[-\s]?25[.\s]?\d+',
    'AMC': r'AMC[-\s]?25[.\s]?\d+',
    'GM': r'GM[-\s]?25[.\s]?\d+',
    'CS-E': r'CS[-\s]?E[-\s]?\d+',
    'CS-APU': r'CS[-\s]?APU[-\s]?\d+'
}
```

### 3. Chunking générique intelligent

```python
def smart_chunk_generic(
    text: str,
    source_file: str,
    chunk_size: int = 1500,
    min_chunk_size: int = 200,
    overlap: int = 100,
    add_source_prefix: bool = True,
    preserve_lists: bool = True,
    preserve_headers: bool = True
) -> List[dict]:
    """
    Chunking intelligent pour documents non-EASA.

    Détection structure via _detect_document_structure():
    - Headers numérotés: "1.1 Overview", "2.3.1 Details"
    - Headers capitalisés: "INTRODUCTION", "REQUIREMENTS"
    - Listes à puces: "- item", "* item", "• item"
    - Listes numérotées: "1. item", "a) item", "(i) item"

    Règles:
    - Headers restent avec leur contenu
    - Listes ne sont jamais coupées
    - Coupure aux fins de phrases (. ! ?)
    - Overlap ajouté après headers/sources
    """
```

### 4. Chunking parent-enfant

```python
def create_parent_child_chunks(
    text: str,
    source_file: str,
    parent_size: int = 3000,
    child_size: int = 800,
    child_overlap: int = 100
) -> Tuple[List[dict], List[dict]]:
    """
    Crée une hiérarchie de chunks:
    - Parent: contexte large (3000+ chars)
    - Children: fragments précis (800 chars)

    Retourne: (parent_chunks, child_chunks)
    Chaque child a un 'parent_id' pour liaison.
    """
```

### 5. Augmentation des chunks

```python
def augment_chunks(
    chunks: List[dict],
    add_keywords: bool = True,
    add_key_phrases: bool = True,
    add_density_info: bool = True
) -> List[dict]:
    """
    Enrichit chaque chunk avec métadonnées.
    """

def _extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """
    Extraction TF-based avec:
    - Filtrage stopwords FR+EN
    - Bonus 2x pour termes techniques aéronautiques
    - Bonus 1.3x pour mots > 8 caractères
    - Extraction codes références (CS, AMC, GM, FAR, JAR)
    """

def _extract_key_phrases(text: str, max_phrases: int = 5) -> List[str]:
    """
    Extraction phrases clés:
    - Requirements: contient "shall", "must"
    - Définitions: contient "means", "is defined as"
    """
```

### 6. Références croisées

```python
def extract_cross_references(text: str) -> List[dict]:
    """
    Détecte les références inter-sections.

    Patterns:
    - Direct: 'CS 25.571', 'AMC 25.1309'
    - Contexte: 'see CS...', 'refer to AMC...', 'in accordance with...'
    - FAR/JAR: 'FAR 25.571', 'JAR 25.571'
    - Interne: 'paragraph (a)', 'sub-paragraph (1)'

    Retourne:
    [{
        'ref_type': 'CS' | 'AMC' | 'GM' | 'FAR' | 'JAR' | 'internal',
        'ref_id': 'CS 25.571',
        'normalized_id': 'cs_25_571',
        'position': 42
    }, ...]
    """

def add_cross_references_to_chunks(chunks: List[dict], max_refs: int = 5):
    """
    Ajoute chunk['references_to'] = ['CS 25.573', 'AMC 25.571', ...]
    """

def build_reference_index(chunks: List[dict]) -> dict:
    """
    Index inversé pour lookup O(1):
    {
        'cs_25_571': [chunk1, chunk2, ...],
        'amc_25_1309': [chunk3, ...],
        ...
    }
    """
```

### 7. Expansion de contexte (query-time)

```python
def expand_chunk_context(
    chunk: dict,
    all_chunks: List[dict],
    reference_index: dict = None,
    include_neighbors: bool = True,
    include_references: bool = True
) -> dict:
    """
    Enrichit un chunk avec contexte additionnel.

    Ajoute:
    - chunk['context_before']: chunk précédent (même fichier)
    - chunk['context_after']: chunk suivant (même fichier)
    - chunk['referenced_sections']: chunks des sections référencées
    """

def get_neighboring_chunks(
    chunk: dict,
    all_chunks: List[dict],
    before: int = 1,
    after: int = 1
) -> Tuple[List[dict], List[dict]]:
    """
    Retourne les chunks voisins du même fichier source.
    """

def expand_search_results(
    results: List[dict],
    all_chunks: List[dict],
    reference_index: dict = None
) -> List[dict]:
    """
    Enrichit tous les résultats de recherche avec contexte.
    """
```

---

## Parsers de documents

### Parser PDF (`pdf_processing.py`)

**Architecture à triple fallback avec extraction de tableaux :**

```python
def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extraction robuste avec détection de tableaux.
    Ordre: pdfplumber → pdfminer.six → PyMuPDF

    Les tableaux sont formatés en markdown avec alignement.
    """

def _extract_with_pdfplumber(path: str) -> str:
    """Extraction via pdfplumber (principal, avec tableaux)"""

def _extract_text_with_pdfminer(path: str) -> str:
    """Extraction via pdfminer.six (fallback 1)"""

def _extract_text_with_pymupdf(path: str) -> str:
    """Extraction via PyMuPDF (fallback 2)"""

def extract_attachments(pdf_path: str) -> List[dict]:
    """Extraction récursive des pièces jointes"""

def clean_filename(filename: str) -> str:
    """
    Nettoyage nom de fichier:
    - Suppression caractères surrogates
    - Préservation extension (.pdf, .docx, etc.)
    - Caractères invalides → underscore
    """

def detect_and_clean_surrogates(text: str) -> str:
    """
    Nettoyage Unicode:
    - Détection surrogates (0xD800-0xDFFF)
    - Essai encodages: UTF-8, UTF-16, Latin-1, ISO-8859-1, CP1252
    - Suppression caractères non-imprimables
    """
```

### Parser DOCX (`docx_processing.py`)

```python
def docx_to_text(path: str) -> str:
    """Extraction texte complet (paragraphes joints par \n)"""

def extract_paragraphs_from_docx(path: str) -> List[str]:
    """Liste des paragraphes individuels"""

def extract_sections_from_docx(path: str) -> List[dict]:
    """
    Découpage par headers (Heading 1/2, Titre 1/2).
    Retourne: [{'title': '...', 'content': '...', 'level': 1}, ...]
    """

def extract_text_from_tables(path: str) -> str:
    """Extraction du contenu des tableaux"""

def normalize_whitespace(text: str) -> str:
    """Normalisation espaces (préserve sauts de ligne)"""
```

### Parser XML EASA (`xml_processing.py`)

```python
from enum import Enum

class SectionPattern(Enum):
    CS_STANDARD = r"CS[-\s]?25[.\s]?\d+"
    AMC = r"AMC[-\s]?25[.\s]?\d+"
    GM = r"GM[-\s]?25[.\s]?\d+"
    CS_E = r"CS[-\s]?E[-\s]?\d+"
    CS_APU = r"CS[-\s]?APU[-\s]?\d+"
    ALL_EASA = r"(CS|AMC|GM)[-\s]?(25|E|APU)[-\s.]?\d+"
    CUSTOM = "custom"

@dataclass
class XMLParseConfig:
    pattern_type: SectionPattern = SectionPattern.ALL_EASA
    custom_pattern: str = None
    include_section_title: bool = True
    min_section_length: int = 50
    excluded_tags: List[str] = field(default_factory=lambda: ['note'])

def parse_xml_document(
    path: str,
    config: XMLParseConfig = None
) -> List[dict]:
    """
    Parse XML avec config.
    Retourne sections: [{'section_id': '...', 'content': '...', ...}, ...]
    """

def extract_text_recursive(element, excluded_tags: List[str] = None) -> str:
    """Extraction texte récursive avec gestion namespaces"""
```

---

## Stockage vectoriel FAISS

### Architecture (`faiss_store.py`)

```python
class FAISSStore:
    """
    Gestionnaire de stockage FAISS.

    Structure fichiers:
    base_path/
    └── collection_name/
        ├── index.faiss      # Index vectoriel FAISS
        └── metadata.json    # Métadonnées chunks
    """

    def __init__(self, base_path: str):
        self.base_path = base_path
        self.dimension = 1024  # Snowflake Arctic

    def add_documents(
        self,
        collection: str,
        texts: List[str],
        embeddings: np.ndarray,
        metadatas: List[dict]
    ) -> List[str]:
        """Ajoute documents et retourne IDs"""

    def search(
        self,
        collection: str,
        query_embedding: np.ndarray,
        k: int = 10
    ) -> List[Tuple[dict, float]]:
        """Recherche k plus proches voisins"""

    def delete_collection(self, collection: str) -> bool:
        """Supprime une collection"""

    def get_collection_stats(self, collection: str) -> dict:
        """Statistiques: count, dimension, etc."""
```

### Format des métadonnées

```json
{
  "ids": ["chunk_id_1", "chunk_id_2", ...],
  "documents": [
    {
      "id": "CS 25.571__chunk_0__abc123",
      "text": "[CS 25.571 - Damage tolerance...]\n...",
      "source_file": "CS-25.pdf",
      "path": "/path/to/CS-25.pdf",
      "chunk_id": "CS 25.571__chunk_0",
      "section_id": "CS 25.571",
      "section_kind": "CS",
      "section_title": "Damage tolerance and fatigue evaluation",
      "language": "en",
      "keywords": ["fatigue", "damage", "tolerance", "structure"],
      "key_phrases": ["shall be evaluated by analysis..."],
      "density_type": "dense",
      "density_score": 0.62,
      "references_to": ["CS 25.573", "AMC 25.571"],
      "is_complete_section": true
    },
    ...
  ]
}
```

---

## Pipeline de requête

### Flux de recherche (`rag_query.py`)

```
Question utilisateur
        │
        ▼
┌───────────────────────────────┐
│  1. EXPANSION DE REQUÊTE      │
│  (optionnel, via DALLEM API)  │
│  - Synonymes                  │
│  - Reformulations             │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│  2. EMBEDDING                 │
│  Snowflake Arctic API         │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│  3. RECHERCHE FAISS           │
│  k=10 plus proches voisins    │
│  Distance L2                  │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│  4. EXPANSION CONTEXTE        │
│  - Chunks voisins             │
│  - Sections référencées       │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│  5. RE-RANKING (optionnel)    │
│  - Feedbacks utilisateurs     │
│  - BGE Reranker API           │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│  6. GÉNÉRATION RÉPONSE        │
│  DALLEM API                   │
│  Prompt + contexte            │
└───────────────────────────────┘
```

### Fonction principale

```python
def query_rag(
    question: str,
    db_path: str,
    collection: str,
    k: int = 10,
    use_reranking: bool = False,
    use_feedback: bool = False,
    expand_context: bool = True
) -> dict:
    """
    Retourne:
    {
        'answer': 'Réponse générée...',
        'sources': [
            {
                'source_file': 'CS-25.pdf',
                'chunk_id': 'CS 25.571__chunk_0',
                'score': 0.85,
                'distance': 0.234,
                'text': '...',
                'section_id': 'CS 25.571',
                'keywords': [...],
                'references_to': [...]
            },
            ...
        ],
        'context_used': '...',  # Debug
        'expanded_context': [...] # Chunks voisins/référencés
    }
    """
```

---

## Configuration

### Fichier config.json

```json
{
  "base_root_dir": "C:\\Data\\FAISS_DATABASE\\BaseDB",
  "csv_import_dir": "C:\\Data\\FAISS_DATABASE\\CSV_Ingestion",
  "csv_export_dir": "C:\\Data\\FAISS_DATABASE\\CSV_Tracking",
  "feedback_dir": "C:\\Data\\FAISS_DATABASE\\Feedbacks"
}
```

### Gestionnaire de configuration (`config_manager.py`)

```python
@dataclass
class StorageConfig:
    base_root_dir: str
    csv_import_dir: str
    csv_export_dir: str
    feedback_dir: str

def load_config() -> StorageConfig:
    """Charge config.json ou retourne valeurs par défaut"""

def save_config(config: StorageConfig) -> bool:
    """Sauvegarde dans config.json"""

def validate_directory(path: str) -> Tuple[bool, str]:
    """Vérifie existence et permissions écriture"""

def validate_all_directories(config: StorageConfig) -> dict:
    """Valide tous les chemins, retourne erreurs"""
```

---

## API et modèles

Le système utilise exclusivement des APIs externes pour les modèles d'IA.

### Embeddings (`models_utils.py`)

```python
# Configuration
EMBED_MODEL = "snowflake-arctic-embed-l-v2.0"
BATCH_SIZE = 32  # Équilibre performance/sécurité
MAX_CHARS_PER_TEXT = 28000  # Limite ~7000 tokens (Snowflake max: 8192)

# Snowflake Arctic (API)
class DirectOpenAIEmbeddings:
    """Client embeddings via API Snowflake (1024 dimensions)"""

def embed_in_batches(texts, role, batch_size, emb_client, log) -> np.ndarray:
    """
    Embedding par batch avec troncature automatique.
    Les textes > 28000 chars sont tronqués pour éviter les erreurs de tokens.
    """
```

### LLM pour génération

```python
# DALLEM (API)
LLM_MODEL = "dallem-val"

def call_dallem_chat(http_client, question, context, log) -> str:
    """Génération via API DALLEM avec retry automatique"""
```

### Reranker

```python
# BGE Reranker (API)
BGE_RERANKER_API_BASE = "https://api.dev.dassault-aviation.pro/bge-reranker-v2-m3"
BGE_RERANKER_ENDPOINT = "/v1/rerank"

def rerank_with_bge(query: str, documents: List[str], top_k: int) -> List[dict]:
    """Re-ranking via API BGE Reranker"""
```

---

## Dépendances

### requirements.txt

```
# Interface
streamlit>=1.28.0
customtkinter>=5.2.0

# FAISS
faiss-cpu>=1.7.4

# Parsing PDF (avec extraction tableaux)
pdfminer.six>=20221105
pymupdf>=1.24.0
pdfplumber>=0.10.0  # Extraction améliorée des tableaux
python-docx>=0.8.11

# Traitement texte
langdetect>=1.0.9
chardet>=5.1.0
unidecode>=1.3.6

# API (Snowflake, DALLEM, BGE Reranker)
openai>=1.0.0
httpx>=0.24.0
requests>=2.31.0

# Core
numpy>=1.24.0
packaging>=23.0
```

### Bibliothèques natives Python utilisées

- `xml.etree.ElementTree` - Parsing XML
- `pathlib.Path` - Manipulation chemins
- `re` - Expressions régulières
- `json` - Sérialisation
- `concurrent.futures` - Parallélisation
- `dataclasses` - Classes de données
- `enum` - Énumérations
- `typing` - Annotations de type

---

## Logs et debug

### Fichier de logs

```
rag_da_debug.log
```

Contient :
- Extraction PDF (succès/fallback/erreurs)
- Traitement Unicode/surrogates
- Chunking (nombre chunks, densité détectée)
- Ingestion FAISS (batches, IDs)
- Requêtes (résultats, scores)
- Erreurs réseau

### Activation debug verbose

Dans `streamlit_RAG.py`, modifier le niveau de logging :

```python
logging.basicConfig(
    level=logging.DEBUG,  # INFO par défaut
    filename='rag_da_debug.log'
)
```

---

**Version:** 1.4
**Dernière mise à jour:** 2025-11-27
