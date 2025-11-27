# 🚀 RaGME_UP - PROP

Système RAG (Retrieval-Augmented Generation) pour l'indexation et l'interrogation de documents techniques avec FAISS, Snowflake Arctic Embeddings et DALLEM. Inclut un système de feedback utilisateur avec re-ranking intelligent.

---

## 📖 Documentation

- **[Guide Utilisateur](GUIDE_UTILISATEUR.md)** - Documentation complète pour utiliser l'application
- **[Installation Réseau](INSTALLATION_RESEAU.md)** - Guide de déploiement multi-utilisateurs
- **[Architecture Technique](ARCHITECTURE_TECHNIQUE.md)** - Documentation technique complète (chunking, parsers, pipeline)

---

## ⚡ Démarrage rapide

### Installation

```bash
# Windows: double-cliquez sur
install.bat
```

### Lancement

```bash
# Windows: double-cliquez sur
launch.bat

# Ou manuellement
streamlit run streamlit_RAG.py
```

L'application s'ouvre automatiquement dans votre navigateur sur `http://localhost:8501`

---

## ✨ Fonctionnalités principales

- 📝 **Gestion CSV** avec interface GUI moderne
- 📥 **Ingestion documents** (PDF, DOCX, TXT) avec tracking automatique
- 🔒 **Coordination multi-utilisateurs** avec système de verrous
- 🗑️ **Purge des bases** FAISS
- ❓ **Questions RAG** avec recherche sémantique et génération de réponses
- 📝 **Feedback utilisateur** : évaluation granulaire des réponses et sources
- 🔄 **Re-ranking intelligent** : amélioration des résultats basée sur les feedbacks
- 📊 **Tableau de bord analytique** : statistiques et tendances des retours
- 👥 **Authentification** utilisateurs pour l'accès aux paramètres

---

## 🧩 Système de Chunking Avancé

Le système RAG utilise un **chunking adaptatif intelligent** qui s'adapte automatiquement au type de document et à la densité du contenu.

### Détection automatique du type de document

| Type de document | Détection | Stratégie appliquée |
|------------------|-----------|---------------------|
| **Documents EASA** | Headers `CS 25.xxx`, `AMC`, `GM` | Chunking par sections réglementaires |
| **Documents génériques** | Tout autre document | Smart chunking avec préservation de structure |

### Techniques de chunking implémentées

#### 1. 📊 Analyse de densité du contenu

Le système analyse automatiquement chaque document pour détecter sa densité :

| Densité | Caractéristiques | Taille chunk |
|---------|------------------|--------------|
| **very_dense** | Code, formules, tableaux, nombreuses références | 800 chars |
| **dense** | Texte technique, spécifications, listes | 1200 chars |
| **normal** | Texte standard, prose technique | 1500 chars |
| **sparse** | Narratif, introductions, descriptions | 2000 chars |

**Métriques analysées :**
- Densité de termes techniques (80+ mots-clés aéronautiques)
- Ratio nombres/formules
- Longueur moyenne des phrases
- Présence de listes et tableaux
- Densité de références (CS, AMC, GM, FAR, JAR)
- Ratio d'acronymes

#### 2. ✈️ Chunking EASA spécialisé

Pour les documents réglementaires EASA (CS-25, CS-E, etc.) :

```
[CS 25.571 - Damage tolerance and fatigue evaluation of structure]
The evaluation... (contenu de la section)
```

**Fonctionnalités :**
- Détection des sections par regex : `CS`, `AMC`, `GM`, `CS-E`, `CS-APU`
- Préservation du contexte `[Section ID - Title]` dans chaque chunk
- Découpage intelligent par sous-paragraphes `(a)`, `(b)`, `(1)`, `(2)`
- Fusion automatique des petites sections (<300 chars)
- Pas de redécoupage des sections déjà petites

#### 3. 📄 Smart chunking générique

Pour les documents non-EASA :

- **Préservation des headers** : Les titres restent avec leur contenu
- **Préservation des listes** : Ne coupe jamais au milieu d'une liste
- **Coupure aux phrases** : Respecte les fins de phrases
- **Contexte source** : Ajoute `[Source: filename]` pour traçabilité
- **Overlap configurable** : Chevauchement pour garder le contexte

#### 4. 🏷️ Augmentation des chunks

Chaque chunk est enrichi avec des métadonnées pour améliorer la recherche :

```python
{
    "text": "...",                    # Contenu du chunk
    "keywords": ["fatigue", "CS 25.571", "structure"],  # Mots-clés extraits
    "key_phrases": ["shall be evaluated..."],           # Phrases clés (exigences)
    "density_type": "dense",          # Type de densité
    "density_score": 0.45,            # Score de densité
    "references_to": ["CS 25.573", "AMC 25.571"]       # Références détectées
}
```

**Extraction de mots-clés :**
- Filtrage des stopwords (FR + EN)
- Bonus pour termes techniques aéronautiques
- Extraction des codes de référence (CS, AMC, GM)

#### 5. 🔗 Détection des références croisées

Le système détecte automatiquement les liens entre sections :

**Patterns détectés :**
- Références directes : `CS 25.571`, `AMC 25.1309`, `GM 25.631`
- Références contextuelles : `see CS 25.571`, `refer to AMC...`, `in accordance with...`
- Références FAR/JAR : `FAR 25.571`, `JAR 25.571`
- Références internes : `paragraph (a)`, `sub-paragraph (1)`

**Stockage :**
```python
chunk["references_to"] = ["CS 25.573", "AMC 25.571"]  # Max 5 références
```

#### 6. 🔍 Expansion de contexte (Query-Time)

Lors de la recherche, le système enrichit automatiquement les résultats :

| Fonctionnalité | Description |
|----------------|-------------|
| **Chunks voisins** | Ajoute les chunks précédent/suivant du même fichier |
| **Chunks référencés** | Si un chunk mentionne `CS 25.573`, inclut les chunks de cette section |
| **Index inversé** | Lookup rapide des chunks par référence |

**Activation :** `use_context_expansion=True` (par défaut)

### Architecture du chunking

```
Document
    │
    ▼
┌─────────────────────────────┐
│  Détection type document    │
│  (EASA vs Générique)        │
└──────────────┬──────────────┘
               │
    ┌──────────┴──────────┐
    │                     │
    ▼                     ▼
┌─────────────┐    ┌─────────────┐
│ EASA Parser │    │ Smart Chunk │
│ (sections)  │    │ (generic)   │
└──────┬──────┘    └──────┬──────┘
       │                  │
       └────────┬─────────┘
                │
                ▼
┌─────────────────────────────┐
│  Analyse densité contenu    │
│  → Adaptation taille chunks │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│     Augmentation chunks     │
│  (keywords, key_phrases)    │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Détection cross-références │
│  (CS, AMC, GM, FAR, JAR)    │
└──────────────┬──────────────┘
               │
               ▼
        Chunks indexés
```

### Fichiers concernés

| Fichier | Rôle |
|---------|------|
| `chunking.py` | Toutes les fonctions de chunking et augmentation |
| `easa_sections.py` | Parser de sections EASA (CS/AMC/GM) |
| `rag_ingestion.py` | Orchestration du chunking lors de l'ingestion |
| `rag_query.py` | Context expansion lors des requêtes |

---

## 📄 Système de Parsing Multi-Format

Le système supporte l'extraction de texte depuis de multiples formats de documents avec des stratégies de fallback robustes.

### Formats supportés

| Format | Bibliothèque principale | Fallback | Fonctionnalités spéciales |
|--------|------------------------|----------|---------------------------|
| **PDF** | pdfplumber | pdfminer.six → PyMuPDF | **Extraction tableaux**, pièces jointes, nettoyage Unicode |
| **DOCX** | python-docx | - | Tables, sections, paragraphes |
| **DOC** | - | - | ⚠️ Non supporté (convertir en .docx) |
| **XML** | xml.etree.ElementTree | - | Patterns EASA configurables |
| **TXT/MD** | Lecture native | - | Détection encodage |
| **CSV** | Lecture native | - | Extraction texte brut |

### Parser PDF (`pdf_processing.py`)

Le parser PDF est le plus sophistiqué avec une architecture à triple fallback et extraction de tableaux :

```
PDF Input
    │
    ▼
┌─────────────────────────┐
│  pdfplumber             │  ← Extraction principale (tableaux)
│  (texte + tableaux)     │
└──────────┬──────────────┘
           │ Échec ou texte suspect?
           ▼
┌─────────────────────────┐
│  pdfminer.six           │  ← Fallback 1
│  (extraction texte)     │
└──────────┬──────────────┘
           │ Échec ou texte suspect?
           ▼
┌─────────────────────────┐
│  PyMuPDF (fitz)         │  ← Fallback 2 robuste
│  (extraction fallback)  │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Extraction pièces      │  ← Automatique
│  jointes récursive      │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Nettoyage Unicode      │  ← Surrogates, encodages
│  & caractères spéciaux  │
└─────────────────────────┘
```

**Fonctionnalités clés :**
- **Extraction tableaux** : pdfplumber détecte et formate les tableaux en markdown
- **Extraction pièces jointes** : Détecte et extrait récursivement les PDF/fichiers attachés
- **Gestion Unicode** : Nettoyage automatique des caractères surrogates
- **Multi-encodage** : Détection automatique (UTF-8, UTF-16, Latin-1, ISO-8859-1, CP1252)
- **Heuristiques qualité** : Détecte si l'extraction est fiable

### Parser DOCX (`docx_processing.py`)

Extraction structurée des documents Word :

```python
# Modes d'extraction disponibles
docx_to_text(path)                    # Texte complet
extract_paragraphs_from_docx(path)    # Liste des paragraphes
extract_sections_from_docx(path)      # Sections par headers (Heading 1/2)
extract_text_from_tables(path)        # Contenu des tableaux
```

**Fonctionnalités :**
- Préservation des sauts de ligne
- Détection des styles de titres (Heading 1/2, Titre 1/2)
- Extraction des tableaux
- Normalisation des espaces

### Parser XML EASA (`xml_processing.py`)

Parser configurable pour les documents XML réglementaires :

```python
# Patterns préconfigurés
class SectionPattern(Enum):
    CS_STANDARD = r"CS[-\s]?25[.\s]?\d+"      # CS 25.101, CS-25.101
    AMC = r"AMC[-\s]?25[.\s]?\d+"              # AMC 25.101
    GM = r"GM[-\s]?25[.\s]?\d+"                # GM 25.101
    CS_E = r"CS[-\s]?E[-\s]?\d+"               # CS-E 100
    CS_APU = r"CS[-\s]?APU[-\s]?\d+"           # CS-APU 100
    ALL_EASA = r"(CS|AMC|GM)[-\s]?..."         # Tous patterns
    CUSTOM = "custom"                          # Pattern personnalisé
```

**Configuration :**
```python
XMLParseConfig(
    pattern_type=SectionPattern.ALL_EASA,
    custom_pattern=None,           # Pour pattern personnalisé
    include_section_title=True,
    min_section_length=50,
    excluded_tags=['note', 'amendment']
)
```

### Chargement unifié (`rag_ingestion.py`)

Le système détecte automatiquement le format et applique le parser approprié :

```python
def load_file_content(path, xml_configs=None):
    extension = Path(path).suffix.lower()

    if extension == '.pdf':
        return extract_text_and_attachments(path)
    elif extension in ['.docx', '.doc']:
        return docx_to_text(path)
    elif extension == '.xml':
        return parse_xml_with_config(path, xml_configs)
    elif extension in ['.txt', '.md']:
        return read_text_file(path)
    elif extension == '.csv':
        return extract_csv_text(path)
```

**Traitement parallèle :**
- ThreadPoolExecutor pour compatibilité Windows
- Nombre de workers = CPU count
- Gestion robuste des erreurs par fichier

---

## ⚙️ Configuration des répertoires

L'application nécessite plusieurs répertoires de stockage. Au premier lancement, si ces répertoires ne sont pas accessibles, une **page de configuration** s'affiche automatiquement.

### Répertoires requis

| Répertoire | Description |
|------------|-------------|
| **Bases FAISS** | Stockage des index vectoriels FAISS |
| **CSV ingestion** | Fichiers CSV pour l'ingestion de documents |
| **CSV tracking** | Fichiers de suivi des documents ingérés |
| **Feedbacks** | Stockage des feedbacks utilisateurs |

### Configuration automatique

1. Au lancement, l'application vérifie l'accessibilité de tous les répertoires
2. Si un répertoire est manquant ou inaccessible :
   - Une page de configuration s'affiche
   - Vous pouvez **créer les répertoires manquants** automatiquement
   - Ou **modifier les chemins** selon votre environnement
3. La configuration est sauvegardée dans `config.json` (fichier local, ignoré par git)

### Fichier de configuration

```json
{
  "base_root_dir": "C:\\Data\\FAISS_DATABASE\\BaseDB",
  "csv_import_dir": "C:\\Data\\FAISS_DATABASE\\CSV_Ingestion",
  "csv_export_dir": "C:\\Data\\FAISS_DATABASE\\CSV_Tracking",
  "feedback_dir": "C:\\Data\\FAISS_DATABASE\\Feedbacks"
}
```

---

## 🔧 Paramètres de Chunking

Les paramètres de chunking peuvent être ajustés dans `chunking.py` et `rag_ingestion.py` :

### Paramètres par défaut

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| `base_chunk_size` | 1000 | Taille de base avant adaptation à la densité |
| `min_chunk_size` | 200 | Taille minimale (fusion si inférieur) |
| `max_chunk_size` | 2000-2500 | Taille maximale après adaptation |
| `overlap` | 100 | Chevauchement entre chunks consécutifs |
| `merge_small_sections` | True | Fusion des sections < 300 caractères |

### Tailles adaptatives par densité

```python
CHUNK_SIZES = {
    "very_dense": 800,   # Code, formules, tableaux techniques
    "dense": 1200,       # Spécifications, listes de requirements
    "normal": 1500,      # Prose technique standard
    "sparse": 2000       # Narratif, introductions
}
```

### Personnalisation

Pour modifier le comportement par défaut, éditez `rag_ingestion.py` :

```python
# Ligne ~180
adapted_chunk_size = _get_adaptive_chunk_size(
    text,
    base_size=1000,      # Modifier ici
    min_size=600,        # Modifier ici
    max_size=2000        # Modifier ici
)
```

---

## 📋 Prérequis

- Python 3.8 ou supérieur
- Windows 10/11 (ou Linux/macOS avec adaptations)
- Accès réseau aux APIs : Snowflake (embeddings), DALLEM (LLM), BGE Reranker

---

## 🆘 Support

Consultez la documentation pour toute question :
- Questions d'utilisation → [Guide Utilisateur](GUIDE_UTILISATEUR.md)
- Installation réseau → [Installation Réseau](INSTALLATION_RESEAU.md)
- Développement/maintenance → [Architecture Technique](ARCHITECTURE_TECHNIQUE.md)

---

**Version:** 1.4
**Dernière mise à jour:** 2025-11-27
