# 📖 Guide Utilisateur RaGME_UP - PROP

**Bienvenue dans RaGME_UP - PROP !** Ce guide vous accompagne pas à pas pour utiliser l'application.

---

## 🚀 Démarrage rapide

### 1. Lancer l'application

```bash
streamlit run streamlit_RAG.py
```

L'application s'ouvre automatiquement dans votre navigateur sur `http://localhost:8501`

---

## 📋 Les 6 onglets de l'application

### 📝 **Onglet 1 : Gestion CSV**

Créez et gérez des fichiers CSV pour organiser vos documents avant l'ingestion.

#### Interface graphique moderne

Cet onglet utilise une **interface graphique de bureau (GUI)** au lieu d'une interface web, vous permettant d'accéder directement aux fichiers de votre système.

#### 🆕 Créer un nouveau CSV

1. Cliquez sur **"📝 Création d'un CSV"** dans l'onglet Gestion CSV
2. Une fenêtre GUI moderne s'ouvre automatiquement

**Méthode A : Scanner un répertoire**

1. Cliquez sur **"📂 Scanner un répertoire"**
2. Collez le chemin du dossier à scanner :
   - Ouvrez l'Explorateur Windows
   - Cliquez dans la barre d'adresse (ou `Ctrl+L`)
   - Copiez le chemin (`Ctrl+C`)
   - Collez dans le champ "Chemin du répertoire"
3. **Options de scan** :
   - ☑️ **Récursif** : inclut tous les sous-dossiers
   - **Extensions** : personnalisez les types de fichiers (PDF, DOCX, TXT par défaut)
4. Cliquez sur **"🔍 Lancer le scan"**
5. Résultat : tous les fichiers trouvés sont ajoutés à la liste

**Méthode B : Ajouter des fichiers manuellement**

1. Cliquez sur **"➕ Ajouter des fichiers"**
2. Sélectionnez un ou plusieurs fichiers dans la boîte de dialogue
3. Les fichiers sont automatiquement ajoutés à la liste

**Assigner les groupes (collections)**

- Chaque fichier peut être assigné à un groupe (= collection FAISS)
- Exemples de groupes : `CS`, `AMC`, `GM`, `ALL`
- Pour modifier un groupe : double-cliquez sur la cellule "Groupe" dans le tableau
- Pour appliquer le même groupe à plusieurs fichiers :
  1. Tapez le nom du groupe dans "Groupe par défaut"
  2. Sélectionnez les lignes dans le tableau
  3. Le groupe est appliqué automatiquement

**Sauvegarder le CSV**

1. Tapez le nom du CSV (sans extension) dans le champ "Nom du CSV" : `mes_documents`
2. Cliquez sur **"💾 Sauvegarder le CSV"**
3. Le CSV est **automatiquement sauvegardé** dans le répertoire configuré
4. Si le fichier existe déjà, une confirmation vous sera demandée
5. Le nom du CSV devient le nom de la base FAISS

**Note** : Plus besoin de choisir manuellement l'emplacement, tout est sauvegardé au bon endroit automatiquement !

#### ✏️ Modifier un CSV existant

**Option 1 : Depuis l'interface Streamlit**

1. Cliquez sur **"✏️ Gestion des CSV"** dans l'onglet Gestion CSV
2. Sélectionnez le CSV à modifier dans le menu déroulant
3. Cliquez sur **"Ouvrir pour édition"**
4. La GUI s'ouvre avec le contenu du CSV chargé

**Option 2 : Depuis l'Explorateur Windows**

1. Ouvrez l'Explorateur et naviguez vers votre répertoire CSV configuré
2. Double-cliquez sur le fichier CSV à modifier
3. La GUI s'ouvre automatiquement (si configuré)

**Édition du CSV**

1. Le contenu s'affiche dans le tableau avec :
   - ✅ Fichiers existants (chemins valides)
   - ❌ Fichiers manquants (chemins invalides) - affichés en rouge
2. Vous pouvez :
   - Modifier les groupes (double-clic sur la cellule)
   - Supprimer des lignes avec **"🗑️ Supprimer sélection"**
   - Ajouter de nouveaux fichiers avec **"➕ Ajouter des fichiers"** ou **"📂 Scanner un répertoire"**
   - Vider complètement la liste avec **"🧹 Tout effacer"**
3. Cliquez sur **"💾 Sauvegarder le CSV"** pour enregistrer les modifications

---

### 📥 **Onglet 2 : Ingestion documents**

Indexez vos documents dans FAISS pour pouvoir les interroger.

#### 🌐 Compatible partages réseau Windows

**Important** : Le système utilise FAISS pour une **parfaite compatibilité avec les partages réseau Windows** :
- ✅ Pas de problèmes de verrous de fichiers SQLite
- ✅ Synchronisation automatique sur réseau
- ✅ Multi-utilisateurs sans conflit
- ✅ Plus rapide et plus fiable

#### 📄 Ingestion via CSV (méthode recommandée)

**Pourquoi cette méthode ?**
- Gère de gros volumes facilement
- Organisée et traçable
- Évite automatiquement les doublons
- Fonctionne en multi-utilisateurs
- **Compatible réseau Windows grâce à FAISS**

**Étapes :**

1. **Préparez votre CSV** dans l'onglet "Gestion CSV" (ou utilisez un existant)

2. **Uploadez le CSV** :
   - Cliquez sur "Upload CSV contenant des chemins de fichiers"
   - Sélectionnez votre fichier `mes_documents.csv`
   - ⚠️ **Important** : Le nom du CSV = nom de la base FAISS
     - `normes_easa.csv` → base `normes_easa`
     - `manuels.csv` → base `manuels`

3. **Lancez l'ingestion** :
   - Cliquez sur **"🚀 Lancer l'ingestion"**
   - Une barre de progression s'affiche
   - Les logs détaillent chaque étape

4. **Résultats** :
   - **Nouveaux fichiers** : documents ingérés avec succès
   - **Fichiers manquants** : fichiers introuvables (chemins invalides)
   - **Déjà présents (skipped)** : fichiers déjà ingérés (évite les doublons)
   - **Pièces jointes** : pièces jointes PDF extraites et ingérées automatiquement

5. **Fichier de tracking créé** :
   - `documents_ingeres_[nom_base].csv` dans le dossier CSV
   - Permet d'éviter les doublons automatiquement lors des prochaines ingestions
   - Contient tous les fichiers déjà ingérés dans cette base

#### 🤖 Que fait l'ingestion automatiquement ?

✅ **Extraction multi-format avec fallback robuste**

| Format | Parser principal | Fallback | Fonctionnalités |
|--------|-----------------|----------|-----------------|
| **PDF** | pdfplumber | pdfminer.six → PyMuPDF | **Tableaux**, pièces jointes, nettoyage Unicode |
| **DOCX** | python-docx | - | Tables, sections, paragraphes |
| **DOC** | pywin32 (Word) | - | ✅ Conversion automatique via Microsoft Word (Windows) |
| **XML** | ElementTree | - | Patterns EASA (CS, AMC, GM) |
| **TXT/MD** | Lecture native | - | Détection encodage auto |
| **CSV** | Lecture native | - | Extraction texte brut |
| **Confluence** | API REST | - | ✅ Espaces entiers via onglet dédié |

**Fonctionnalités d'extraction :**
- **Extraction tableaux PDF** : Détection et formatage en markdown avec pdfplumber
- **Extraction pièces jointes PDF** : Détection récursive des fichiers attachés
- **Multi-encodage** : UTF-8, UTF-16, Latin-1, ISO-8859-1, CP1252
- **Nettoyage Unicode** : Suppression automatique des caractères surrogates
- **Heuristiques qualité** : Détection d'extraction défaillante

✅ **Traitement parallèle optimisé**
- ThreadPoolExecutor (compatible Windows + PyMuPDF)
- Nombre de workers = nombre de CPU
- Barre de progression en temps réel
- Gestion d'erreurs par fichier (pas d'interruption globale)

✅ **Détection EASA intelligente**
- Patterns détectés : `CS 25.xxx`, `AMC 25.xxx`, `GM 25.xxx`, `CS-E`, `CS-APU`
- Exemple : `CS 25.613 Fatigue evaluation of metallic structure`
- Métadonnées stockées : `section_id`, `section_kind`, `section_title`

✅ **Chunking adaptatif intelligent**

Le système analyse automatiquement la **densité du contenu** et adapte la taille des chunks :

| Type de contenu | Caractéristiques détectées | Taille chunk |
|-----------------|---------------------------|--------------|
| **Très dense** | Code, formules, tableaux | 800 caractères |
| **Dense** | Spécifications, listes | 1200 caractères |
| **Normal** | Prose technique | 1500 caractères |
| **Léger** | Narratif, introductions | 2000 caractères |

**Métriques analysées :**
- Densité de termes techniques (80+ mots-clés aéronautiques)
- Ratio nombres/formules
- Longueur moyenne des phrases
- Présence de listes et tableaux
- Densité de références (CS, AMC, GM, FAR, JAR)

**Règles de chunking :**
- Préservation des headers avec leur contenu
- Ne coupe jamais au milieu d'une liste
- Respecte les frontières de phrases
- Overlap de 100 caractères pour continuité
- Ajout préfixe `[Source: filename]` pour traçabilité

✅ **Augmentation sémantique des chunks**

Chaque chunk est enrichi automatiquement :
- **Mots-clés** : Top 10 termes (TF scoring + bonus technique)
- **Phrases clés** : Exigences ("shall", "must"), définitions
- **Type de densité** : very_dense, dense, normal, sparse
- **Références croisées** : CS, AMC, GM, FAR, JAR détectés (max 5)

✅ **Déduplication**
- CSV de tracking par base : `documents_ingeres_[nom_base].csv`
- Skip automatique des fichiers déjà ingérés
- Pas de doublons même sur plusieurs sessions

✅ **Stockage FAISS réseau**
- Sauvegarde automatique après chaque batch (4000 chunks)
- Compatible partages réseau Windows
- Pas de verrous SQLite
- Index vectoriel 1024 dimensions (Snowflake Arctic)

---

### 🌐 **Onglet 3 : Confluence**

Ingérez le contenu d'un espace Confluence entier directement dans le RAG.

#### 🔗 Connexion à Confluence

1. **Renseignez vos informations de connexion** :
   - **URL Confluence** : `https://votre-entreprise.atlassian.net` (Cloud) ou URL de votre serveur
   - **Nom d'utilisateur** : votre email ou identifiant
   - **Mot de passe / Token API** :
     - Pour Confluence Cloud : créez un token API dans vos paramètres Atlassian
     - Pour Confluence Server : utilisez votre mot de passe

2. **Testez la connexion** :
   - Cliquez sur **"🔗 Tester la connexion"**
   - ✅ Si réussi : affiche votre nom d'utilisateur
   - ❌ Si échoué : vérifiez vos identifiants

#### 📁 Sélection de l'espace

1. **Liste des espaces** : une liste déroulante affiche tous les espaces accessibles
2. **Saisie manuelle** : entrez directement la clé de l'espace (ex: `PROJ`, `DOC`)
3. **Info espace** : affiche le nom et la description de l'espace sélectionné

#### ⚙️ Configuration de l'ingestion

| Paramètre | Description |
|-----------|-------------|
| **Base FAISS cible** | Sélectionnez la base où stocker le contenu |
| **Nom de la collection** | Par défaut = clé de l'espace (ex: `proj`) |
| **Reconstruire** | ✅ Recommandé pour mise à jour hebdomadaire (supprime l'existant) |

#### 🚀 Lancer l'ingestion

1. Cliquez sur **"🚀 Ingérer l'espace Confluence"**
2. **Progression affichée** :
   - Extraction des pages depuis Confluence
   - Conversion HTML → texte
   - Chunking et embedding
   - Stockage dans FAISS
3. **Résumé final** : nombre de pages et chunks créés

#### ⏰ Automatisation hebdomadaire

Pour une synchronisation automatique :
1. Créez un script Python utilisant `confluence_processing.py`
2. Planifiez-le avec le **Planificateur de tâches Windows** ou **cron**

> 💡 **Astuce** : L'option "Reconstruire" est idéale pour les mises à jour hebdomadaires car elle garantit une synchronisation complète.

---

### 🗑️ **Onglet 4 : Purge des bases**

Supprimez tout le contenu d'une base (les collections sont vidées mais pas supprimées).

#### ⚠️ Attention : Action irréversible !

**Étapes :**

1. **Sélectionnez la base** à purger dans le menu déroulant

2. **Consultez les statistiques** :
   - Nombre de collections
   - Total de chunks indexés
   - Détail par collection
   - CSV de tracking associé

3. **Confirmez la purge** :
   - Tapez **exactement** le nom de la base : `normes_easa`
   - Le bouton **"🗑️ PURGER LA BASE"** devient actif

4. **Cliquez sur PURGER LA BASE** :
   - Toutes les collections sont vidées
   - Le CSV de tracking est supprimé
   - Un résumé détaillé s'affiche

5. **Rechargez la page** pour voir les changements

**Quand utiliser la purge ?**
- Vous voulez réinitialiser complètement une base
- Vous avez ingéré de mauvaises données
- Vous voulez repartir de zéro avec une nouvelle organisation

---

### ❓ **Onglet 5 : Questions RAG**

Posez des questions sur vos documents indexés et obtenez des réponses contextuelles.

#### 🎯 Sélection de la base et collection

**En haut de l'onglet** :

1. **Sélectionnez une base** dans le menu déroulant :
   - Liste toutes les bases FAISS disponibles
   - Exemple : `normes_easa`, `manuels`, etc.

2. **Sélectionnez une collection** dans le menu déroulant :
   - `CS` : seulement les Certification Specifications
   - `AMC` : seulement les Acceptable Means of Compliance
   - `GM` : seulement les Guidance Material
   - `ALL` : toutes les collections (recherche globale)

#### 💬 Poser une question

**Étapes :**

1. **Tapez votre question** dans la zone de texte :
   - Exemple : *"What are the fatigue evaluation requirements for CS 25?"*
   - Soyez précis et clair
   - Utilisez des termes techniques présents dans vos documents

2. **Cliquez sur "🤖 Poser la question"**

3. **Résultat** :
   - 🧠 **Réponse** du LLM basée sur vos documents
   - 📚 **Sources** citées avec :
     - 🟢 Score élevé (≥ 0.8) = très pertinent
     - 🟠 Score moyen (0.6-0.8) = pertinent
     - 🔴 Score faible (< 0.6) = peu pertinent
   - 📄 **Bouton "Ouvrir"** pour ouvrir le document source dans son application par défaut
   - 🧩 Contexte brut utilisé (pour debug)

#### 📂 Ouvrir les documents sources

Le bouton **"Ouvrir"** à côté de chaque source permet d'ouvrir directement le fichier dans son application par défaut (Adobe Reader pour PDF, Word pour DOCX, etc.).

**Avantages** :
- ✅ Vérifiez la source dans son contexte complet
- ✅ Les résultats de recherche restent affichés (pas d'effacement)
- ✅ Ouverture automatique dans l'application appropriée

#### 🔍 Comprendre les sources

Chaque source affiche :
- **Nom du fichier** : `CS_25.pdf`
- **Chunk ID** : identifiant du morceau de texte
- **Score** : pertinence (0 = pas pertinent, 1 = très pertinent)
- **Distance** : distance L2 FAISS (plus petit = meilleur)
- **Section EASA** : si détectée (ex: `CS 25.613`)
- **Mots-clés** : termes techniques extraits du chunk
- **Références** : sections CS/AMC/GM mentionnées dans le chunk
- **Passage utilisé** : le texte exact récupéré de vos documents

#### 🔗 Expansion de contexte automatique

Le système enrichit automatiquement les résultats de recherche :

| Fonctionnalité | Description |
|----------------|-------------|
| **Chunks voisins** | Inclut le chunk précédent/suivant du même fichier |
| **Sections référencées** | Si un chunk mentionne `CS 25.573`, inclut les chunks de cette section |
| **Index inversé** | Lookup rapide O(1) des chunks par référence |

Cela permet d'obtenir plus de contexte sans multiplier les requêtes vectorielles.

#### 🔄 Amélioration par retours utilisateurs (Re-ranking)

Une option **"🔄 Utiliser les retours utilisateurs pour améliorer les résultats"** permet d'activer le re-ranking intelligent :

- **Sources bien notées** : les sources ayant reçu de bons feedbacks sont favorisées
- **Sources mal notées** : les sources ayant reçu de mauvais feedbacks sont pénalisées
- **Questions similaires** : si une question similaire a déjà été posée et évaluée, le système utilise cette information pour améliorer les résultats

> 💡 Plus vous donnez de feedbacks, plus le système s'améliore !

#### 📝 Donner votre avis (Feedback simplifié)

Après chaque réponse, deux boutons apparaissent :

- **👍 Oui** : La réponse vous a aidé
- **👎 Non** : La réponse n'est pas satisfaisante

**Si vous cliquez 👎 :**
Un champ texte s'affiche pour décrire la **réponse que vous attendiez**. Cette information est précieuse pour améliorer les futures recherches !

> 💡 Plus vous donnez de feedbacks, plus le système s'améliore pour tous les utilisateurs !

#### 💾 Cache local pour performances réseau

Si vous travaillez sur un réseau lent, le système propose un **cache local** pour accélérer les requêtes.

**Dans la barre latérale (sidebar)** :

1. **Section "💾 Cache local"** : affiche l'état du cache pour chaque collection
2. **Bouton "📥 Copier local"** : copie la collection sélectionnée en local
3. **Utilisation automatique** : une fois le cache créé, il est utilisé automatiquement

**Fonctionnement :**

| Situation | Comportement |
|-----------|--------------|
| **Pas de cache** | Requêtes via réseau (plus lent) |
| **Cache créé** | Requêtes en local (ultra-rapide) |
| **Cache obsolète** | Bannière d'avertissement + requêtes via réseau |

**⚠️ Avertissement cache obsolète :**

Si la base réseau a été modifiée (nouvelle ingestion), une bannière orange s'affiche :

> ⚠️ **Cache local obsolète** - La base a été modifiée sur le réseau. Cliquez sur "📥 Copier local" pour mettre à jour. Les requêtes utilisent actuellement la base réseau.

**Actions recommandées :**
- Cliquez sur **"📥 Copier local"** pour mettre à jour le cache
- Ou ignorez : les requêtes continueront via le réseau

> 💡 Le cache est stocké dans `~/.cache/ragme_up/` et est automatiquement invalidé après chaque ingestion locale.

---

### 📊 **Onglet 6 : Tableau de bord analytique**

Visualisez les statistiques et tendances des retours utilisateurs.

#### 📊 Filtres

- **Base à analyser** : sélectionnez une base spécifique ou "Toutes les bases"
- **Période d'analyse** : 7, 14, 30, 60 ou 90 derniers jours

#### 📈 Métriques globales

- **Total feedbacks** : nombre total de feedbacks enregistrés
- **👍 Positifs** : nombre de réponses jugées utiles
- **Taux de satisfaction** : pourcentage de feedbacks positifs

#### 📉 Graphiques de tendance

- **Évolution des feedbacks** : graphique en barres montrant les feedbacks positifs et négatifs par jour

#### 📋 Statistiques détaillées

- **Satisfaction par collection** : tableau avec les feedbacks 👍/👎 par collection
- **Questions avec feedback négatif** : liste des questions où les utilisateurs ont cliqué 👎, avec la réponse attendue
- **Activité par utilisateur** : répartition des feedbacks par utilisateur

#### 📥 Export des données

- **Exporter en CSV** : téléchargez tous les feedbacks au format CSV
- **Rafraîchir les statistiques** : mettez à jour les données affichées

---

## ❓ FAQ - Questions fréquentes

### Installation et Réseau

**Q : Puis-je utiliser l'application sur un partage réseau Windows ?**
- ✅ **Oui !** FAISS est conçu pour fonctionner parfaitement sur réseau
- ✅ Pas de problèmes de verrous de fichiers
- ✅ Plusieurs utilisateurs peuvent travailler simultanément
- ✅ Synchronisation automatique des fichiers

**Q : Où sont stockées mes données ?**
- Configuré dans `streamlit_RAG.py` (lignes 48-51)
- Par défaut sur partage réseau : `N:\...\FAISS_DATABASE\`
- Bases FAISS : `BaseDB\[nom_base]`
- CSV tracking : `Fichiers_Tracking_CSV\documents_ingeres_[nom_base].csv`
- CSV ingestion : `CSV_Ingestion\[nom].csv`

### Ingestion

**Q : Les pièces jointes PDF sont-elles gérées ?**
- ✅ **Oui, automatiquement !**
- Extraction et ingestion des fichiers joints (PDF, images, etc.)
- Gestion des noms de fichiers avec caractères spéciaux
- Extensions préservées automatiquement

**Q : Que se passe-t-il avec des PDFs contenant des caractères spéciaux ?**
- ✅ Gestion automatique des caractères Unicode surrogates
- ✅ Nettoyage des noms de fichiers invalides
- ✅ Préservation des extensions (.pdf, .docx, etc.)

**Q : Pourquoi l'ingestion utilise des threads au lieu de processus ?**
- Meilleure compatibilité Windows avec PyMuPDF
- Pas de MemoryError ou crashes de workers
- Chargement parallèle toujours actif et performant

### Performance

**Q : FAISS est-il rapide ?**
- ✅ **Oui, très rapide !**
- Recherche vectorielle optimisée
- Pas de couche SQLite (overhead réduit)
- Bonne scalabilité

**Q : Combien de documents puis-je indexer ?**
- Pas de limite théorique
- Testé avec plusieurs milliers de documents
- Performance stable même sur partage réseau

**Q : Le réseau est lent, comment accélérer ?**
- ✅ **Utilisez le cache local !**
- Dans la sidebar, cliquez sur "📥 Copier local" pour la collection
- Le cache est automatiquement utilisé pour toutes les requêtes suivantes
- Si la base réseau change, un avertissement vous invite à mettre à jour

**Q : Où est stocké le cache local ?**
- Dans `~/.cache/ragme_up/` (dossier utilisateur)
- Structure miroir de la base réseau
- Fichier `.hash` pour validation automatique

### Chunking et Parsing

**Q : Comment fonctionne le chunking adaptatif ?**
- Le système analyse automatiquement la **densité du contenu**
- Documents denses (code, formules) → chunks plus petits (800 car.)
- Documents légers (narratif) → chunks plus grands (2000 car.)
- Métriques : termes techniques, ratio numérique, longueur phrases

**Q : Quels formats de documents sont supportés ?**
- **PDF** : pdfplumber (tableaux) + pdfminer.six + PyMuPDF fallback + pièces jointes
- **DOCX** : python-docx avec extraction tables et sections
- **DOC** : ✅ Conversion automatique via Microsoft Word (Windows uniquement)
- **XML** : Parser EASA configurable (CS, AMC, GM, CS-E, CS-APU)
- **TXT/MD/CSV** : Lecture native avec détection encodage
- **Confluence** : ✅ Ingestion via API REST (onglet dédié)

**Q : Les sections EASA sont-elles détectées automatiquement ?**
- ✅ **Oui !** Patterns détectés : `CS 25.xxx`, `AMC`, `GM`, `CS-E`, `CS-APU`
- Chaque chunk conserve : section_id, section_kind, section_title
- Préfixe de contexte ajouté : `[CS 25.571 - Damage tolerance...]`

**Q : Comment sont extraites les références croisées ?**
- Patterns détectés : `see CS 25.571`, `refer to AMC...`, `in accordance with...`
- Références FAR/JAR : `FAR 25.571`, `JAR 25.571`
- Références internes : `paragraph (a)`, `sub-paragraph (1)`
- Max 5 références stockées par chunk

### Confluence

**Q : Comment ingérer un espace Confluence ?**
- Utilisez l'onglet "🌐 Confluence" dans l'application
- Renseignez URL, identifiant et mot de passe/token
- Sélectionnez l'espace à ingérer
- Cliquez sur "🚀 Ingérer l'espace Confluence"

**Q : Quelle authentification pour Confluence Cloud ?**
- Utilisez un **Token API** (pas votre mot de passe)
- Créez-le dans : Paramètres Atlassian → Sécurité → Tokens API
- Votre identifiant = votre email

**Q : Comment automatiser la synchronisation Confluence ?**
- Créez un script Python utilisant `confluence_processing.py`
- Planifiez-le avec le Planificateur de tâches Windows (hebdomadaire)
- Utilisez l'option "Reconstruire" pour une synchronisation complète

### Requêtes

**Q : Comment fonctionne la distance dans FAISS ?**
- FAISS utilise la distance L2 (euclidienne)
- Plus petit score = plus pertinent

---

## 🆘 Besoin d'aide ?

### Logs

Les logs détaillés sont dans : `rag_da_debug.log`

Consultez-les en cas d'erreur pour voir ce qui s'est passé.

### Contact

Pour toute question ou problème, contactez l'équipe de développement RaGME_UP - PROP.

---

## 🎯 Workflow recommandé

### Pour démarrer un nouveau projet

1. **Organisez vos documents** dans un ou plusieurs dossiers
2. **Créez un CSV** via l'onglet "Gestion CSV"
3. **Lancez l'ingestion** via l'onglet "Ingestion documents"
4. **Posez vos questions** via l'onglet "Questions RAG"

### Pour ajouter des documents à une base existante

1. **Créez un CSV** avec uniquement les nouveaux fichiers
2. **Nommez-le comme la base existante** : `ma_base.csv`
3. **Lancez l'ingestion** : les doublons seront skippés automatiquement

### Travail en équipe sur réseau

1. **Configurez les chemins réseau** dans `streamlit_RAG.py`
2. **Partagez le répertoire FAISS** avec droits lecture/écriture
3. **Chaque utilisateur** peut ingérer et requêter simultanément
4. **Les requêtes RAG** peuvent être faites en parallèle sans problème

---

## 🆕 Nouveautés de cette version (v1.6)

### 💾 Cache local automatique (NOUVEAU)
- 📥 **Bouton "Copier local"** : copie la base FAISS en local pour performances optimales
- 🔄 **Utilisation automatique** : une fois créé, le cache est utilisé sans intervention
- ✅ **Validation automatique** : le cache est comparé à la base réseau à chaque requête
- ⚠️ **Avertissement si obsolète** : bannière orange si la base réseau a changé
- 🌐 **Fallback réseau** : si le cache est obsolète, les requêtes passent par le réseau

### 📊 Extraction de tableaux PDF
- 📋 **pdfplumber** : détection automatique des tableaux dans les PDF
- 📝 **Formatage markdown** : tableaux formatés avec colonnes alignées
- 🔄 **Triple fallback** : pdfplumber → pdfminer.six → PyMuPDF

### ⚡ Améliorations de performance
- 🚀 **Cache Streamlit** : réponses instantanées pour requêtes répétées (30 min)
- 📦 **BATCH_SIZE optimisé** : 32 (équilibre performance/sécurité)
- 🔒 **Troncature automatique** : textes > 28000 chars tronqués (limite Snowflake)
- 💾 **Cache FAISS** : stores cachés 10 min pour chargement rapide

### 🌐 APIs uniquement
- ✅ **Snowflake** : embeddings (snowflake-arctic-embed-l-v2.0)
- ✅ **DALLEM** : génération de réponses (dallem-val)
- ✅ **BGE Reranker** : re-ranking intelligent (bge-reranker-v2-m3)
- ❌ Modèles locaux supprimés (simplification)

### 🧠 Chunking Adaptatif Intelligent
- 📊 **Analyse de densité** : détection automatique du type de contenu
- 📏 **Taille adaptative** : 800-2000 caractères selon densité
- 🏷️ **Augmentation sémantique** : mots-clés, phrases clés, références
- 🔗 **Références croisées** : détection CS, AMC, GM, FAR, JAR

### 📄 Parsing Multi-Format
- **PDF** : pdfplumber (tableaux) + pdfminer.six + PyMuPDF + pièces jointes
- **DOCX** : python-docx avec tables, sections, paragraphes
- **DOC** : ✅ Conversion automatique via Microsoft Word (Windows)
- **XML** : Parser EASA configurable (CS, AMC, GM, CS-E, CS-APU)

### 🌐 Ingestion Confluence (NOUVEAU)
- 🔗 **Connexion API** : Support Cloud (atlassian.net) et Server
- 📁 **Espaces entiers** : ingestion de toutes les pages d'un espace
- 🔄 **Conversion HTML→texte** : tableaux, listes, headers préservés
- ⏰ **Automatisation** : idéal pour synchronisation hebdomadaire
- 🎨 **Interface dédiée** : onglet "🌐 Confluence" dans l'application

### 📝 Système de feedback utilisateur
- 👍👎 **Feedback rapide** : un simple clic pouce haut ou pouce bas
- 💡 **Réponse attendue** : champ pour indiquer la réponse souhaitée si 👎
- 📊 **Tableau de bord** : taux de satisfaction et questions problématiques

### FAISS
- ✨ **FAISS** pour une meilleure compatibilité réseau Windows
- 🚀 **Rapide** : recherche vectorielle optimisée
- 🌐 **Compatible réseau** : pas de problèmes de verrous
- 💾 **Auto-save** : sauvegarde après chaque ajout

### Corrections critiques
- 🐛 Fix erreur pdfminer StringIO (encode)
- 🐛 Fix erreur token limit Snowflake (8192 max)
- 🐛 Fix validation longueurs dans FAISS
- 🐛 Fix caractères surrogates dans noms de fichiers

---

**Bon RAG avec RaGME_UP - PROP ! 🚀**
