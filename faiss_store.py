"""
FAISS-based vector store
Optimisé pour les partages réseau Windows (pas de SQLite)

Architecture:
- Un dossier par base de données (db_path)
- Dans chaque base: sous-dossiers pour chaque collection
- Dans chaque collection: index.faiss + metadata.json

Avantages:
- Pas de SQLite (pas de problèmes de verrouillage réseau)
- Fichiers simples qui se synchronisent bien
- Rapide
- Compatible partages réseau Windows
"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Optional
import logging

# Import FAISS
try:
    import faiss
except ImportError:
    raise ImportError(
        "FAISS n'est pas installé. Installez-le avec:\n"
        "  pip install faiss-cpu\n"
        "ou pour GPU:\n"
        "  pip install faiss-gpu"
    )

logger = logging.getLogger(__name__)


class FaissCollection:
    """Collection FAISS pour stocker et rechercher des embeddings"""

    def __init__(self, collection_path: str, name: str, dimension: int = 1024):
        """
        Args:
            collection_path: Chemin du dossier de la collection
            name: Nom de la collection
            dimension: Dimension des embeddings (1024 pour Snowflake Arctic)
        """
        self.collection_path = collection_path
        self.name = name
        self.dimension = dimension

        # Chemins des fichiers
        self.index_path = os.path.join(collection_path, "index.faiss")
        self.metadata_path = os.path.join(collection_path, "metadata.json")

        # Créer le dossier si nécessaire
        os.makedirs(collection_path, exist_ok=True)

        # Charger ou créer l'index FAISS
        if os.path.exists(self.index_path):
            logger.info(f"[FAISS] Loading existing index: {self.index_path}")
            self.index = faiss.read_index(self.index_path)
        else:
            logger.info(f"[FAISS] Creating new index with dimension {dimension}")
            # IndexFlatL2 = recherche exhaustive avec distance L2
            # Pour de grosses bases, on pourrait utiliser IndexIVFFlat
            self.index = faiss.IndexFlatL2(dimension)

        # Charger ou créer les métadonnées
        if os.path.exists(self.metadata_path):
            logger.info(f"[FAISS] Loading metadata: {self.metadata_path}")
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.metadata = data.get("metadata", [])
                self.ids = data.get("ids", [])
        else:
            logger.info("[FAISS] Creating new metadata store")
            self.metadata = []
            self.ids = []

    def add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Ajoute des documents avec leurs embeddings à la collection.

        Args:
            ids: Liste des IDs uniques
            embeddings: Liste des vecteurs embeddings
            documents: Liste des textes des documents
            metadatas: Liste optionnelle de métadonnées (dicts)
        """
        # Vérifier que toutes les listes ont la même longueur
        if not (len(ids) == len(embeddings) == len(documents)):
            raise ValueError(
                f"ids, embeddings et documents doivent avoir la même taille. "
                f"Reçu: ids={len(ids)}, embeddings={len(embeddings)}, documents={len(documents)}"
            )

        if metadatas is None:
            metadatas = [{} for _ in ids]

        if len(metadatas) != len(ids):
            raise ValueError("metadatas doit avoir la même taille que ids")

        # Convertir embeddings en numpy array
        emb_array = np.array(embeddings, dtype=np.float32)

        # Vérifier la dimension
        if emb_array.shape[1] != self.dimension:
            raise ValueError(
                f"Dimension des embeddings ({emb_array.shape[1]}) "
                f"ne correspond pas à la dimension de l'index ({self.dimension})"
            )

        # Ajouter à l'index FAISS
        start_idx = self.index.ntotal
        self.index.add(emb_array)

        # Stocker les métadonnées
        for i, (id_, doc, meta) in enumerate(zip(ids, documents, metadatas)):
            # Enrichir les métadonnées avec le document
            full_meta = {
                "id": id_,
                "document": doc,
                "faiss_idx": start_idx + i,  # Index dans FAISS
                **meta
            }
            self.metadata.append(full_meta)
            self.ids.append(id_)

        logger.info(f"[FAISS] Added {len(ids)} documents (total: {self.index.ntotal})")

        # Sauvegarder automatiquement
        self._save()

    def query(
        self,
        query_embeddings: List[List[float]],
        n_results: int = 10,
        include: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Recherche les documents les plus similaires.

        Args:
            query_embeddings: Liste des vecteurs de requête
            n_results: Nombre de résultats à retourner
            include: Liste des champs à inclure

        Returns:
            Dict: {"ids": [[...]], "documents": [[...]], "metadatas": [[...]], "distances": [[...]]}
        """
        if include is None:
            include = ["documents", "metadatas", "distances"]

        # Convertir en numpy
        query_array = np.array(query_embeddings, dtype=np.float32)

        # Vérifier qu'on a des données
        if self.index.ntotal == 0:
            logger.warning("[FAISS] Index is empty, returning no results")
            return {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]]
            }

        # Limiter n_results au nombre de vecteurs disponibles
        n_results = min(n_results, self.index.ntotal)

        # Recherche FAISS
        distances, indices = self.index.search(query_array, n_results)

        # Formater les résultats
        results = {"ids": [], "documents": [], "metadatas": [], "distances": []}

        for i in range(len(query_embeddings)):
            query_ids = []
            query_docs = []
            query_metas = []
            query_dists = []

            for j, idx in enumerate(indices[i]):
                if idx == -1:  # FAISS retourne -1 si pas assez de résultats
                    continue

                # Trouver les métadonnées correspondantes
                # idx est l'index dans FAISS, on doit trouver le bon élément dans metadata
                meta = next((m for m in self.metadata if m.get("faiss_idx") == idx), None)

                if meta:
                    query_ids.append(meta.get("id", str(idx)))

                    if "documents" in include:
                        query_docs.append(meta.get("document", ""))

                    if "metadatas" in include:
                        # Copier les métadonnées sans les champs internes
                        clean_meta = {k: v for k, v in meta.items()
                                     if k not in ["id", "document", "faiss_idx"]}
                        query_metas.append(clean_meta)

                    if "distances" in include:
                        query_dists.append(float(distances[i][j]))

            results["ids"].append(query_ids)
            results["documents"].append(query_docs)
            results["metadatas"].append(query_metas)
            results["distances"].append(query_dists)

        logger.info(f"[FAISS] Query returned {len(results['ids'][0])} results")
        return results

    def count(self) -> int:
        """Retourne le nombre de vecteurs dans l'index"""
        return self.index.ntotal

    def delete(self):
        """Supprime la collection (fichiers sur disque)"""
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        if os.path.exists(self.metadata_path):
            os.remove(self.metadata_path)
        logger.info(f"[FAISS] Deleted collection: {self.name}")

    def _save(self):
        """Sauvegarde l'index et les métadonnées sur disque"""
        # S'assurer que le répertoire existe (critique pour les partages réseau Windows)
        os.makedirs(self.collection_path, exist_ok=True)

        # Sauvegarder l'index FAISS
        faiss.write_index(self.index, self.index_path)

        # Sauvegarder les métadonnées
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump({
                "metadata": self.metadata,
                "ids": self.ids
            }, f, ensure_ascii=False, indent=2)

        logger.info(f"[FAISS] Saved index and metadata to {self.collection_path}")


class FaissStore:
    """Store FAISS pour gérer plusieurs collections"""

    def __init__(self, path: str):
        """
        Args:
            path: Chemin du répertoire de la base de données
        """
        self.path = path
        os.makedirs(path, exist_ok=True)
        logger.info(f"[FAISS] Store initialized at: {path}")

    def get_or_create_collection(
        self,
        name: str,
        dimension: int = 1024
    ) -> FaissCollection:
        """
        Récupère ou crée une collection.

        Args:
            name: Nom de la collection
            dimension: Dimension des embeddings

        Returns:
            FaissCollection
        """
        collection_path = os.path.join(self.path, name)
        return FaissCollection(collection_path, name, dimension)

    def list_collections(self) -> List[str]:
        """Liste les noms de toutes les collections"""
        if not os.path.exists(self.path):
            return []

        collections = []
        for item in os.listdir(self.path):
            item_path = os.path.join(self.path, item)
            if os.path.isdir(item_path):
                # Vérifier qu'il y a bien un index ou des métadonnées
                if (os.path.exists(os.path.join(item_path, "index.faiss")) or
                    os.path.exists(os.path.join(item_path, "metadata.json"))):
                    collections.append(item)

        return collections

    def delete_collection(self, name: str):
        """Supprime une collection"""
        collection_path = os.path.join(self.path, name)
        if os.path.exists(collection_path):
            collection = FaissCollection(collection_path, name)
            collection.delete()
            # Supprimer le dossier s'il est vide
            try:
                os.rmdir(collection_path)
            except OSError:
                pass  # Le dossier n'est pas vide, on le laisse
            logger.info(f"[FAISS] Deleted collection: {name}")

    def get_collection(self, name: str) -> FaissCollection:
        """
        Récupère une collection existante.

        Args:
            name: Nom de la collection

        Returns:
            FaissCollection

        Raises:
            ValueError si la collection n'existe pas
        """
        collection_path = os.path.join(self.path, name)
        if not os.path.exists(collection_path):
            raise ValueError(f"Collection '{name}' does not exist")

        return FaissCollection(collection_path, name)


def build_faiss_store(path: str) -> FaissStore:
    """Crée un store FAISS"""
    return FaissStore(path)
