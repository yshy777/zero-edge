"""
ClusterManager — Shared module for ProtoRMPI and ProtoCATS

Implements the five-phase entity clustering + prototype comparison pipeline:
  Phase 1: Entity clustering (structural + semantic features)
  Phase 2: Prototype extraction (per-cluster centroid)
  Phase 3: Global feature library construction
  Phase 4: New entity feature extraction
  Phase 5: Prototype comparison → soft assignment → link prediction

Can be used with both RMPI (GNN-based) and CATS (LLM-based).
"""

import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


class ClusterManager:
    """
    Manages entity clustering, prototype extraction, and new entity comparison
    for inductive knowledge graph completion.

    Usage:
        # --- Training Phase (Phases 1-3) ---
        cm = ClusterManager(entity2text, relation2id, triple_set)
        cm.cluster_entities(n_clusters=50, method="kmeans")

        # --- Inference Phase (Phases 4-5) ---
        cluster_membership = cm.soft_assign(new_entity_feature, temperature=0.1)
        predicted_relations = cm.predict_relations(new_entity, top_k=5)
    """

    def __init__(
        self,
        entity2text: Dict[str, str],
        relation2id: Dict[str, int],
        triple_set: List[Tuple[str, str, str]],
        entity_embeddings: Optional[Dict[str, np.ndarray]] = None,
        use_semantic: bool = True,
    ):
        """
        Args:
            entity2text: entity_id → text description
            relation2id: relation_str → relation_id
            triple_set: list of (head, relation, tail) training triples
            entity_embeddings: optional pre-trained entity embeddings (e.g., TransE)
            use_semantic: whether to include char-ngram semantic features in the
                clustering feature vector. Set False for datasets whose entity IDs
                carry no text semantics (e.g. WN18RR numeric IDs) -- then clustering
                is driven purely by structural features (degree, neighbor degrees,
                relation-type distribution).
        """
        self.entity2text = entity2text
        self.relation2id = relation2id
        self.id2relation = {v: k for k, v in relation2id.items()}
        self.triple_set = triple_set
        self.entity_embeddings = entity_embeddings or {}
        self.use_semantic = use_semantic

        # Derived data structures
        self.entities = sorted(set(
            [h for h, _, _ in triple_set] + [t for _, _, t in triple_set]
        ))
        self.entity_to_idx = {e: i for i, e in enumerate(self.entities)}
        self.num_entities = len(self.entities)
        self.num_relations = len(relation2id)

        # Entity → relations mapping
        self.entity2relations = defaultdict(set)
        self.entity2neighbors = defaultdict(set)
        for h, r, t in triple_set:
            self.entity2relations[h].add(r)
            self.entity2relations[t].add(r)
            self.entity2neighbors[h].add(t)
            self.entity2neighbors[t].add(h)

        # Clustering results
        self.cluster_labels: Optional[np.ndarray] = None  # [num_entities]
        self.prototypes: Optional[Dict[int, np.ndarray]] = None  # cluster_id → prototype vector
        self.cluster2entities: Optional[Dict[int, List[str]]] = None
        self.cluster2triples: Optional[Dict[int, Dict[str, List]]] = None
        self.n_clusters: int = 0

    # =========================================================================
    # Phase 1: Entity Clustering
    # =========================================================================

    def _structural_features(self, entity_id: str) -> np.ndarray:
        """
        Extract structural feature vector for an entity.
        Features: [degree, avg_neighbor_degree, max_neighbor_degree,
                   std_neighbor_degree, relation_type_distribution...]
        """
        neighbors = self.entity2neighbors[entity_id]
        relations = self.entity2relations[entity_id]
        degree = len(neighbors)
        num_rels = len(relations)

        # Neighbor degree statistics
        neighbor_degrees = [
            len(self.entity2neighbors[n])
            for n in neighbors
        ]
        if neighbor_degrees:
            deg_stats = np.array([
                np.mean(neighbor_degrees),
                np.max(neighbor_degrees),
                np.min(neighbor_degrees),
                np.std(neighbor_degrees)
            ])
        else:
            deg_stats = np.zeros(4)

        # Relation type distribution (sparse one-hot)
        rel_vec = np.zeros(self.num_relations)
        for r in relations:
            if r in self.relation2id:
                rel_vec[self.relation2id[r]] += 1

        # Normalize relation vector
        rel_vec = rel_vec / (rel_vec.sum() + 1e-8)

        # Basic features
        basic_feat = np.array([degree, num_rels])

        return np.concatenate([basic_feat, deg_stats, rel_vec])

    def _semantic_features(self, entity_id: str) -> np.ndarray:
        """Extract semantic feature from entity text (if available)."""
        text = self.entity2text.get(entity_id, entity_id)
        # Simple bag-of-character-ngrams as a lightweight baseline
        # (Can be replaced with SentenceTransformer embeddings)
        ngrams = {}
        for n in [2, 3]:
            for i in range(len(text) - n + 1):
                ngram = text[i:i+n]
                ngrams[ngram] = ngrams.get(ngram, 0) + 1

        # Convert to fixed-size feature (use top 128 features)
        sorted_ngrams = sorted(ngrams.items(), key=lambda x: -x[1])[:128]
        feat = np.zeros(128)
        for i, (_, count) in enumerate(sorted_ngrams):
            feat[i] = count

        return feat / (feat.sum() + 1e-8)

    def _extract_entity_features(self) -> np.ndarray:
        """
        Extract mixed features (structural + semantic) for all entities.
        Returns: np.ndarray of shape (num_entities, feature_dim)
        """
        features = []
        for entity in self.entities:
            struct_f = self._structural_features(entity)
            # Normalize structural features
            struct_f = struct_f / (np.linalg.norm(struct_f) + 1e-8)

            if self.use_semantic:
                sem_f = self._semantic_features(entity)
                sem_f = sem_f / (np.linalg.norm(sem_f) + 1e-8)
                features.append(np.concatenate([struct_f, sem_f]))
            else:
                features.append(struct_f)

        return np.array(features)

    def cluster_entities(
        self,
        n_clusters: int = None,
        method: str = "kmeans",
        random_state: int = 42
    ):
        """
        Phase 1: Cluster all entities based on mixed features.

        Args:
            n_clusters: number of clusters (default: sqrt(num_entities))
            method: "kmeans" or "gmm"
        """
        if n_clusters is None:
            n_clusters = max(5, int(np.sqrt(self.num_entities)))

        self.n_clusters = n_clusters
        features = self._extract_entity_features()

        print(f"[ClusterManager] Clustering {self.num_entities} entities "
              f"into {n_clusters} clusters (dim={features.shape[1]})...")

        if method == "kmeans":
            model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        elif method == "gmm":
            model = GaussianMixture(n_components=n_clusters, random_state=random_state)
        else:
            raise ValueError(f"Unknown clustering method: {method}")

        self.cluster_labels = model.fit_predict(features)
        self._feature_matrix = features

        print(f"[ClusterManager] Clustering done. "
              f"Cluster sizes: min={np.min(np.bincount(self.cluster_labels))}, "
              f"max={np.max(np.bincount(self.cluster_labels))}, "
              f"mean={np.mean(np.bincount(self.cluster_labels)):.1f}")

        # Phase 2 & 3: Extract prototypes and build feature library
        self._compute_prototypes()
        self._build_cluster_index()

        return self

    # =========================================================================
    # Phase 2: Prototype Extraction
    # =========================================================================

    def _compute_prototypes(self):
        """
        Phase 2: Compute the prototype (centroid) for each cluster.
        The prototype is the mean feature vector of all entities in the cluster.
        """
        self.prototypes = {}
        for c in range(self.n_clusters):
            mask = self.cluster_labels == c
            if mask.sum() > 0:
                self.prototypes[c] = self._feature_matrix[mask].mean(axis=0)
            else:
                self.prototypes[c] = np.zeros(self._feature_matrix.shape[1])

    # =========================================================================
    # Phase 3: Global Feature Library
    # =========================================================================

    def _build_cluster_index(self):
        """
        Phase 3: Build index structures for the global feature library.
        - cluster2entities: cluster_id → [entity_ids]
        - cluster2triples: cluster_id → {relation: [(h,r,t), ...]}
        """
        # cluster2entities
        self.cluster2entities = defaultdict(list)
        for i, entity in enumerate(self.entities):
            c = self.cluster_labels[i]
            self.cluster2entities[c].append(entity)

        # cluster2triples: index triples by head-entity's cluster and relation
        self.cluster2triples = defaultdict(lambda: defaultdict(list))
        for h, r, t in self.triple_set:
            if h in self.entity_to_idx and r in self.relation2id:
                c_h = self.cluster_labels[self.entity_to_idx[h]]
                self.cluster2triples[c_h][r].append((h, r, t))

    def get_cluster_relation_distribution(self, cluster_id: int) -> Dict[str, float]:
        """
        Get the relation frequency distribution for a given cluster.

        Returns: {relation_str: normalized_frequency}
        """
        rel_counts = defaultdict(int)
        total = 0
        for r, triples in self.cluster2triples[cluster_id].items():
            rel_counts[r] = len(triples)
            total += len(triples)

        return {r: c / total for r, c in rel_counts.items()} if total > 0 else {}

    # =========================================================================
    # Phase 4: New Entity Feature Extraction
    # =========================================================================

    def extract_new_entity_features(
        self,
        new_entity_id: str,
        neighbor_entities: List[str],
        neighbor_relations: List[str]
    ) -> np.ndarray:
        """
        Phase 4: Extract features for a new (unseen) entity.

        Args:
            new_entity_id: the entity identifier
            neighbor_entities: list of neighboring entity IDs
            neighbor_relations: list of relation types connecting to neighbors

        Returns:
            Feature vector for the new entity (same dimension as training features)
        """
        # Structural features from local neighborhood
        degree = len(neighbor_entities)
        num_rels = len(set(neighbor_relations))

        neighbor_degrees = [len(self.entity2neighbors.get(n, [])) for n in neighbor_entities]
        if neighbor_degrees:
            deg_stats = np.array([
                np.mean(neighbor_degrees),
                np.max(neighbor_degrees),
                np.min(neighbor_degrees),
                np.std(neighbor_degrees)
            ])
        else:
            deg_stats = np.zeros(4)

        rel_vec = np.zeros(self.num_relations)
        for r in neighbor_relations:
            if r in self.relation2id:
                rel_vec[self.relation2id[r]] += 1
        rel_vec = rel_vec / (rel_vec.sum() + 1e-8)

        basic_feat = np.array([degree, num_rels])
        struct_f = np.concatenate([basic_feat, deg_stats, rel_vec])
        struct_f = struct_f / (np.linalg.norm(struct_f) + 1e-8)

        if not self.use_semantic:
            return struct_f

        # Semantic features from entity text
        text = self.entity2text.get(new_entity_id, new_entity_id)
        ngrams = {}
        for n in [2, 3]:
            for i in range(len(text) - n + 1):
                ngram = text[i:i+n]
                ngrams[ngram] = ngrams.get(ngram, 0) + 1

        sorted_ngrams = sorted(ngrams.items(), key=lambda x: -x[1])[:128]
        sem_f = np.zeros(128)
        for i, (_, count) in enumerate(sorted_ngrams):
            sem_f[i] = count
        sem_f = sem_f / (sem_f.sum() + 1e-8)

        return np.concatenate([struct_f, sem_f])

    # =========================================================================
    # Phase 5: Prototype Comparison → Soft Assignment → Link Prediction
    # =========================================================================

    def soft_assign(
        self,
        new_entity_feature: np.ndarray,
        temperature: float = 0.1
    ) -> np.ndarray:
        """
        Phase 5(a): Soft-assign a new entity to all clusters based on prototype similarity.

        Args:
            new_entity_feature: feature vector from Phase 4
            temperature: softmax temperature (lower = harder assignment)

        Returns:
            alpha: soft cluster membership vector of shape (n_clusters,) summing to 1
        """
        sims = np.zeros(self.n_clusters)
        for c in range(self.n_clusters):
            proto = self.prototypes[c]
            sim = np.dot(new_entity_feature, proto) / (
                np.linalg.norm(new_entity_feature) * np.linalg.norm(proto) + 1e-8
            )
            sims[c] = sim

        # Temperature-scaled softmax
        alpha = np.exp(sims / temperature)
        alpha = alpha / alpha.sum()
        return alpha

    def predict_relations(
        self,
        new_entity_feature: np.ndarray,
        top_k: int = 5,
        temperature: float = 0.1
    ) -> List[Tuple[str, float]]:
        """
        Phase 5(b): Predict likely relations for a new entity based on cluster membership.

        Args:
            new_entity_feature: feature vector from Phase 4
            top_k: number of top predicted relations to return
            temperature: softmax temperature

        Returns:
            List of (relation_str, confidence_score) sorted by confidence
        """
        alpha = self.soft_assign(new_entity_feature, temperature)

        # Aggregate relation distributions from all clusters, weighted by alpha
        rel_scores = defaultdict(float)
        for c in range(self.n_clusters):
            rel_dist = self.get_cluster_relation_distribution(c)
            for r, prob in rel_dist.items():
                rel_scores[r] += alpha[c] * prob

        # Sort and return top-K
        sorted_rels = sorted(rel_scores.items(), key=lambda x: -x[1])
        return sorted_rels[:top_k]

    def predict_triple_plausibility(
        self,
        head_entity: str,
        relation: str,
        tail_entity: str,
        head_is_new: bool = False,
        tail_is_new: bool = False,
        head_feature: Optional[np.ndarray] = None,
        tail_feature: Optional[np.ndarray] = None,
        temperature: float = 0.1
    ) -> float:
        """
        Phase 5(c): Score the plausibility of a triple involving new entities.

        The score is based on:
        - If head is new: how well its cluster membership supports the relation
        - If tail is new: how well its cluster membership supports the relation
        - The structural consistency between the two clusters

        Returns:
            Plausibility score in [0, 1]
        """
        if head_is_new and head_feature is not None:
            head_alpha = self.soft_assign(head_feature, temperature)
        elif head_entity in self.entity_to_idx:
            hard_alpha = np.zeros(self.n_clusters)
            c = self.cluster_labels[self.entity_to_idx[head_entity]]
            hard_alpha[c] = 1.0
            head_alpha = hard_alpha
        else:
            return 0.0

        if tail_is_new and tail_feature is not None:
            tail_alpha = self.soft_assign(tail_feature, temperature)
        elif tail_entity in self.entity_to_idx:
            hard_alpha = np.zeros(self.n_clusters)
            c = self.cluster_labels[self.entity_to_idx[tail_entity]]
            hard_alpha[c] = 1.0
            tail_alpha = hard_alpha
        else:
            return 0.0

        # Score = P(relation | head's cluster) × P(relation | tail's cluster)
        # weighted by the soft assignment
        score = 0.0
        for c_h in range(self.n_clusters):
            for c_t in range(self.n_clusters):
                rel_dist_h = self.get_cluster_relation_distribution(c_h)
                rel_dist_t = self.get_cluster_relation_distribution(c_t)

                prob_h = rel_dist_h.get(relation, 0.0)
                prob_t = rel_dist_t.get(relation, 0.0)

                score += head_alpha[c_h] * tail_alpha[c_t] * (prob_h + prob_t) / 2

        return score

    def get_entity_cluster(self, entity_id: str) -> int:
        """Get the hard cluster assignment for a known entity."""
        if entity_id in self.entity_to_idx:
            return self.cluster_labels[self.entity_to_idx[entity_id]]
        return -1

    def get_cluster_fewshot_triples(
        self,
        entity_id: str,
        relation: str,
        max_triples: int = 5
    ) -> List[Tuple[str, str, str]]:
        """
        CATS-specific: Generate few-shot examples from the entity's cluster.
        Replaces CATS's diverse_fewshot_triple_finder().

        Selects triples from the same cluster that share the target relation,
        prioritizing diverse head/tail entities.
        """
        if entity_id not in self.entity_to_idx:
            return []

        cluster_id = self.cluster_labels[self.entity_to_idx[entity_id]]
        candidates = self.cluster2triples[cluster_id].get(relation, [])

        if len(candidates) <= max_triples:
            return candidates

        # Diversity selection: prefer entities not yet used
        selected = []
        used_heads = {entity_id}
        used_tails = set()

        for h, r, t in candidates:
            if h not in used_heads and t not in used_tails:
                selected.append((h, r, t))
                used_heads.add(h)
                used_tails.add(t)
                if len(selected) == max_triples:
                    return selected

        # Fallback: fill remaining slots
        for h, r, t in candidates:
            if (h, r, t) not in selected and len(selected) < max_triples:
                selected.append((h, r, t))

        return selected[:max_triples]
