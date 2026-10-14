"""
Outcome-Aware Case Retriever for Memento-Agri.

Supports semantic and multi-attribute case retrieval with outcome-aware reranking.
Features dual backend support:
1. FAISS + SentenceTransformers when available.
2. High-performance Scikit-Learn TF-IDF + Feature Cosine fallback when FAISS/PyTorch
   are not installed or on platforms without prebuilt wheels (e.g., Windows Python 3.13).
"""
from pathlib import Path
import json
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union

from config import TOP_K, RANDOM_STATE, EMBEDDING_MODEL

# Check optional FAISS & Sentence-Transformers availability
try:
    import faiss  # type: ignore
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False


def _safe_float(val, default=0.0):
    try:
        if val is None or pd.isna(val):
            return default
        return float(val)
    except (ValueError, TypeError):
        return default


def _extract_query_info(query_case: Union[Dict, str]) -> Dict[str, Any]:
    """Normalize query whether it is a dict, nested agent context, or string."""
    if isinstance(query_case, str):
        return {
            "query_text": query_case,
            "crop": "",
            "state": "",
            "district": "",
            "rainfall_anomaly": 0.0,
            "temperature_anomaly": 0.0,
            "climate_risk": 0.5,
        }

    context = query_case.get("context", {}) if isinstance(query_case.get("context"), dict) else query_case
    risk = query_case.get("risk", {}) if isinstance(query_case.get("risk"), dict) else {}

    crop = str(context.get("crop", query_case.get("crop", ""))).strip().lower()
    state = str(context.get("state", query_case.get("state", ""))).strip().lower()
    district = str(context.get("district", query_case.get("district", ""))).strip().lower()

    rain_anom = _safe_float(context.get("rainfall_anomaly", query_case.get("rainfall_anomaly", 0.0)))
    temp_anom = _safe_float(context.get("temperature_anomaly", query_case.get("temperature_anomaly", 0.0)))
    risk_score = _safe_float(risk.get("score", query_case.get("climate_risk", 0.5)))

    text = query_case.get("text", "")
    if not text:
        text = f"{crop} in {district} {state} with rainfall anomaly {rain_anom:+.1%} and temperature anomaly {temp_anom:+.1%}"

    return {
        "query_text": text,
        "crop": crop,
        "state": state,
        "district": district,
        "rainfall_anomaly": rain_anom,
        "temperature_anomaly": temp_anom,
        "climate_risk": risk_score,
    }


def build_index(cases: List[Dict[str, Any]], index_path: Union[str, Path], meta_path: Union[str, Path]) -> int:
    """
    Build vector index from cases and save index + metadata.
    Uses FAISS if available, otherwise builds a scikit-learn cosine matrix.
    """
    index_path = Path(index_path)
    meta_path = Path(meta_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    texts = [c.get("text", "") for c in cases]

    built = False
    if _FAISS_AVAILABLE and _ST_AVAILABLE:
        try:
            model = SentenceTransformer(EMBEDDING_MODEL)
            embeddings = model.encode(texts, batch_size=128, show_progress_bar=False, normalize_embeddings=True)
            dim = embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(np.array(embeddings, dtype=np.float32))
            faiss.write_index(index, str(index_path))
            built = True
        except Exception:
            built = False

    if not built:
        # Scikit-learn high-performance TF-IDF fallback
        from sklearn.feature_extraction.text import TfidfVectorizer
        import joblib

        vectorizer = TfidfVectorizer(max_features=256, stop_words="english")
        tfidf_mat = vectorizer.fit_transform(texts)
        payload = {
            "backend": "sklearn",
            "vectorizer": vectorizer,
            "matrix": tfidf_mat,
            "n_cases": len(cases),
        }
        joblib.dump(payload, str(index_path))

    # Save metadata JSON
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2)

    return len(cases)


class CaseRetriever:
    """
    Retrieves historical cases based on contextual climate similarity
    and performs outcome-aware reranking.
    """

    def __init__(self, index_path: Union[str, Path] = None, metadata_path: Union[str, Path] = None):
        self.index_path = Path(index_path) if index_path else None
        self.metadata_path = Path(metadata_path) if metadata_path else None
        self.cases: List[Dict[str, Any]] = []
        self.backend = "sklearn"
        self.faiss_index = None
        self.st_model = None
        self.sklearn_payload = None

        if self.metadata_path and self.metadata_path.exists():
            self._load_metadata()

        if self.index_path and self.index_path.exists():
            self._load_index()

    @classmethod
    def load(cls, index_path: Union[str, Path], metadata_path: Union[str, Path]) -> "CaseRetriever":
        return cls(index_path=index_path, metadata_path=metadata_path)

    def _load_metadata(self):
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.cases = json.load(f)

    def _load_index(self):
        if _FAISS_AVAILABLE and _ST_AVAILABLE:
            try:
                self.faiss_index = faiss.read_index(str(self.index_path))
                self.st_model = SentenceTransformer(EMBEDDING_MODEL)
                self.backend = "faiss"
                return
            except Exception:
                pass

        # Fallback to joblib / sklearn index
        try:
            import joblib
            self.sklearn_payload = joblib.load(str(self.index_path))
            self.backend = "sklearn"
        except Exception:
            self.backend = "direct"

    def retrieve(
        self,
        query_case: Union[Dict[str, Any], str],
        top_k: int = TOP_K,
        outcome_aware: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k similar historical cases with outcome-aware reranking.

        Parameters
        ----------
        query_case : Dict or str
            The query context or string query.
        top_k : int
            Number of cases to retrieve.
        outcome_aware : bool
            Whether to rerank cases favoring positive/resilient yield outcomes.

        Returns
        -------
        List[Dict[str, Any]]
            Top retrieved cases with similarity and final scores.
        """
        if not self.cases:
            return []

        q_info = _extract_query_info(query_case)
        n = len(self.cases)
        similarities = np.zeros(n, dtype=float)

        # 1. Base semantic or text-vector similarity
        if self.backend == "faiss" and self.faiss_index is not None and self.st_model is not None:
            try:
                emb = self.st_model.encode([q_info["query_text"]], normalize_embeddings=True)
                scores, indices = self.faiss_index.search(np.array(emb, dtype=np.float32), min(n, 200))
                for score, idx in zip(scores[0], indices[0]):
                    if 0 <= idx < n:
                        similarities[idx] = max(0.0, float(score))
            except Exception:
                pass

        elif self.backend == "sklearn" and self.sklearn_payload is not None:
            try:
                from sklearn.metrics.pairwise import cosine_similarity
                vec = self.sklearn_payload["vectorizer"].transform([q_info["query_text"]])
                sims = cosine_similarity(vec, self.sklearn_payload["matrix"]).ravel()
                similarities = np.maximum(0.0, sims)
            except Exception:
                pass

        # 2. Add domain-informed attribute matching bonus
        # Matching crop is paramount in agricultural advisory
        for i, c in enumerate(self.cases):
            crop_match = (
                q_info["crop"]
                and q_info["crop"] in str(c.get("crop", "")).lower()
            )
            district_match = (
                q_info["district"]
                and q_info["district"] in str(c.get("district", "")).lower()
            )
            state_match = (
                q_info["state"]
                and q_info["state"] in str(c.get("state", "")).lower()
            )

            # Climate distance penalty
            c_rain_anom = _safe_float(c.get("rainfall_anomaly"))
            c_temp_anom = _safe_float(c.get("temperature_anomaly"))
            climate_dist = abs(q_info["rainfall_anomaly"] - c_rain_anom) + abs(q_info["temperature_anomaly"] - c_temp_anom)
            climate_similarity = max(0.0, 1.0 - (climate_dist / 2.0))

            base_sim = similarities[i] * 0.4 + climate_similarity * 0.3
            if crop_match:
                base_sim += 0.20
            if state_match:
                base_sim += 0.05
            if district_match:
                base_sim += 0.05

            similarities[i] = min(1.0, base_sim)

        # 3. Outcome-aware reranking
        scored_cases = []
        for i, c in enumerate(self.cases):
            sim = float(similarities[i])
            outcome = c.get("outcome", {})
            yield_anom = _safe_float(outcome.get("yield_anomaly", c.get("yield_anomaly", 0.0)))

            # Outcome reward: positive yield despite stress is a hallmark of climate resilience
            outcome_score = max(-1.0, min(1.0, yield_anom))
            
            if outcome_aware:
                # Outcome-aware combination
                final_score = sim + 0.15 * max(0.0, outcome_score)
            else:
                final_score = sim

            case_copy = dict(c)
            case_copy["similarity"] = round(sim, 4)
            case_copy["outcome_score"] = round(outcome_score, 4)
            case_copy["final_score"] = round(final_score, 4)
            scored_cases.append(case_copy)

        # Sort descending by final score
        scored_cases.sort(key=lambda x: x["final_score"], reverse=True)
        return scored_cases[:top_k]
