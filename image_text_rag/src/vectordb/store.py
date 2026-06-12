"""
向量存储核心 — FAISS IndexIDMap + JSON 元数据持久化

v2 改进:
  - IndexIDMap: O(1) remove_ids() 删除
  - 取消双重 L2 归一化：CLIP 输出已归一化，无需重复
  - 简化 ID 映射：FAISS 原生 add_with_ids 管理

用法:
    store = VectorStore(dim=512)
    ids = store.add(vectors, metadatas)
    results = store.search(query_vec, top_k=5)
    store.delete(ids)       # O(1)
    store.save()
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..common import load_config, get_logger
Config = load_config()
_log = get_logger(__name__)

_FAISS_AVAILABLE = False
try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    pass


@dataclass
class SearchResult:
    """单条检索结果"""
    id: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class VectorStore:
    """
    向量存储 & 检索引擎 (v2 — IndexIDMap)

    用法:
        store = VectorStore(dim=512)
        ids = store.add(vectors, metadatas)
        results = store.search(query_vec, top_k=5)
        store.delete(ids)
        store.save()
    """

    def __init__(
        self,
        dim: int | None = None,
        index_type: str | None = None,
        persist_dir: str | None = None,
    ):
        self._dim = dim or 512
        self._index_type = index_type or Config.INDEX_TYPE
        self._persist_dir = Path(persist_dir or Config.PERSIST_DIR)

        self._index: Any = None
        self._metadata: Dict[int, Dict[str, Any]] = {}  # int_id → meta
        self._id_to_int: Dict[str, int] = {}             # ext_id → int_id
        self._int_to_id: Dict[int, str] = {}             # int_id → ext_id
        self._next_int_id: int = 0

        self._build_index()

    # ── 属性 ──

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def count(self) -> int:
        return self._index.ntotal if self._index is not None else 0

    @property
    def index_type(self) -> str:
        return self._index_type

    # ── 索引构建 ──

    def _build_index(self):
        if not _FAISS_AVAILABLE:
            self._index = None
            self._numpy_vectors: Optional[np.ndarray] = None
            return

        if self._index_type == "flat":
            # IndexIDMap 包装 → 支持 O(1) remove_ids
            base = faiss.IndexFlatIP(self._dim)
            self._index = faiss.IndexIDMap(base)
        elif self._index_type == "hnsw":
            base = faiss.IndexHNSWFlat(self._dim, 32)
            self._index = faiss.IndexIDMap(base)
        elif self._index_type == "ivf":
            nlist = max(4, int(np.sqrt(1000)))
            quantizer = faiss.IndexFlatIP(self._dim)
            base = faiss.IndexIVFFlat(quantizer, self._dim, nlist,
                                       faiss.METRIC_INNER_PRODUCT)
            self._index = faiss.IndexIDMap(base)
            self._index_trained = False
        else:
            raise ValueError(f"不支持的索引类型: {self._index_type}")

        _log.info("IndexIDMap(%s) dim=%d", self._index_type, self._dim)

    # ── 添加 ──

    def add(
        self,
        vectors: np.ndarray,
        metadatas: List[Dict[str, Any]] | None = None,
        ids: List[str] | None = None,
    ) -> List[str]:
        """
        批量添加向量。

        Args:
            vectors:   (N, dim) float32 — CLIP 已归一化，无需重复 L2
            metadatas: 元数据列表
            ids:       外部 ID，不传则自动生成 uuid
        Returns:
            分配的 ID 列表
        """
        N = vectors.shape[0]
        if vectors.shape[1] != self._dim:
            raise ValueError(f"维度不匹配: 期望 {self._dim}, 实际 {vectors.shape[1]}")
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)

        if metadatas is None:
            metadatas = [{}] * N
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(N)]

        if _FAISS_AVAILABLE and self._index is not None:
            self._add_faiss(vectors, metadatas, ids)
        else:
            self._add_numpy(vectors, metadatas, ids)

        return ids

    def _add_faiss(self, vectors, metadatas, ids):
        # IVF 需训练
        if self._index_type == "ivf" and not getattr(self, "_index_trained", True):
            try:
                base = faiss.downcast_index(self._index.index)
                base.train(vectors)
                self._index_trained = True
            except Exception as e:
                _log.warning("IVF 训练失败: %s", e)

        # 分配内部 int ID 并写入
        int_ids = np.array(
            [self._next_int_id + i for i in range(len(ids))],
            dtype=np.int64,
        )
        self._index.add_with_ids(vectors, int_ids)

        for i, (ext_id, meta) in enumerate(zip(ids, metadatas)):
            int_id = self._next_int_id + i
            self._metadata[int_id] = meta
            self._id_to_int[ext_id] = int_id
            self._int_to_id[int_id] = ext_id

        self._next_int_id += len(ids)

    def _add_numpy(self, vectors, metadatas, ids):
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        if self._numpy_vectors is None:
            self._numpy_vectors = vectors
        else:
            self._numpy_vectors = np.concatenate(
                [self._numpy_vectors, vectors], axis=0
            )
        for i, (ext_id, meta) in enumerate(zip(ids, metadatas)):
            int_id = self._next_int_id + i
            self._metadata[int_id] = meta
            self._id_to_int[ext_id] = int_id
            self._int_to_id[int_id] = ext_id
        self._next_int_id += len(ids)

    # ── 检索 ──

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int | None = None,
        *,
        min_score: float = 0.0,
    ) -> List[SearchResult]:
        if top_k is None:
            top_k = Config.TOP_K
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        if query_vector.dtype != np.float32:
            query_vector = query_vector.astype(np.float32)

        if _FAISS_AVAILABLE and self._index is not None:
            return self._search_faiss(query_vector, top_k, min_score)
        else:
            return self._search_numpy(query_vector, top_k, min_score)

    def _search_faiss(self, q, top_k, min_score):
        k = min(top_k, self._index.ntotal)
        if k == 0:
            return []
        scores, faiss_ids = self._index.search(q, k)
        results = []
        for score, fid in zip(scores[0], faiss_ids[0]):
            if fid == -1:
                continue
            if float(score) < min_score:
                continue
            fid = int(fid)
            results.append(SearchResult(
                id=self._int_to_id.get(fid, str(fid)),
                score=float(score),
                metadata=self._metadata.get(fid, {}),
            ))
        return results

    def _search_numpy(self, q, top_k, min_score):
        if self._numpy_vectors is None or len(self._numpy_vectors) == 0:
            return []
        scores = np.dot(self._numpy_vectors, q.T).flatten()
        k = min(top_k, len(scores))
        if k == 0:
            return []
        top_indices = np.argsort(scores)[::-1][:k]
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score < min_score:
                continue
            results.append(SearchResult(
                id=self._int_to_id.get(idx, str(idx)),
                score=score,
                metadata=self._metadata.get(idx, {}),
            ))
        return results

    def batch_search(
        self, query_vectors: np.ndarray, top_k: int | None = None, **kwargs
    ) -> List[List[SearchResult]]:
        return [self.search(qv, top_k, **kwargs) for qv in query_vectors]

    # ── 删除 (O(1) IndexIDMap) ──

    def delete(self, ids: List[str]) -> int:
        """O(1) 删除 — 使用 IndexIDMap.remove_ids()"""
        to_remove = []
        for eid in ids:
            int_id = self._id_to_int.get(eid)
            if int_id is not None:
                to_remove.append(int_id)

        if not to_remove:
            return 0

        if _FAISS_AVAILABLE and self._index is not None:
            id_array = np.array(to_remove, dtype=np.int64)
            removed = self._index.remove_ids(id_array)
            for int_id in to_remove:
                eid = self._int_to_id.pop(int_id, None)
                if eid:
                    self._id_to_int.pop(eid, None)
                self._metadata.pop(int_id, None)
            return int(removed)
        else:
            for int_id in to_remove:
                eid = self._int_to_id.pop(int_id, None)
                if eid:
                    self._id_to_int.pop(eid, None)
                self._metadata.pop(int_id, None)
            return len(to_remove)

    def clear(self):
        self._build_index()
        self._metadata.clear()
        self._id_to_int.clear()
        self._int_to_id.clear()
        self._next_int_id = 0
        self._numpy_vectors = None

    # ── 持久化 ──

    def save(self, path: str | None = None):
        save_dir = Path(path or self._persist_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        index_path = save_dir / "index.faiss"
        meta_path = save_dir / "metadata.json"
        state_path = save_dir / "state.json"

        if _FAISS_AVAILABLE and self._index is not None:
            faiss.write_index(self._index, str(index_path))
        elif self._numpy_vectors is not None:
            np.save(save_dir / "vectors.npy", self._numpy_vectors)

        state = {
            "dim": self._dim,
            "index_type": self._index_type,
            "next_int_id": self._next_int_id,
            "count": self.count,
        }
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

        meta_payload = {
            "metadata": {str(k): v for k, v in self._metadata.items()},
            "id_to_int": self._id_to_int,
            "int_to_id": {str(k): v for k, v in self._int_to_id.items()},
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_payload, f, ensure_ascii=False, indent=2)

        _log.info("已保存 → %s (%d条)", save_dir, self.count)

    def load(self, path: str | None = None) -> bool:
        load_dir = Path(path or self._persist_dir)
        index_path = load_dir / "index.faiss"
        meta_path = load_dir / "metadata.json"
        state_path = load_dir / "state.json"
        numpy_path = load_dir / "vectors.npy"

        if not state_path.exists():
            _log.warning("存档不存在: %s", state_path)
            return False

        with open(state_path) as f:
            state = json.load(f)
        self._dim = state["dim"]
        self._index_type = state["index_type"]
        self._next_int_id = state["next_int_id"]

        if index_path.exists() and _FAISS_AVAILABLE:
            self._index = faiss.read_index(str(index_path))
        elif numpy_path.exists():
            self._numpy_vectors = np.load(str(numpy_path))
            self._index = None
        else:
            self._build_index()

        with open(meta_path) as f:
            payload = json.load(f)
        self._metadata = {int(k): v for k, v in payload["metadata"].items()}
        self._id_to_int = payload["id_to_int"]
        self._int_to_id = {int(k): v for k, v in payload["int_to_id"].items()}

        _log.info("已加载 ← %s (%d条)", load_dir, self.count)
        return True
