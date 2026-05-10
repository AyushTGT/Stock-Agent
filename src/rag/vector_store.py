from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import ClassVar, Optional

from src.models.types import NewsArticle

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "financial_news"
_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_CHROMA_DIR = Path("./data/chroma_db")


def _article_to_doc(article: NewsArticle) -> str:
    sectors = ", ".join(article.entities.sectors) or "general"
    stocks = ", ".join(article.entities.stocks) or "none"
    return f"{article.headline}. {article.summary}. Sectors: {sectors}. Stocks: {stocks}."


class VectorStore:
    _instance: ClassVar[Optional["VectorStore"]] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()
    _available: ClassVar[bool] = True

    def __init__(self) -> None:
        self._client = None
        self._collection = None
        self._embedder = None
        self._article_cache: dict[str, NewsArticle] = {}

    @classmethod
    def get_instance(cls) -> "VectorStore":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    inst = VectorStore()
                    inst._init_chromadb()
                    cls._instance = inst
        return cls._instance

    def _init_chromadb(self) -> None:
        import concurrent.futures
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer
            _CHROMA_DIR.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(_CHROMA_DIR))
            self._collection = self._client.get_or_create_collection(
                name=_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            print("[DEBUG] loading SentenceTransformer model...", flush=True)
            ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = ex.submit(SentenceTransformer, _EMBED_MODEL)
            ex.shutdown(wait=False)
            try:
                self._embedder = future.result(timeout=15)
            except concurrent.futures.TimeoutError:
                logger.warning("SentenceTransformer load timed out — RAG disabled")
                VectorStore._available = False
                return
            print("[DEBUG] SentenceTransformer loaded.", flush=True)
            logger.info("ChromaDB initialized with %d documents", self._collection.count())
        except ImportError:
            logger.warning("chromadb or sentence-transformers not installed — RAG disabled")
            VectorStore._available = False
        except Exception as exc:
            logger.warning("ChromaDB init failed: %s — RAG disabled", exc)
            VectorStore._available = False

    @property
    def available(self) -> bool:
        return VectorStore._available and self._collection is not None

    def upsert(self, articles: list[NewsArticle]) -> int:
        if not self.available:
            return 0
        docs = [_article_to_doc(a) for a in articles]
        embeddings = self._embedder.encode(docs).tolist()
        self._collection.upsert(
            ids=[a.id for a in articles],
            embeddings=embeddings,
            documents=docs,
            metadatas=[{
                "id": a.id,
                "headline": a.headline,
                "sentiment": a.sentiment,
                "sentiment_score": a.sentiment_score,
                "scope": a.scope,
                "impact_level": a.impact_level,
                "sectors": ",".join(a.entities.sectors),
                "stocks": ",".join(a.entities.stocks),
                "indices": ",".join(a.entities.indices),
                "conflict_flag": str(a.conflict_flag),
                "published_at": a.published_at,
                "source": a.source,
            } for a in articles],
        )
        for a in articles:
            self._article_cache[a.id] = a
        return len(articles)

    def query(self, text: str, n_results: int = 10, filters: dict | None = None) -> list[NewsArticle]:
        if not self.available or not text.strip():
            return []
        try:
            embedding = self._embedder.encode([text]).tolist()
            kwargs: dict = {"query_embeddings": embedding, "n_results": min(n_results, self.count())}
            if filters:
                kwargs["where"] = filters
            results = self._collection.query(**kwargs)
            articles: list[NewsArticle] = []
            for meta in (results.get("metadatas") or [[]])[0]:
                news_id = meta.get("id", "")
                if news_id in self._article_cache:
                    articles.append(self._article_cache[news_id])
            return articles
        except Exception as exc:
            logger.debug("ChromaDB query failed: %s", exc)
            return []

    def count(self) -> int:
        if not self.available:
            return 0
        try:
            return self._collection.count()
        except Exception:
            return 0

    def reset(self) -> None:
        if self.available:
            try:
                self._client.delete_collection(_COLLECTION_NAME)
                self._collection = self._client.get_or_create_collection(
                    name=_COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"},
                )
                self._article_cache.clear()
            except Exception as exc:
                logger.warning("ChromaDB reset failed: %s", exc)
