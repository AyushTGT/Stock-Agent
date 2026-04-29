from __future__ import annotations

import logging

from src.models.types import NewsArticle
from src.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


def ingest_from_loader(loader) -> int:
    store = VectorStore.get_instance()
    if not store.available:
        return 0
    if store.count() > 0:
        return store.count()
    articles = loader.get_news()
    if not articles:
        return 0
    count = store.upsert(articles)
    logger.info("Ingested %d articles into ChromaDB", count)
    return count


def ingest_articles(articles: list[NewsArticle]) -> int:
    store = VectorStore.get_instance()
    if not store.available or not articles:
        return 0
    count = store.upsert(articles)
    logger.info("Incrementally ingested %d articles", count)
    return count
