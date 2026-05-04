from qdrant_client import QdrantClient

from config import QDRANT_DIR, QDRANT_URL

_client: QdrantClient | None = None


def qdrant_client() -> QdrantClient:
    """Return one process-wide Qdrant client.

    QDRANT_URL uses server mode, which is preferred for Docker and concurrent UI
    actions. Without it, embedded local mode remains available for quick demos.
    """
    global _client
    if _client is None:
        if QDRANT_URL:
            _client = QdrantClient(url=QDRANT_URL)
        else:
            _client = QdrantClient(path=str(QDRANT_DIR))
    return _client


def reset_qdrant_client() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None
