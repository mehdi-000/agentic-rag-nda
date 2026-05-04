import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / "docs"
INDEX_DIR = BASE_DIR / "index"
PARENTS_DIR = INDEX_DIR / "parents"
QDRANT_DIR = INDEX_DIR / "qdrant"
PARSE_STATUS_FILE = INDEX_DIR / "parse_status.json"

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
QDRANT_URL = os.getenv("QDRANT_URL")

LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b-instruct")
FAST_MODEL = os.getenv("FAST_MODEL", "qwen2.5:3b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
EMBED_DIM = 768

CHILD_COLLECTION = "nda_child_chunks"

PARENT_SIZE = 800
PARENT_OVERLAP = 150
CHILD_SIZE = 300
CHILD_OVERLAP = 75

MIN_BLOCK_CHARS = 40
HEADER_FOOTER_MAX_CHARS = 120
HEADING_FONT_RATIO = 1.15

CONTEXTUAL_ENRICHMENT = False

TOP_K = 5
SCORE_THRESHOLD = 0.0
MAX_RETRIES = 2
