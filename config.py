"""
Central configuration for the long-term-memory chatbot.
Keeping every tunable in one place is a small thing that makes the
project look deliberately engineered rather than hacked together.
"""

# ---- LLM (via Ollama, local & free) ----
OLLAMA_MODEL = "llama3.2"          # pull with: ollama pull llama3.2
OLLAMA_BASE_URL = "http://localhost:11434"

# ---- Embeddings (local, free, runs on CPU) ----
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ---- Vector store ----
CHROMA_PERSIST_DIR = "./chroma_db"
MEMORY_COLLECTION_NAME = "long_term_memory"

# ---- Memory behavior ----
SHORT_TERM_WINDOW = 6          # number of raw turns kept in working memory
RETRIEVAL_TOP_K = 4            # how many long-term memories to pull per turn
CONSOLIDATION_EVERY_N_TURNS = 12   # trigger consolidation job after this many turns
MAX_RAW_MEMORIES_BEFORE_CONSOLIDATION = 20  # safety cap
