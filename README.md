# Long-Term Memory Chatbot



A chatbot that remembers facts about you across sessions — not by dumping
raw chat history into an ever-growing context window, but with a proper
two-tier memory architecture: a short-term working buffer and a
consolidated long-term semantic store.

100% free and local: no paid APIs, no API keys. Runs entirely on your machine.

## Why this project

Most "chatbot with memory" tutorials just embed every message and retrieve
top-k on each turn. That works for a demo but breaks down in the real
world: the memory store grows unbounded, near-duplicate memories crowd out
genuinely relevant ones, and nothing ever gets updated when a fact changes.

This project addresses those problems directly:

| Problem | Solution implemented here |
|---|---|
| Context window can't hold full history | Two-tier memory: short-term sliding window + long-term vector store |
| Raw chat is noisy to retrieve over | LLM-based fact extraction converts exchanges into clean, atomic facts before storing |
| Memory store grows forever | Periodic **consolidation** job merges/de-duplicates facts, using an LLM to resolve contradictions |
| Retrieval needs to be relevant, not just recent | Semantic similarity search (embeddings) over the fact store, not keyword or recency matching |

## Architecture

```
User message
     │
     ▼
Retrieve top-k relevant memories (vector similarity search)
     │
     ▼
Build prompt: system + retrieved memories + recent turns + new message
     │
     ▼
LLM generates response (Ollama, local)
     │
     ├──► Extract new durable facts from this exchange (LLM call)
     │        └──► Store as embeddings in ChromaDB
     │
     └──► Every N turns: run consolidation
              └──► LLM merges/de-dupes/updates the fact store
```

## Stack

- **LLM**: [Ollama](https://ollama.com) running Llama 3.2 locally — free, no API key, works offline
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` — local, free, CPU-friendly
- **Vector store**: ChromaDB — local, persists to disk, zero external service
- **Orchestration**: LangChain
- **UI**: Streamlit (optional — CLI works standalone)

## Setup

### 1. Install Ollama and pull a model
Download Ollama from https://ollama.com, then:
```bash
ollama pull llama3.2
ollama serve   # if it isn't already running as a service
```

### 2. Set up the Python project (in PyCharm or terminal)
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run it

CLI version:
```bash
python chat_cli.py
```

Web UI version (recommended for demos/screenshots):
```bash
streamlit run app_streamlit.py
```

## Project structure

```
memory_chatbot/
├── config.py                 # all tunable parameters in one place
├── chat_engine.py             # orchestrates retrieval → prompt → LLM → extraction
├── chat_cli.py                 # terminal interface
├── app_streamlit.py            # web UI with memory-internals panel
├── memory/
│   ├── store.py                # ChromaDB wrapper: add/retrieve/delete memories
│   ├── extractor.py             # LLM-based fact extraction from conversation
│   └── consolidator.py          # periodic memory merge/de-dup/compression job
└── requirements.txt
```

## Design decisions worth mentioning in an interview

- **Why extract facts instead of embedding raw messages?** Raw chat turns
  are noisy — full of filler, references to "it" and "that," and
  conversational scaffolding. An atomic fact like "User prefers dark mode
  UIs" embeds and retrieves far more reliably than "yeah I really don't
  like it when apps are all white and bright, kind of hurts my eyes."

- **Why consolidate instead of just capping storage size?** A hard cap
  (e.g. "keep only last 100 memories") would silently drop old but still
  valid facts. Consolidation instead merges and resolves contradictions —
  closer to how you'd want a real assistant's memory to behave over months
  of use.

- **Trade-offs**: local LLMs (Llama 3.2 via Ollama) are slower and less
  reliable at structured output than hosted frontier models — the
  extractor and consolidator include defensive JSON parsing for exactly
  this reason. In a production system you'd likely use a hosted model for
  these steps and reserve local models only where latency/cost matters most.

## Possible extensions
- Add importance-weighted forgetting (decay low-importance memories over time)
- Swap in a knowledge graph for relational facts ("X works at Y")
- Add per-user memory namespaces for multi-tenant use
- Swap Ollama for a hosted API (Claude, GPT) behind a config flag for comparison
