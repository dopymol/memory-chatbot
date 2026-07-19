"""
consolidator.py — the memory lifecycle manager.

Problem this solves: if you only ever ADD memories, the store grows
forever, retrieval gets noisier (near-duplicate facts compete for
top-k slots), and cost/latency creep up. A production memory system
needs a compression step, just like a database needs vacuuming.

Strategy here: periodically pull all "fact" memories, ask the LLM to
merge/de-duplicate/summarize them into a smaller set of higher-quality
facts, delete the old ones, and store the consolidated set instead.
"""

import json
import re
from typing import List

from langchain_ollama import ChatOllama

import config
from memory.store import MemoryStore

CONSOLIDATION_PROMPT = """You are a memory consolidation system. Below is a list of facts
that have been recorded about a user over time. Some may be redundant,
outdated, or overly granular.

Merge and de-duplicate them into a clean, non-redundant set of facts.
If a later fact contradicts or updates an earlier one, keep only the
most recent/accurate version. Keep facts atomic (one idea each).

Respond ONLY with a JSON array of the consolidated fact strings. No other text.

Facts:
{facts_list}

Consolidated JSON array:"""


class MemoryConsolidator:
    def __init__(self, store: MemoryStore):
        self.store = store
        self.llm = ChatOllama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0,
        )

    def should_consolidate(self) -> bool:
        fact_count = self.store.count(memory_type="fact")
        return fact_count >= config.MAX_RAW_MEMORIES_BEFORE_CONSOLIDATION

    def consolidate(self) -> dict:
        """Run one consolidation pass. Returns a small report for logging/demo purposes."""
        facts = self.store.get_all_memories(memory_type="fact")
        if len(facts) < 2:
            return {"ran": False, "reason": "not enough facts to consolidate"}

        facts_list_text = "\n".join(f"- {f['text']}" for f in facts)
        prompt = CONSOLIDATION_PROMPT.format(facts_list=facts_list_text)
        response = self.llm.invoke(prompt)
        consolidated_facts = self._parse_json_array(response.content)

        if not consolidated_facts:
            return {"ran": False, "reason": "LLM returned no consolidated facts"}

        # Delete old raw facts, store the compressed set
        old_ids = [f["id"] for f in facts]
        self.store.delete_memories(old_ids)

        for fact in consolidated_facts:
            self.store.add_memory(fact, memory_type="fact", importance=0.7)

        return {
            "ran": True,
            "facts_before": len(facts),
            "facts_after": len(consolidated_facts),
        }

    @staticmethod
    def _parse_json_array(raw_text: str) -> List[str]:
        raw_text = raw_text.strip()
        match = re.search(r"\[.*\]", raw_text, re.DOTALL)
        if not match:
            return []
        try:
            facts = json.loads(match.group(0))
            return [f.strip() for f in facts if isinstance(f, str) and f.strip()]
        except json.JSONDecodeError:
            return []
