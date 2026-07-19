from collections import deque
from datetime import datetime
from typing import List, Dict

from langchain_ollama import ChatOllama

import config
from memory.store import MemoryStore
from memory.extractor import FactExtractor
from memory.consolidator import MemoryConsolidator

SYSTEM_PROMPT = """You are a helpful assistant with long-term memory of this user.
Use the "Relevant memories" below if they help answer naturally — don't
mention that you have a memory system, just use the information as if
you simply remember it. If a memory isn't relevant, ignore it."""


class ChatEngine:
    def __init__(self):
        self.llm = ChatOllama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0.7,
        )
        self.store = MemoryStore()
        self.extractor = FactExtractor()
        self.consolidator = MemoryConsolidator(self.store)

        # short-term memory: raw recent turns, capped window
        self.short_term: deque = deque(maxlen=config.SHORT_TERM_WINDOW)
        self.turn_count = 0

    def _build_prompt(self, user_message: str, relevant_memories: List[Dict]) -> str:
        current_time_str = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
        time_block = f"\n\nCurrent date and time: {current_time_str}"
        memory_block = ""
        if relevant_memories:
            memory_lines = "\n".join(f"- {m['text']}" for m in relevant_memories)
            memory_block = f"\n\nRelevant memories about this user:\n{memory_lines}"

        history_block = ""
        if self.short_term:
            history_lines = "\n".join(
                f"User: {t['user']}\nAssistant: {t['assistant']}" for t in self.short_term
            )
            history_block = f"\n\nRecent conversation:\n{history_lines}"

        return (
            f"{SYSTEM_PROMPT}{time_block}{memory_block}{history_block}\n\n"
            f"User: {user_message}\nAssistant:"
        )

    def chat(self, user_message: str) -> Dict:
        """Process one turn. Returns dict with the reply plus debug info
        useful for a demo (what memories were retrieved, whether consolidation ran)."""

        relevant_memories = self.store.retrieve_relevant(user_message)
        prompt = self._build_prompt(user_message, relevant_memories)

        response = self.llm.invoke(prompt)
        assistant_message = response.content

        # Update short-term window
        self.short_term.append({"user": user_message, "assistant": assistant_message})

        # Extract and store new long-term facts from this exchange
        new_facts = self.extractor.extract(user_message, assistant_message)
        for fact in new_facts:
            self.store.add_memory(fact, memory_type="fact")

        # Also keep a raw log (useful for debugging/demo, not used for recall directly)
        self.store.add_memory(
            f"User said: {user_message}", memory_type="raw_turn", importance=0.2
        )

        self.turn_count += 1

        consolidation_report = {"ran": False}
        if self.consolidator.should_consolidate():
            consolidation_report = self.consolidator.consolidate()

        return {
            "reply": assistant_message,
            "retrieved_memories": relevant_memories,
            "new_facts_extracted": new_facts,
            "consolidation": consolidation_report,
        }
