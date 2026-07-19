"""
extractor.py — turns raw conversation into durable, storable facts.

This is the piece that separates a "real" long-term memory system from
a naive one that just embeds every raw message. Instead of storing
"user said: I just adopted a golden retriever puppy named Biscuit",
we extract a clean fact: "User has a dog named Biscuit (golden retriever)".

Clean, atomic facts embed and retrieve far better than raw chat turns.
"""

import json
import re
from typing import List

from langchain_ollama import ChatOllama

import config

FACT_EXTRACTION_PROMPT = """You are a memory extraction system. Read the exchange below and extract
any NEW, durable facts about the user worth remembering long-term
(preferences, identity details, ongoing projects, relationships, goals, constraints).

Rules:
- Only extract facts that are actually stated or clearly implied, never invent details.
- Skip small talk, greetings, and anything purely about the current moment.
- Each fact should be a short, self-contained sentence.
- If there is nothing worth remembering, return an empty list.

Respond ONLY with a JSON array of strings. No other text, no markdown fences.

Exchange:
User: {user_message}
Assistant: {assistant_message}

JSON array:"""


class FactExtractor:
    def __init__(self):
        self.llm = ChatOllama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0,  # deterministic extraction, not creative generation
        )

    def extract(self, user_message: str, assistant_message: str) -> List[str]:
        prompt = FACT_EXTRACTION_PROMPT.format(
            user_message=user_message, assistant_message=assistant_message
        )
        response = self.llm.invoke(prompt)
        return self._parse_json_array(response.content)

    @staticmethod
    def _parse_json_array(raw_text: str) -> List[str]:
        """Local models often don't follow 'JSON only' instructions reliably.
        Try strict JSON first; if that fails, fall back to salvaging facts
        from bullet points / numbered lines / quoted strings instead of
        silently returning nothing."""
        raw_text = raw_text.strip()

        # Attempt 1: find a JSON array and parse it properly
        match = re.search(r"\[.*\]", raw_text, re.DOTALL)
        if match:
            try:
                facts = json.loads(match.group(0))
                parsed = [f.strip() for f in facts if isinstance(f, str) and f.strip()]
                if parsed:
                    return parsed
            except json.JSONDecodeError:
                pass

        # Attempt 2: fallback — salvage facts from bullet/numbered/quoted lines
        fallback_facts = []
        for line in raw_text.splitlines():
            line = line.strip()
            line = re.sub(r'^[-*\d.\)]+\s*', '', line)
            line = line.strip('"\'')
            if line and len(line) > 5 and not line.lower().startswith(("json", "here", "no facts", "[")):
                fallback_facts.append(line)

        return fallback_facts
