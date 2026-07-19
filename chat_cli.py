"""
chat_cli.py — simple command-line interface for local testing.

Run: python chat_cli.py
Type 'exit' to quit, 'memories' to dump everything currently stored.
"""

from chat_engine import ChatEngine


def main():
    print("Long-Term Memory Chatbot (CLI mode)")
    print("Type 'exit' to quit, 'memories' to inspect stored long-term memories.\n")

    engine = ChatEngine()

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "exit":
            break
        if user_input.lower() == "memories":
            facts = engine.store.get_all_memories(memory_type="fact")
            print(f"\n--- {len(facts)} stored facts ---")
            for f in facts:
                print(f" • {f['text']}")
            print()
            continue

        result = engine.chat(user_input)

        print(f"\nAssistant: {result['reply']}\n")

        if result["retrieved_memories"]:
            print("[debug] Retrieved memories used for context:")
            for m in result["retrieved_memories"]:
                print(f"   - {m['text']}  (score={m['similarity_score']:.3f})")

        if result["new_facts_extracted"]:
            print("[debug] New facts extracted this turn:")
            for f in result["new_facts_extracted"]:
                print(f"   + {f}")

        if result["consolidation"].get("ran"):
            c = result["consolidation"]
            print(f"[debug] Consolidation ran: {c['facts_before']} facts -> {c['facts_after']} facts")

        print()


if __name__ == "__main__":
    main()
