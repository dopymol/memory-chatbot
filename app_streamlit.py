"""
app_streamlit.py — polished demo UI.

Run: streamlit run app_streamlit.py

Shows the chat plus a sidebar that visualizes what the memory system
is doing under the hood (retrieved memories, newly extracted facts,
consolidation events). This "show your work" panel is what makes the
project readable to a non-technical interviewer skimming a screen
recording, and credible to a technical one.
"""

import streamlit as st
from chat_engine import ChatEngine

st.set_page_config(page_title="Memory Chatbot", page_icon="🧠", layout="wide")

if "engine" not in st.session_state:
    st.session_state.engine = ChatEngine()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_debug" not in st.session_state:
    st.session_state.last_debug = None

st.title("🧠 Chatbot with Long-Term Memory")
st.caption("Local & free: Ollama LLM + sentence-transformers embeddings + ChromaDB")

col_chat, col_memory = st.columns([2, 1])

with col_chat:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Say something...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.engine.chat(user_input)
            st.write(result["reply"])

        st.session_state.messages.append({"role": "assistant", "content": result["reply"]})
        st.session_state.last_debug = result
        st.rerun()

with col_memory:
    st.subheader("Memory system internals")

    debug = st.session_state.last_debug
    if debug is None:
        st.info("Chat to see retrieval & extraction in action.")
    else:
        st.markdown("**🔎 Memories retrieved this turn**")
        if debug["retrieved_memories"]:
            for m in debug["retrieved_memories"]:
                st.write(f"- {m['text']}  \n  `similarity: {m['similarity_score']:.3f}`")
        else:
            st.write("_None retrieved_")

        st.markdown("**✍️ New facts extracted**")
        if debug["new_facts_extracted"]:
            for f in debug["new_facts_extracted"]:
                st.write(f"- {f}")
        else:
            st.write("_None this turn_")

        if debug["consolidation"].get("ran"):
            c = debug["consolidation"]
            st.success(f"🗜️ Consolidation ran: {c['facts_before']} → {c['facts_after']} facts")

    st.divider()
    st.subheader("All stored long-term facts")
    all_facts = st.session_state.engine.store.get_all_memories(memory_type="fact")
    st.write(f"Total: {len(all_facts)}")
    for f in all_facts:
        st.caption(f"• {f['text']}")
