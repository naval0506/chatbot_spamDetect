"""
Interface Streamlit — lancer : streamlit run app.py
"""

import streamlit as st, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from chatbot_model import ChatbotBanking

TEST_PROMPTS = [
    "I lost my card, what should I do?",
    "How long does a bank transfer take?",
    "Why is my top up still pending?",
    "I forgot my passcode",
    "Can I use Apple Pay with my card?",
]

st.set_page_config(page_title="Bannki", page_icon="🏦",
                   layout="centered", initial_sidebar_state="expanded")

st.markdown("""
<style>
.hero{background:linear-gradient(135deg,#1a237e,#1565c0);color:#fff;padding:22px 28px;
      border-radius:14px;text-align:center;margin-bottom:20px}
.hero h2{margin:0 0 4px;font-size:1.55rem}
.hero p{margin:0;opacity:.85;font-size:.9rem}
.badge{display:inline-block;background:#e3f2fd;color:#1565c0;
       border-radius:8px;padding:2px 10px;font-size:.78rem;font-weight:700}
.meta{color:#9e9e9e;font-size:.79rem;margin-top:3px}
.prompt-grid{display:grid;grid-template-columns:1fr;gap:8px;margin:8px 0 14px}
</style>""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading models...")
def get_bot():
    b = ChatbotBanking()
    b.initialiser()
    return b

# ── Session state ──
if "msgs" not in st.session_state:
    st.session_state.msgs = [{
        "role": "assistant", "content":
        "Ask a banking question. I will detect the intent from Banking77 and show the closest examples.", "meta": {}}]
if "n_q" not in st.session_state:
    st.session_state.n_q = 0
if "n_ok" not in st.session_state:
    st.session_state.n_ok = 0

# ── Sidebar Info ──
with st.sidebar:
    st.markdown("### 📊 Stats")
    c1, c2 = st.columns(2)
    c1.metric("Questions", st.session_state.n_q)
    taux = int(st.session_state.n_ok / max(st.session_state.n_q, 1) * 100)
    c2.metric("Resolved", f"{taux}%")
    st.divider()
    st.markdown("### 🧪 Try")
    for i, prompt in enumerate(TEST_PROMPTS):
        if st.button(prompt, key=f"test_prompt_{i}", use_container_width=True):
            st.session_state.quick = prompt
            st.rerun()
    st.divider()
    st.markdown("### 📂 Categories")
    try:
        bot = get_bot()
        cats = bot.categories()
        cat = st.selectbox("Filter by category", ["— all —"] + cats)
        if cat != "— all —":
            qs = bot.questions_par_cat(cat)
            for q in qs:
                label = (q[:44] + "…") if len(q) > 45 else q
                if st.button(f"❓ {label}", key=q):
                    st.session_state.quick = q
    except Exception as e:
        st.error(f"⚠️ Error: {e}")
        st.stop()
    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.msgs = [{"role": "assistant", "content": "Chat cleared!", "meta": {}}]
        st.rerun()

# ── Hero ──
st.markdown("""
<div class="hero">
  <h2>🏦 Bannki</h2>
  <p>Powered by Banking77 Dataset · TF-IDF &amp; LinearSVC</p>
</div>""", unsafe_allow_html=True)

# ── Chat history ──
for m in st.session_state.msgs:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        meta = m.get("meta", {})
        if meta.get("score") and meta.get("categorie"):
            src = meta.get("source", "faq")
            st.markdown(
                f'<span class="badge">📌 {meta["categorie"]}</span>'
                f'<span class="meta"> · {meta["score"]:.0%} confidence ({src})</span>',
                unsafe_allow_html=True)
        if meta.get("question_matchee"):
            st.caption(f"Closest example: {meta['question_matchee']}")
        if meta.get("suggestions"):
            st.caption("Related examples: " + " · ".join(f'*{s[:65]}*' for s in meta["suggestions"]))

# ── Input ──
quick = st.session_state.pop("quick", None) if "quick" in st.session_state else None
user_in = st.chat_input("How can I help you today?")
to_do = user_in or quick

if to_do:
    st.session_state.msgs.append({"role": "user", "content": to_do, "meta": {}})
    with st.chat_message("user"):
        st.markdown(to_do)
    with st.chat_message("assistant"):
        bot = get_bot()
        r = bot.repondre(to_do)
        st.markdown(r["reponse"])
        if r.get("score") and r.get("categorie"):
            st.markdown(
                f'<span class="badge">📌 {r["categorie"]}</span>'
                f'<span class="meta"> · {r["score"]:.0%} confidence ({r["source"]})</span>',
                unsafe_allow_html=True)
        if r.get("question_matchee"):
            st.caption(f"Closest example: {r['question_matchee']}")
        if r.get("suggestions"):
            st.caption("Related examples: " + " · ".join(f'*{s[:65]}*' for s in r["suggestions"]))
        st.session_state.n_q += 1
        if r.get("source") != "not_found":
            st.session_state.n_ok += 1
        st.session_state.msgs.append({"role": "assistant", "content": r["reponse"], "meta": r})
    st.rerun()
