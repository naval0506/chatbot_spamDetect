"""
Interface Streamlit — lancer : streamlit run app.py
Pages : 💬 Chatbot  |  ⚙️ Model Training  |  🔍 Model Validation
"""

import streamlit as st, sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from chatbot_model import ChatbotBanking

st.set_page_config(page_title="Banking Chatbot", page_icon="🏦",
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
</style>""", unsafe_allow_html=True)

# ── Navigation ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧭 Navigation")
    page = st.radio("Go to", ["💬 Chatbot", "⚙️ Model Training", "🔍 Model Validation"])
    st.divider()

# ════════════════════════════════════════════════════════════════════════════════
#  PAGE 1 : CHATBOT
# ════════════════════════════════════════════════════════════════════════════════
if page == "💬 Chatbot":
    @st.cache_resource(show_spinner="Loading models...")
    def get_bot():
        b = ChatbotBanking()
        b.initialiser()
        return b

    # ── Session state ──
    if "msgs" not in st.session_state:
        st.session_state.msgs = [{
            "role": "assistant", "content":
            "👋 Hello! I'm your **banking assistant**. "
            "Ask me anything about cards, transfers, top-up and more!", "meta": {}}]
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
      <h2>🏦 Banking Virtual Assistant</h2>
      <p>Powered by Banking77 Dataset · TF-IDF &amp; Logistic Regression</p>
    </div>""", unsafe_allow_html=True)

    # ── Chat history ──
    for m in st.session_state.msgs:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            meta = m.get("meta", {})
            if meta.get("score"):
                src = meta.get("source", "faq")
                st.markdown(
                    f'<span class="badge">📌 {meta["categorie"]}</span>'
                    f'<span class="meta"> · {meta["score"]:.0%} confidence ({src})</span>',
                    unsafe_allow_html=True)

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
            if r.get("score"):
                st.markdown(
                    f'<span class="badge">📌 {r["categorie"]}</span>'
                    f'<span class="meta"> · {r["score"]:.0%} confidence ({r["source"]})</span>',
                    unsafe_allow_html=True)
            if r.get("suggestions"):
                st.caption("Related: " + " · ".join(f'*{s[:55]}*' for s in r["suggestions"]))
            st.session_state.n_q += 1
            if r.get("source") != "not_found":
                st.session_state.n_ok += 1
            st.session_state.msgs.append({"role": "assistant", "content": r["reponse"], "meta": r})
        st.rerun()

# ════════════════════════════════════════════════════════════════════════════════
#  PAGE 2 : MODEL TRAINING
# ════════════════════════════════════════════════════════════════════════════════
elif page == "⚙️ Model Training":
    from trainer import ModelTrainer
    import pandas as pd
    import plotly.express as px

    st.title("⚙️ Model Training Dashboard")
    st.markdown("Train and evaluate your chatbot model on the Banking77 dataset.")

    TRAIN_PATH = "data/train.csv"
    TEST_PATH  = "data/test.csv"
    TMP_MODEL  = "data/tmp_model.pkl"
    PROD_MODEL = "data/classifier_model.pkl"

    if not os.path.exists(TRAIN_PATH):
        st.warning("Dataset missing! Run `python setup.py` first.")
        st.stop()

    df_train = pd.read_csv(TRAIN_PATH)
    df_test  = pd.read_csv(TEST_PATH)

    c1, c2, c3 = st.columns(3)
    c1.metric("Train Samples", len(df_train))
    c2.metric("Test Samples", len(df_test))
    c3.metric("Categories", df_train['label'].nunique())

    st.divider()

    if "trainer" not in st.session_state:
        st.session_state.trainer = ModelTrainer()

    # ── Train button ──
    if st.button("🚀 Start Training", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(p, text):
            progress_bar.progress(p)
            status_text.text(text)

        metrics = st.session_state.trainer.train(
            df_train, df_test, progress_callback=update_progress)
        st.session_state.trainer.save(TMP_MODEL)
        st.success("✅ Model trained and saved to temporary storage!")

    # ── Show metrics if model exists ──
    if os.path.exists(TMP_MODEL):
        # Reload metrics if trainer state is empty (page refresh)
        if not st.session_state.trainer.metrics:
            st.session_state.trainer.load(TMP_MODEL)

        metrics = st.session_state.trainer.metrics
        if metrics:
            st.markdown("### 📊 Evaluation Metrics")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy",  f"{metrics['accuracy']:.2%}")
            m2.metric("Precision", f"{metrics['precision']:.2%}")
            m3.metric("Recall",    f"{metrics['recall']:.2%}")
            m4.metric("F1-Score",  f"{metrics['f1_score']:.2%}")

            # ── Confusion Matrix ──
            if st.session_state.trainer.cm is not None:
                st.markdown("### 📉 Confusion Matrix (Top 15 categories)")
                cm = st.session_state.trainer.cm[:15, :15]
                labels = st.session_state.trainer.classes[:15]
                fig = px.imshow(cm, x=labels, y=labels, text_auto=True,
                                color_continuous_scale="Blues", aspect="auto")
                fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig, use_container_width=True)

            # ── Per-class metrics bar chart ──
            if st.session_state.trainer.per_class_metrics is not None:
                st.markdown("### 📊 Per-Category F1-Score")
                df_pc = st.session_state.trainer.per_class_metrics.sort_values(
                    "f1_score", ascending=True).tail(20)
                fig2 = px.bar(df_pc, x="f1_score", y="category",
                              orientation='h', color="f1_score",
                              color_continuous_scale="Viridis",
                              labels={"f1_score": "F1-Score", "category": ""})
                fig2.update_layout(height=500, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig2, use_container_width=True)

            st.divider()

            # ── Deploy ──
            if metrics['precision'] > 0.85:
                st.balloons()
                st.success("✅ High precision detected! The model is ready for production.")
                if st.button("📥 Deploy to Production", type="primary",
                             use_container_width=True):
                    import shutil
                    shutil.copy(TMP_MODEL, PROD_MODEL)
                    st.success("Model deployed! It is now active in the Chat interface.")
                    if "get_bot" in st.session_state:
                        del st.session_state.get_bot
            else:
                st.warning("⚠️ Precision is below 85%. You might want to tune the model further.")

# ════════════════════════════════════════════════════════════════════════════════
#  PAGE 3 : MODEL VALIDATION
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Model Validation":
    from trainer import ModelTrainer
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go

    st.title("🔍 Model Validation")
    st.markdown(
        "Validate your model with **stratified k-fold cross-validation** "
        "and view detailed per-class metrics.")

    TRAIN_PATH = "data/train.csv"
    TEST_PATH  = "data/test.csv"
    TMP_MODEL  = "data/tmp_model.pkl"

    if not os.path.exists(TRAIN_PATH):
        st.warning("Dataset missing! Run `python setup.py` first.")
        st.stop()

    df_train = pd.read_csv(TRAIN_PATH)
    df_test  = pd.read_csv(TEST_PATH)

    if "trainer" not in st.session_state:
        st.session_state.trainer = ModelTrainer()

    # Load existing model if available
    if os.path.exists(TMP_MODEL) and not st.session_state.trainer.metrics:
        st.session_state.trainer.load(TMP_MODEL)

    # ── Cross-Validation ──────────────────────────────────────────────────────
    st.markdown("### 🔄 Cross-Validation")
    cv_folds = st.slider("Number of folds (k)", min_value=2, max_value=10, value=5)

    if st.button("▶️ Run Cross-Validation", type="primary", use_container_width=True):
        X = pd.concat([df_train['text'], df_test['text']], ignore_index=True)
        y = pd.concat([df_train['label'], df_test['label']], ignore_index=True)

        progress_bar = st.progress(0)
        status_text = st.empty()

        def cv_progress(p, text):
            progress_bar.progress(p)
            status_text.text(text)

        cv_metrics = st.session_state.trainer.validate(
            X, y, cv=cv_folds, progress_callback=cv_progress)

        st.success(f"✅ {cv_folds}-fold cross-validation completed!")

        # Store in session state for persistence
        st.session_state.cv_metrics = cv_metrics

    # ── Display CV Results ────────────────────────────────────────────────────
    cv_metrics = st.session_state.get("cv_metrics",
                                       st.session_state.trainer.cv_metrics)
    if cv_metrics:
        st.markdown("### 📊 Cross-Validation Results")

        cv1, cv2, cv3, cv4 = st.columns(4)
        cv1.metric("Accuracy",
                    f"{cv_metrics['accuracy_mean']:.2%}",
                    f"± {cv_metrics['accuracy_std']:.2%}")
        cv2.metric("Precision",
                    f"{cv_metrics['precision_mean']:.2%}",
                    f"± {cv_metrics['precision_std']:.2%}")
        cv3.metric("Recall",
                    f"{cv_metrics['recall_mean']:.2%}",
                    f"± {cv_metrics['recall_std']:.2%}")
        cv4.metric("F1-Score",
                    f"{cv_metrics['f1_mean']:.2%}",
                    f"± {cv_metrics['f1_std']:.2%}")

        # ── Fold-by-fold accuracy chart ──
        if 'fold_accuracies' in cv_metrics:
            st.markdown("### 📈 Accuracy per Fold")
            fold_data = pd.DataFrame({
                "Fold": [f"Fold {i+1}" for i in range(len(cv_metrics['fold_accuracies']))],
                "Accuracy": cv_metrics['fold_accuracies']
            })
            fig = px.bar(fold_data, x="Fold", y="Accuracy",
                         color="Accuracy", color_continuous_scale="Tealgrn",
                         text_auto=".2%")
            fig.add_hline(y=cv_metrics['accuracy_mean'], line_dash="dash",
                          line_color="red",
                          annotation_text=f"Mean: {cv_metrics['accuracy_mean']:.2%}")
            fig.update_layout(yaxis_range=[0, 1],
                              margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)

    # ── Classification Report ─────────────────────────────────────────────────
    if st.session_state.trainer.classification_report_text:
        st.markdown("### 📋 Detailed Classification Report")
        with st.expander("View full classification report"):
            st.code(st.session_state.trainer.classification_report_text)

    # ── Per-class metrics table ───────────────────────────────────────────────
    if st.session_state.trainer.per_class_metrics is not None:
        st.markdown("### 📊 Per-Category Performance")
        df_pc = st.session_state.trainer.per_class_metrics
        st.dataframe(
            df_pc.style.background_gradient(
                subset=["precision", "recall", "f1_score"],
                cmap="RdYlGn", vmin=0, vmax=1),
            use_container_width=True, height=400)

        # Bottom 10 categories
        st.markdown("### ⚠️ Weakest Categories (Bottom 10 by F1)")
        bottom = df_pc.nsmallest(10, "f1_score")
        fig3 = px.bar(bottom, x="f1_score", y="category", orientation="h",
                       color="f1_score", color_continuous_scale="OrRd_r",
                       labels={"f1_score": "F1-Score", "category": ""})
        fig3.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig3, use_container_width=True)

    st.divider()

    # ── Export Report ─────────────────────────────────────────────────────────
    if st.session_state.trainer.metrics:
        if st.button("📄 Export Markdown Report", use_container_width=True):
            report_path = st.session_state.trainer.export_report(
                "data/training_report.md")
            st.success(f"Report exported to `{report_path}`")
            with open(report_path, 'r') as f:
                st.download_button("⬇️ Download Report", f.read(),
                                   file_name="training_report.md",
                                   mime="text/markdown")
