import os
import json
import requests
import pandas as pd
import streamlit as st
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── RAG imports ───────────────────────────────────────────────────────────────
try:
    import chromadb
    from chromadb.utils import embedding_functions
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Career Roadmap & Advisor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# OLLAMA — Free local AI (no API key needed)
# Install: https://ollama.com  →  then run: ollama pull llama3
# ─────────────────────────────────────────────────────────────────────────────
OLLAMA_URL  = "http://localhost:11434/api/generate"
OLLAMA_CHAT = "http://localhost:11434/api/chat"
OLLAMA_TAGS = "http://localhost:11434/api/tags"
DEFAULT_MODEL = "llama3"          # change to "mistral", "phi3", etc. if preferred

@st.cache_data(show_spinner=False)
def get_ollama_models() -> List[str]:
    """Return list of locally installed Ollama models."""
    try:
        r = requests.get(OLLAMA_TAGS, timeout=3)
        if r.status_code == 200:
            return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        pass
    return []

def ollama_available() -> bool:
    return len(get_ollama_models()) > 0

def ollama_stream(prompt: str, model: str, system: str = "") -> str:
    """Stream a response from Ollama; returns full text."""
    payload = {"model": model, "prompt": prompt, "stream": True}
    if system:
        payload["system"] = system
    try:
        with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=120) as r:
            r.raise_for_status()
            chunks = []
            for line in r.iter_lines():
                if line:
                    data = json.loads(line)
                    chunks.append(data.get("response", ""))
                    if data.get("done"):
                        break
            return "".join(chunks)
    except requests.exceptions.ConnectionError:
        return "__OFFLINE__"
    except Exception as e:
        return f"__ERROR__: {e}"

def ollama_chat_stream(messages: List[dict], model: str, system: str = ""):
    """Generator that yields text chunks for streaming chat responses."""
    payload = {"model": model, "messages": messages, "stream": True}
    if system:
        payload["system"] = system
    try:
        with requests.post(OLLAMA_CHAT, json=payload, stream=True, timeout=180) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line:
                    data = json.loads(line)
                    chunk = data.get("message", {}).get("content", "")
                    if chunk:
                        yield chunk
                    if data.get("done"):
                        break
    except requests.exceptions.ConnectionError:
        yield "\n\n⚠️ **Ollama is not running.** Start it with: `ollama serve`"
    except Exception as e:
        yield f"\n\n⚠️ Error: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(csv_path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    expected = {
        "Career", "Category", "InterestKeywords", "Description", "IdealAudience",
        "EntryPaths", "CoreSkills", "ToolsTech", "Roadmap", "12WeekPlan"
    }
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    def nz(x):
        return "" if pd.isna(x) else str(x)

    df["_combined_text"] = (
        df["InterestKeywords"].apply(nz) + " \n" +
        df["Description"].apply(nz) + " \n" +
        df["CoreSkills"].apply(nz) + " \n" +
        df["ToolsTech"].apply(nz) + " \n" +
        df["Career"].apply(nz) + " \n" +
        df["Category"].apply(nz)
    )
    df["RoadmapList"] = df["Roadmap"].fillna("").apply(
        lambda s: [x.strip() for x in str(s).split("|") if x.strip()])
    df["WeekPlanList"] = df["12WeekPlan"].fillna("").apply(
        lambda s: [x.strip() for x in str(s).split("|") if x.strip()])
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. TF-IDF
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def build_vectorizer(texts: List[str], ngram: Tuple[int, int] = (1, 2)):
    vect = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=ngram,
        min_df=1,
        max_df=0.9,
    )
    X = vect.fit_transform(texts)
    return vect, X


# ─────────────────────────────────────────────────────────────────────────────
# 3. RAG — ChromaDB
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def build_rag_store(_df: pd.DataFrame):
    if not RAG_AVAILABLE:
        return None
    try:
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        client = chromadb.Client()
        try:
            client.delete_collection("careers")
        except Exception:
            pass
        collection = client.create_collection(
            name="careers",
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )
        docs, ids, metas = [], [], []
        for i, row in _df.iterrows():
            doc = (
                f"Career: {row['Career']}\n"
                f"Category: {row['Category']}\n"
                f"Description: {row['Description']}\n"
                f"Core Skills: {row['CoreSkills']}\n"
                f"Tools: {row['ToolsTech']}\n"
                f"Keywords: {row['InterestKeywords']}"
            )
            docs.append(doc)
            ids.append(str(i))
            metas.append({"career": row["Career"], "category": row["Category"], "idx": i})

        batch_size = 500
        for start in range(0, len(docs), batch_size):
            collection.add(
                documents=docs[start:start + batch_size],
                ids=ids[start:start + batch_size],
                metadatas=metas[start:start + batch_size],
            )
        return collection
    except Exception as e:
        st.warning(f"RAG index build failed: {e}")
        return None


def rag_search(collection, query: str, n_results: int = 10, category_filter: str = None):
    if collection is None or not query.strip():
        return []
    where = {"category": category_filter} if category_filter and category_filter != "(All)" else None
    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where,
        )
        return [meta["idx"] for meta in results["metadatas"][0]]
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# 4. AI GENERATION — for unknown careers & personalised plans
# ─────────────────────────────────────────────────────────────────────────────
def ai_generate_career_card(career_name: str, model: str) -> dict:
    """Ask Ollama to generate a full career card for any career not in the CSV."""
    system = (
        "You are a career advisor. When given a career title, respond ONLY with a "
        "valid JSON object — no markdown, no explanation, no extra text. "
        "Use exactly these keys: "
        "Career, Category, Description, IdealAudience, EntryPaths, "
        "CoreSkills, ToolsTech, InterestKeywords, Roadmap, 12WeekPlan. "
        "For Roadmap and 12WeekPlan use pipe '|' as separator between steps/weeks."
    )
    prompt = (
        f"Generate a complete career profile for: {career_name}\n\n"
        "Roadmap should have 5-6 stages separated by |.\n"
        "12WeekPlan should have exactly 12 weekly items separated by |, "
        "each starting with 'Week N: ...'."
    )
    raw = ollama_stream(prompt, model, system)
    if raw.startswith("__"):
        return {}
    try:
        # Strip any accidental markdown fences
        clean = raw.strip().strip("```json").strip("```").strip()
        return json.loads(clean)
    except Exception:
        return {}


def ai_generate_plan(career: str, goal: str, focus: str, audience: str,
                     base_weeks: List[str], model: str) -> str:
    """Generate a fully personalised 12-week plan using AI."""
    system = (
        "You are an expert career coach. Create practical, specific, "
        "week-by-week plans. Be concrete — name real tools, resources, and actions."
    )
    base_text = "\n".join(f"- {w}" for w in base_weeks) if base_weeks else "No base plan available."
    prompt = (
        f"Career: {career}\n"
        f"Person: {audience}\n"
        f"Goal: {goal}\n"
        f"Special focus: {focus if focus else 'None'}\n\n"
        f"Base 12-week plan from our database:\n{base_text}\n\n"
        "Create a fully personalised 12-week plan that:\n"
        "1. Integrates the person's specific goal throughout\n"
        "2. Adds concrete resources, tools, and milestones\n"
        "3. Adapts to the special focus if provided\n"
        "4. Ends with a clear outcome for Week 12\n\n"
        "Format: Week 1: ... Week 2: ... (one per line)"
    )
    result = ollama_stream(prompt, model, system)
    if result.startswith("__OFFLINE__"):
        return None
    return result


def ai_career_advice(messages: List[dict], df: pd.DataFrame, model: str):
    """Streaming chat that uses the CSV as context."""
    career_list = ", ".join(df["Career"].tolist()[:80]) + " (and more...)"
    categories  = ", ".join(df["Category"].unique().tolist())
    system = (
        "You are a friendly, expert career advisor. You have access to a database of "
        f"261+ careers across these categories: {categories}.\n\n"
        f"Some example careers in the database: {career_list}\n\n"
        "Help users explore careers, understand what skills they need, "
        "how to get started, what salary ranges look like, and how to transition. "
        "Be encouraging, specific, and practical. "
        "If the user mentions a career not in the list, you can still advise on it. "
        "Keep responses concise but helpful — use bullet points when listing steps or skills."
    )
    return ollama_chat_stream(messages, model, system)


# ─────────────────────────────────────────────────────────────────────────────
# 5. SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title("🎯 Career Roadmap & Advisor")

# ── AI status banner ──────────────────────────────────────────────────────────
with st.sidebar.expander("🤖 AI Status (Ollama)", expanded=True):
    models = get_ollama_models()
    if models:
        selected_model = st.selectbox("Model", models,
                                      index=models.index(DEFAULT_MODEL)
                                      if DEFAULT_MODEL in models else 0)
        st.success(f"✅ Ollama running — {len(models)} model(s) available")
    else:
        selected_model = DEFAULT_MODEL
        st.warning(
            "Ollama not detected. AI features will be disabled.\n\n"
            "**To enable AI:**\n"
            "1. Install: https://ollama.com\n"
            "2. Run: `ollama serve`\n"
            "3. Pull a model: `ollama pull llama3`\n"
            "4. Refresh this page"
        )
    AI_ON = len(models) > 0

search_mode = st.sidebar.radio(
    "Search mode",
    ["🔍 TF-IDF (keyword)", "🧠 RAG — semantic search"],
    help=(
        "TF-IDF: fast keyword matching.\n\n"
        "RAG: understands meaning, not just keywords. "
        "Needs: pip install chromadb sentence-transformers"
    )
)
use_rag = "RAG" in search_mode

if use_rag and not RAG_AVAILABLE:
    st.sidebar.error(
        "RAG not available. Install:\n\n"
        "```\npip install chromadb sentence-transformers\n```\n"
        "Then restart the app."
    )
    use_rag = False

st.sidebar.caption(
    "Powered by TF-IDF similarity" if not use_rag
    else "Powered by all-MiniLM-L6-v2 embeddings + ChromaDB"
)

with st.sidebar.expander("📁 Load Dataset", expanded=True):
    use_upload = st.toggle("Upload a CSV instead of default", value=False)
    data_path = "career_interest.csv"
    if use_upload:
        uploaded = st.file_uploader("Upload CSV with the expected columns", type=["csv"])
        if uploaded is not None:
            data_path = uploaded
    try:
        df = load_data(data_path)
        st.success(f"Loaded {len(df)} careers.")
    except Exception as e:
        st.error("Could not load dataset.")
        st.exception(e)
        st.stop()

categories = ["(All)"] + sorted(df["Category"].dropna().unique().tolist())
sel_category = st.sidebar.selectbox("Filter by Category", categories, index=0)
k_topn = st.sidebar.slider("How many recommendations?", 5, 50, 10, 1)

if not use_rag:
    k_weight = st.sidebar.slider("Weight: Interest Keywords", 0.0, 2.0, 1.0, 0.1)
    d_weight = st.sidebar.slider("Weight: Description/Skills/Tools", 0.0, 2.0, 1.0, 0.1)
    st.sidebar.markdown("---")
    with st.sidebar.expander("⚙️ Advanced"):
        ngram_min, ngram_max = st.select_slider(
            "TF-IDF n-gram range", options=[1, 2, 3], value=(1, 2))
else:
    k_weight, d_weight = 1.0, 1.0
    ngram_min, ngram_max = 1, 2


# ─────────────────────────────────────────────────────────────────────────────
# 6. BUILD INDEXES
# ─────────────────────────────────────────────────────────────────────────────
keywords_corpus = df["InterestKeywords"].fillna("").astype(str).tolist()
combined_corpus = df["_combined_text"].fillna("").astype(str).tolist()
kvect, KX = build_vectorizer(keywords_corpus, ngram=(ngram_min, ngram_max))
cvect, CX = build_vectorizer(combined_corpus, ngram=(ngram_min, ngram_max))

if use_rag:
    with st.spinner("Building semantic index — first run takes ~30s to download model..."):
        rag_collection = build_rag_store(df)


# ─────────────────────────────────────────────────────────────────────────────
# 7. TABS — Main navigation
# ─────────────────────────────────────────────────────────────────────────────
tab_search, tab_chat, tab_generate, tab_plan, tab_browse = st.tabs([
    "🔍 Career Search",
    "💬 AI Career Chat",
    "✨ Generate Any Career",
    "🗓️ Personalise Plan",
    "📚 Browse Dataset",
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — CAREER SEARCH (original, preserved)
# ═════════════════════════════════════════════════════════════════════════════
with tab_search:
    st.title("🎓 Career Roadmap & Advisor")

    if use_rag:
        st.info(
            "**RAG mode active** — describe your situation in plain English.\n\n"
            "Example: *'I like understanding why people behave the way they do'* "
            "→ finds Psychology, Counselling, UX Research without matching exact words."
        )
    else:
        st.write("Enter your interests or background to get matched careers, roadmaps, and 12-week plans.")

    query = st.text_area(
        "Describe your interests, skills, or goals:",
        placeholder=(
            "e.g. 'I enjoy building AI tools, working with APIs, and want to get into ML engineering'"
            if not use_rag else
            "e.g. 'I want to work with data but I prefer analysing trends and presenting insights over coding'"
        ),
        height=100,
    )
    audience = st.text_input("Who is the plan for?", value="student",
                              placeholder="student / career switcher / early professional")

    def tfidf_scores(user_query: str):
        if not user_query.strip():
            return pd.Series([0.0] * len(df))
        q_kw = kvect.transform([user_query])
        q_cb = cvect.transform([user_query])
        s_kw = cosine_similarity(q_kw, KX).ravel()
        s_cb = cosine_similarity(q_cb, CX).ravel()
        return pd.Series(k_weight * s_kw + d_weight * s_cb)

    if use_rag and query.strip():
        rag_idx = rag_search(
            rag_collection, query, n_results=k_topn,
            category_filter=sel_category if sel_category != "(All)" else None
        )
        if rag_idx:
            res = df.iloc[rag_idx].copy()
            res["Similarity"] = [1.0 - (i / max(len(rag_idx), 1)) * 0.4 for i in range(len(rag_idx))]
            res["_mode"] = "RAG"
        else:
            res = pd.DataFrame()
    else:
        scores = tfidf_scores(query)
        res = df.copy()
        res["Similarity"] = scores
        res["_mode"] = "TF-IDF"
        if sel_category != "(All)":
            res = res[res["Category"] == sel_category]
        res = res.sort_values("Similarity", ascending=False).head(k_topn)

    if use_rag and query.strip() and len(res) > 0:
        with st.expander("🔎 How RAG found these results", expanded=False):
            st.caption(
                "Your query was converted into a 384-dimension semantic vector using "
                "`all-MiniLM-L6-v2`. ChromaDB found the nearest career profiles using "
                "cosine similarity. Top 3 retrieved chunks:"
            )
            for _, row in res.head(3).iterrows():
                st.markdown(f"**{row['Career']}** — {row['Category']}")
                st.markdown(
                    f"> *Keywords:* {str(row['InterestKeywords'])[:120]}...  \n"
                    f"> *Skills:* {str(row['CoreSkills'])[:100]}..."
                )
                st.markdown("---")

    st.subheader("🔎 Recommendations")

    if len(res) == 0:
        st.info("Enter a query above to see recommendations, or relax your filters.")
    else:
        for _, row in res.iterrows():
            sim_pct = f"{row['Similarity'] * 100:0.1f}%"
            badge = "🧠 RAG" if row.get("_mode") == "RAG" else "📊 TF-IDF"
            with st.container(border=True):
                top = st.columns([0.7, 0.3])
                with top[0]:
                    st.markdown(f"### **{row['Career']}**")
                    st.markdown(f"**Category:** {row['Category']}  &nbsp;&nbsp; {badge}")
                    st.markdown(f"**Match:** {sim_pct}")
                    st.markdown(f"**Ideal Audience:** {row['IdealAudience']}")
                    st.markdown(f"**Entry Paths:** {row['EntryPaths']}")
                with top[1]:
                    st.progress(min(max(float(row["Similarity"]), 0.0), 1.0))

                st.markdown("---")
                tabs = st.tabs(["Overview", "Roadmap", "12-Week Plan"])
                with tabs[0]:
                    st.markdown(f"**Description**\n\n{row['Description']}")
                    st.markdown(f"**Core Skills:** {row['CoreSkills']}")
                    st.markdown(f"**Tools/Tech:** {row['ToolsTech']}")
                    st.markdown(f"**Interest Keywords:** {row['InterestKeywords']}")
                with tabs[1]:
                    for i, step in enumerate(row["RoadmapList"], start=1):
                        st.markdown(f"**{i}.** {step}")
                with tabs[2]:
                    for i, wk in enumerate(row["WeekPlanList"], start=1):
                        with st.expander(f"Week {i}"):
                            st.write(wk)

        export_cols = [
            "Career", "Category", "Similarity", "Description", "CoreSkills",
            "ToolsTech", "InterestKeywords", "Roadmap", "12WeekPlan",
            "IdealAudience", "EntryPaths"
        ]
        export_df = res[[c for c in export_cols if c in res.columns]].copy()
        st.download_button(
            label="⬇️ Download recommendations (CSV)",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name="career_recommendations.csv",
            mime="text/csv",
        )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — AI CAREER CHAT (NEW)
# ═════════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.header("💬 AI Career Advisor Chat")

    if not AI_ON:
        st.warning(
            "AI chat requires Ollama. Install it at https://ollama.com, "
            "then run `ollama serve` and `ollama pull llama3`."
        )
    else:
        st.caption(
            f"Talking to **{selected_model}** — running locally, free, no API key. "
            "Ask anything about careers, skills, transitions, salaries, or paths."
        )

        # Init chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Display history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Input
        if prompt := st.chat_input("Ask the career advisor anything..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                full_response = ""
                for chunk in ai_career_advice(
                    st.session_state.chat_history, df, selected_model
                ):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)

            st.session_state.chat_history.append(
                {"role": "assistant", "content": full_response}
            )

        if st.session_state.chat_history:
            if st.button("🗑️ Clear chat", key="clear_chat"):
                st.session_state.chat_history = []
                st.rerun()

        # Starter prompts
        if not st.session_state.get("chat_history"):
            st.markdown("**Try asking:**")
            starters = [
                "What careers suit someone who loves both art and technology?",
                "How do I transition from nursing to UX design?",
                "What's the salary range for a Data Scientist in 2025?",
                "I'm 30 with no degree — what tech careers are realistic for me?",
                "Compare Software Engineer vs Product Manager career paths",
            ]
            for s in starters:
                if st.button(s, key=f"starter_{s[:20]}"):
                    st.session_state.chat_history = [{"role": "user", "content": s}]
                    st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — GENERATE ANY CAREER (NEW)
# ═════════════════════════════════════════════════════════════════════════════
with tab_generate:
    st.header("✨ Generate a Career Card for Any Job")
    st.write(
        "Can't find your career in the database? Enter any job title and AI will "
        "generate a complete profile — roadmap, skills, tools, and 12-week plan."
    )

    if not AI_ON:
        st.warning("This feature requires Ollama. See the AI Status panel in the sidebar.")
    else:
        gen_career = st.text_input(
            "Career title",
            placeholder="e.g. Marine Biologist, Blockchain Developer, Forensic Accountant..."
        )

        col_a, col_b = st.columns([0.3, 0.7])
        with col_a:
            gen_btn = st.button("🚀 Generate Career Card", type="primary",
                                disabled=not gen_career.strip())

        if gen_btn and gen_career.strip():
            # Check if already in CSV
            existing = df[df["Career"].str.lower() == gen_career.strip().lower()]
            if not existing.empty:
                st.info(
                    f"**{gen_career}** is already in the database! "
                    "Showing the existing entry:"
                )
                row = existing.iloc[0]
                st.markdown(f"**Category:** {row['Category']}")
                st.markdown(f"**Description:** {row['Description']}")
                st.markdown(f"**Core Skills:** {row['CoreSkills']}")
            else:
                with st.spinner(f"Generating career profile for **{gen_career}**..."):
                    card = ai_generate_career_card(gen_career.strip(), selected_model)

                if not card:
                    st.error(
                        "Generation failed. The model may have returned invalid JSON. "
                        "Try again or use a different model."
                    )
                else:
                    st.success(f"✅ Career card generated for **{card.get('Career', gen_career)}**")

                    with st.container(border=True):
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown(f"### {card.get('Career', gen_career)}")
                            st.markdown(f"**Category:** {card.get('Category', 'N/A')}")
                            st.markdown(f"**Ideal Audience:** {card.get('IdealAudience', 'N/A')}")
                            st.markdown(f"**Entry Paths:** {card.get('EntryPaths', 'N/A')}")
                        with c2:
                            st.markdown(f"**Core Skills:** {card.get('CoreSkills', 'N/A')}")
                            st.markdown(f"**Tools/Tech:** {card.get('ToolsTech', 'N/A')}")
                            st.markdown(f"**Keywords:** {card.get('InterestKeywords', 'N/A')}")

                        st.markdown("---")
                        st.markdown(f"**Description:** {card.get('Description', 'N/A')}")

                        gtabs = st.tabs(["Roadmap", "12-Week Plan"])
                        with gtabs[0]:
                            roadmap_steps = [s.strip() for s in
                                             card.get("Roadmap", "").split("|") if s.strip()]
                            for i, step in enumerate(roadmap_steps, 1):
                                st.markdown(f"**{i}.** {step}")
                        with gtabs[1]:
                            week_steps = [s.strip() for s in
                                          card.get("12WeekPlan", "").split("|") if s.strip()]
                            for i, wk in enumerate(week_steps, 1):
                                with st.expander(f"Week {i}"):
                                    st.write(wk)

                    # Option to add to session dataset
                    if st.button("➕ Add this career to my session dataset"):
                        new_row = pd.DataFrame([card])
                        # Align columns
                        for col in df.columns:
                            if col not in new_row.columns:
                                new_row[col] = ""
                        new_row["_combined_text"] = (
                            str(card.get("InterestKeywords", "")) + " " +
                            str(card.get("Description", "")) + " " +
                            str(card.get("CoreSkills", "")) + " " +
                            str(card.get("ToolsTech", ""))
                        )
                        new_row["RoadmapList"] = [
                            [s.strip() for s in card.get("Roadmap", "").split("|") if s.strip()]
                        ]
                        new_row["WeekPlanList"] = [
                            [s.strip() for s in card.get("12WeekPlan", "").split("|") if s.strip()]
                        ]
                        df = pd.concat([df, new_row], ignore_index=True)
                        st.success("Added to session! It will appear in search results.")

                    # Download
                    st.download_button(
                        "⬇️ Download as CSV",
                        data=pd.DataFrame([card]).to_csv(index=False).encode("utf-8"),
                        file_name=f"{gen_career.replace(' ', '_')}_career_card.csv",
                        mime="text/csv",
                    )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — PERSONALISE PLAN (AI-powered upgrade of original)
# ═════════════════════════════════════════════════════════════════════════════
with tab_plan:
    st.header("🗓️ Personalise a 12-Week Plan")

    col1, col2 = st.columns([0.4, 0.6])
    with col1:
        base_career = st.selectbox("Pick a base career", df["Career"].unique(), key="plan_career")
        plan_audience = st.text_input("Who is this for?", value="student",
                                      placeholder="student / career switcher / professional")
        goal = st.text_area("What is your goal for the next 12 weeks?", key="plan_goal")
        add_focus = st.text_input("Any special focus?",
                                  placeholder="e.g. 'SQL and dashboards', 'cloud basics'")

        use_ai_plan = st.toggle("✨ Use AI to personalise (requires Ollama)", value=AI_ON,
                                disabled=not AI_ON)

    with col2:
        if st.button("Generate Personalised Plan", type="primary"):
            base_row = df[df["Career"] == base_career].iloc[0]
            base_weeks = base_row["WeekPlanList"]

            if use_ai_plan and AI_ON:
                with st.spinner("AI is crafting your personalised plan..."):
                    ai_plan = ai_generate_plan(
                        base_career, goal, add_focus,
                        plan_audience, base_weeks, selected_model
                    )

                if ai_plan is None:
                    st.error("Ollama is not running. Start it with `ollama serve`.")
                else:
                    st.success("✅ AI-personalised plan generated!")
                    st.markdown(ai_plan)

                    st.download_button(
                        "⬇️ Download Plan (Markdown)",
                        data=f"# AI-Personalised 12-Week Plan: {base_career}\n\n{ai_plan}".encode("utf-8"),
                        file_name="ai_personalised_12week_plan.md",
                        mime="text/markdown",
                    )
            else:
                # Original rule-based generation
                plan_out = []
                for i, wk in enumerate(base_weeks, start=1):
                    enhanced = wk
                    if goal:
                        enhanced += f" — (Aligned to goal: {goal})"
                    if add_focus and any(
                        k in wk.lower()
                        for k in ["project", "build", "practice", "learn", "deploy", "interview", "portfolio"]
                    ):
                        enhanced += f" — (Focus: {add_focus})"
                    plan_out.append(f"Week {i}: {enhanced}")

                st.success("Plan generated!")
                for line in plan_out:
                    st.write(line)

                md = "# 12-Week Personalised Plan\n\n" + os.linesep.join([f"- {l}" for l in plan_out])
                st.download_button(
                    "⬇️ Download Plan (Markdown)",
                    data=md.encode("utf-8"),
                    file_name="personalised_12week_plan.md",
                    mime="text/markdown",
                )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — BROWSE (original, preserved)
# ═════════════════════════════════════════════════════════════════════════════
with tab_browse:
    st.header("📚 Browse Dataset")

    bcol1, bcol2 = st.columns([0.6, 0.4])
    with bcol1:
        st.dataframe(
            df[["Career", "Category", "CoreSkills", "ToolsTech", "InterestKeywords"]]
            .sort_values(["Category", "Career"]),
            use_container_width=True,
        )
    with bcol2:
        st.write("**Tip:** Use the search box at the top-right of the table to find specific keywords.")
        st.metric("Total careers in database", len(df))

        if use_rag:
            st.info(
                "**How RAG works:**\n\n"
                "1. Query → 384-dim vector via `all-MiniLM-L6-v2`\n"
                "2. ChromaDB finds nearest career vectors (cosine similarity)\n"
                "3. Results ranked by semantic closeness, not keyword overlap"
            )
        if AI_ON:
            st.info(
                f"**AI powered by Ollama ({selected_model})**\n\n"
                "Running 100% locally — no internet required, no API cost, "
                "no data sent to the cloud."
            )

st.caption("Career Roadmap Advisor v3 — TF-IDF + RAG + Local AI (Ollama)")
