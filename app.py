import os
import pandas as pd
import streamlit as st
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Streamlit page config
st.set_page_config(
    page_title="Career Roadmap & Advisor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------
# Load dataset function
# ------------------------
@st.cache_data(show_spinner=False)
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Ensure expected columns exist
    expected = {
        "Career", "Category", "InterestKeywords", "Description", "IdealAudience",
        "EntryPaths", "CoreSkills", "ToolsTech", "Roadmap", "12WeekPlan"
    }
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    # Normalize text fields and create a combined text for TF-IDF
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

    # Pre-split Roadmap and 12WeekPlan into lists for display
    df["RoadmapList"] = df["Roadmap"].fillna("").apply(lambda s: [x.strip() for x in str(s).split("|") if x.strip()])
    df["WeekPlanList"] = df["12WeekPlan"].fillna("").apply(lambda s: [x.strip() for x in str(s).split("|") if x.strip()])

    return df


# ------------------------
# TF-IDF Vectorizer
# ------------------------
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


# ------------------------
# Sidebar: Dataset Loading
# ------------------------
st.sidebar.title("🎯 Career Roadmap & Advisor")
st.sidebar.caption("Powered by TF-IDF similarity")

with st.sidebar.expander("📁 Load Dataset", expanded=True):
    use_upload = st.toggle("Upload a CSV instead of default", value=False)
    data_path = "career_interest.csv"
    if use_upload:
        uploaded = st.file_uploader("Upload CSV with the expected columns", type=["csv"])
        if uploaded is not None:
            data_path = uploaded

    try:
        df = load_data(data_path)
        st.success(f"Loaded {len(df)} careers from dataset.")
    except Exception as e:
        st.error("Could not load dataset. Please upload the correct CSV.")
        st.exception(e)
        st.stop()


# ------------------------
# Sidebar Filters
# ------------------------
categories = ["(All)"] + sorted(df["Category"].dropna().unique().tolist())
sel_category = st.sidebar.selectbox("Filter by Category", categories, index=0)

k_weight = st.sidebar.slider("Weight: Interest Keywords", 0.0, 2.0, 1.0, 0.1)
d_weight = st.sidebar.slider("Weight: Description/Skills/Tools", 0.0, 2.0, 1.0, 0.1)

k_topn = st.sidebar.slider("How many recommendations?", 5, 50, 10, 1)

st.sidebar.markdown("---")
with st.sidebar.expander("⚙️ Advanced"):
    ngram_min, ngram_max = st.select_slider(
        "TF-IDF n-gram range",
        options=[1, 2, 3],
        value=(1, 2)
    )


# ------------------------
# Build TF-IDF matrices
# ------------------------
keywords_corpus = df["InterestKeywords"].fillna("").astype(str).tolist()
combined_corpus = df["_combined_text"].fillna("").astype(str).tolist()

kvect, KX = build_vectorizer(keywords_corpus, ngram=(ngram_min, ngram_max))
cvect, CX = build_vectorizer(combined_corpus, ngram=(ngram_min, ngram_max))


# ------------------------
# Main UI
# ------------------------
st.title("🎓 Career Roadmap & Advisor")
st.write("Enter your interests or background. We'll match you to careers and show roadmaps & 12-week plans.")

query = st.text_area(
    "Describe your interests, skills, or goals (e.g., 'coding, APIs, analytics, healthcare AI, product strategy')",
    height=100,
)

audience = st.text_input("Who is the plan for? (e.g., student, career switcher, early professional)", value="student")

colA, colB = st.columns([1, 1])
with colA:
    st.subheader("🔎 Recommendations")

# ------------------------
# Similarity Calculation
# ------------------------
def get_scores(user_query: str):
    if not user_query.strip():
        return pd.Series([0.0] * len(df))
    q_kw = kvect.transform([user_query])
    q_cb = cvect.transform([user_query])
    s_kw = cosine_similarity(q_kw, KX).ravel()
    s_cb = cosine_similarity(q_cb, CX).ravel()
    s = k_weight * s_kw + d_weight * s_cb
    return pd.Series(s)


scores = get_scores(query)
res = df.copy()
res["Similarity"] = scores

if sel_category != "(All)":
    res = res[res["Category"] == sel_category]

res = res.sort_values("Similarity", ascending=False).head(k_topn)


# ------------------------
# Display Recommendations
# ------------------------
if len(res) == 0:
    st.info("Enter a query to see recommendations, or relax filters.")
else:
    for _, row in res.iterrows():
        sim_pct = f"{row['Similarity']*100:0.1f}%"
        with st.container(border=True):
            top = st.columns([0.7, 0.3])
            with top[0]:
                st.markdown(f"### **{row['Career']}**  ")
                st.markdown(f"**Category:** {row['Category']}  ")
                st.markdown(f"**Similarity:** {sim_pct}")
                st.markdown(f"**Ideal Audience:** {row['IdealAudience']}")
                st.markdown(f"**Entry Paths:** {row['EntryPaths']}")
            with top[1]:
                st.progress(min(max(row['Similarity'], 0), 1.0))

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
        "Career", "Category", "Similarity", "Description", "CoreSkills", "ToolsTech",
        "InterestKeywords", "Roadmap", "12WeekPlan", "IdealAudience", "EntryPaths"
    ]
    export_df = res[export_cols].copy()
    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download these recommendations (CSV)",
        data=csv_bytes,
        file_name="career_recommendations.csv",
        mime="text/csv",
    )


# ------------------------
# Personalized 12-Week Plan
# ------------------------
st.markdown("\n---\n")
st.header("🧩 Personalize a 12-Week Plan")

col1, col2 = st.columns([0.4, 0.6])
with col1:
    base_career = st.selectbox("Pick a base career to personalize", df["Career"].unique())
    goal = st.text_area("What is your goal for the next 12 weeks? (e.g., 'land a junior data analyst role')")
    add_focus = st.text_input("Any special focus? (e.g., 'SQL and dashboards', 'cloud basics')")

with col2:
    if st.button("Generate Personalized Plan", type="primary"):
        base = df[df["Career"] == base_career].iloc[0]
        base_weeks = base["WeekPlanList"][:]
        plan_out = []
        for i, wk in enumerate(base_weeks, start=1):
            enhanced = wk
            if goal:
                enhanced += f" — (Aligned to goal: {goal})"
            if add_focus and any(k in wk.lower() for k in ["project", "build", "practice", "learn", "week", "deploy", "interview", "portfolio"]):
                enhanced += f" — (Focus: {add_focus})"
            plan_out.append(f"Week {i}: {enhanced}")

        st.success("Personalized plan generated!")
        for line in plan_out:
            st.write(line)

        md = "# 12-Week Personalized Plan\n\n" + os.linesep.join([f"- {l}" for l in plan_out])
        st.download_button(
            "⬇️ Download Plan (Markdown)",
            data=md.encode("utf-8"),
            file_name="personalized_12week_plan.md",
            mime="text/markdown",
        )


# ------------------------
# Browse Dataset
# ------------------------
st.markdown("\n---\n")
st.header("📚 Browse Dataset")

bcol1, bcol2 = st.columns([0.6, 0.4])
with bcol1:
    st.dataframe(
        df[["Career", "Category", "CoreSkills", "ToolsTech", "InterestKeywords"]]
        .sort_values(["Category", "Career"]), use_container_width=True
    )
with bcol2:
    st.write("**Tip:** Use the search box at the top-right of the table to find specific keywords.")

st.caption("© Your Project — Streamlit Career Advisor | TF-IDF similarity over interests, description, skills, and tools.")
