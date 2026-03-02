"""
ERB Editorial Tools — Streamlit App
=====================================
Reviewer finder + topic exploration tools for the editorial team.

Usage:
    streamlit run app.py

Requirements:
    pip install streamlit pandas openpyxl scikit-learn requests plotly
"""

import streamlit as st
import pandas as pd
import numpy as np
import math
import re
import time
import requests
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ERB Editorial Tools",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ──────────────────────────────────────────────────────────────
BOARD_FILE = "ERB_ETP_enriched.xlsx"  # Change to your enriched file path
SCOPUS_FILE = "scopus_reviewer_database.xlsx"
S2_BASE = "https://api.semanticscholar.org/graph/v1"
JOURNAL_COLS = [
    "Pubs_ERD", "Pubs_ETP", "Pubs_FBR", "Pubs_IJEBR", "Pubs_ISBJ",
    "Pubs_JBV", "Pubs_JBVI", "Pubs_JSBM", "Pubs_SBE", "Pubs_SEJ", "Pubs_VC",
]
JOURNAL_ABBREVS = [c.replace("Pubs_", "") for c in JOURNAL_COLS]

# ── Data loading (cached) ─────────────────────────────────────────────────
@st.cache_data
def load_board(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    profiles = []
    for _, row in df.iterrows():
        parts = []
        for col, weight in [("Areas_of_Expertise", 2), ("Keywords", 2),
                            ("S2_Extracted_Keywords", 2), ("S2_Recent_Titles", 1)]:
            val = str(row.get(col, ""))
            if val and val != "nan":
                parts.extend([val] * weight)
        profiles.append(" ".join(parts).lower())
    df["_profile"] = profiles
    return df


@st.cache_resource
def build_index(profiles: tuple):
    vectorizer = TfidfVectorizer(
        max_features=5000, ngram_range=(1, 2),
        stop_words="english", min_df=1, sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(list(profiles))
    return vectorizer, matrix


@st.cache_data
def load_scopus(path: str, erb_names: set) -> pd.DataFrame:
    df = pd.read_excel(path)
    profiles = []
    for _, row in df.iterrows():
        parts = []
        for col, weight in [("Areas_of_Expertise", 2), ("Keywords", 2),
                            ("Recent_Titles", 1)]:
            val = str(row.get(col, ""))
            if val and val != "nan":
                parts.extend([val] * weight)
        profiles.append(" ".join(parts).lower())
    df["_profile"] = profiles
    df["Is_ERB"] = df["Name"].str.lower().str.strip().isin(erb_names)
    return df


def find_reviewers(query: str, df, vectorizer, matrix, top_n=10, exclude=None):
    scores = cosine_similarity(
        vectorizer.transform([query.lower()]), matrix
    ).flatten()
    result = df.copy()
    result["Relevance"] = scores
    if exclude:
        exc = [n.lower().strip() for n in exclude]
        result = result[~result["Name"].str.lower().str.strip().isin(exc)]
    result = result.nlargest(top_n, "Relevance")
    result = result[result["Relevance"] > 0.01]
    return result


# ── Semantic Scholar helpers ───────────────────────────────────────────────
S2_MAX_RETRIES = 3
S2_RETRY_BACKOFF = 10  # seconds; doubles each retry

def s2_search_papers(query: str, year_from: int = None, year_to: int = None,
                     limit: int = 100, api_key: str = None) -> list:
    headers = {"x-api-key": api_key} if api_key else {}
    params = {
        "query": query,
        "limit": min(limit, 100),
        "fields": "title,abstract,year,citationCount,authors,journal,fieldsOfStudy",
        "fieldsOfStudy": "Business",
    }
    if year_from or year_to:
        params["year"] = f"{year_from or ''}-{year_to or ''}"
    backoff = S2_RETRY_BACKOFF
    for attempt in range(S2_MAX_RETRIES + 1):
        try:
            r = requests.get(f"{S2_BASE}/paper/search", params=params,
                             headers=headers, timeout=15)
            if r.status_code == 429:
                if attempt < S2_MAX_RETRIES:
                    st.warning(f"Rate limited, retrying in {backoff}s...")
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                st.error("S2 API rate limited — max retries exceeded. Try again later.")
                return None
            if r.status_code >= 500:
                if attempt < S2_MAX_RETRIES:
                    st.warning(f"Server error ({r.status_code}), retrying in {backoff}s...")
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                st.error(f"S2 API server error ({r.status_code}) — max retries exceeded.")
                return None
            r.raise_for_status()
            return r.json().get("data", [])
        except Exception as e:
            st.error(f"S2 API error: {e}")
            return None
    return None


def extract_topic_keywords(papers: list, top_n: int = 30) -> list:
    stop = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
        "been", "have", "has", "had", "this", "that", "it", "its", "we", "our",
        "not", "no", "so", "if", "than", "how", "what", "which", "who",
        "study", "research", "paper", "article", "analysis", "results",
        "findings", "evidence", "approach", "effect", "effects", "role",
        "case", "using", "based", "new", "can", "may", "does", "also",
        "between", "more", "most", "some", "such", "through", "do",
        "examining", "exploring", "understanding", "investigate", "review",
    }
    all_text = []
    for p in papers:
        title = (p.get("title") or "").lower()
        abstract = (p.get("abstract") or "").lower()
        all_text.append(title + " " + title + " " + abstract)

    combined = " ".join(all_text)
    words = re.findall(r'[a-z][a-z-]+[a-z]', combined)
    words = [w for w in words if w not in stop and len(w) > 2]

    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
    bigram_counts = Counter(bg for bg in bigrams
                            if bg.split()[0] not in stop and bg.split()[1] not in stop)
    unigram_counts = Counter(words)

    keywords = []
    for bg, c in bigram_counts.most_common(top_n):
        if c >= 2:
            keywords.append((bg, c))
    for ug, c in unigram_counts.most_common(top_n * 2):
        if ug not in " ".join([k[0] for k in keywords]) and c >= 3:
            keywords.append((ug, c))
        if len(keywords) >= top_n:
            break
    return keywords


# ── Load data ──────────────────────────────────────────────────────────────
try:
    df = load_board(BOARD_FILE)
    vectorizer, tfidf_matrix = build_index(tuple(df["_profile"].tolist()))
    board_loaded = True
except FileNotFoundError:
    board_loaded = False

try:
    erb_names_lower = set(df["Name"].str.lower().str.strip()) if board_loaded else set()
    scopus_df = load_scopus(SCOPUS_FILE, frozenset(erb_names_lower))
    scopus_vec, scopus_matrix = build_index(tuple(scopus_df["_profile"].tolist()))
    scopus_loaded = True
except FileNotFoundError:
    scopus_loaded = False

# ── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.title("📚 ERB Editorial Tools")
tool = st.sidebar.radio(
    "Select tool:",
    ["🔍 Reviewer Finder", "🌐 Topic Explorer", "📊 Board Overview"],
)

try:
    _secrets_key = st.secrets.get("S2_API_KEY", "")
except Exception:
    _secrets_key = ""
if _secrets_key:
    api_key = _secrets_key
    st.sidebar.success("S2 API key loaded from secrets")
else:
    api_key = st.sidebar.text_input("Semantic Scholar API Key",
                                    type="password",
                                    help="Needed for Topic Explorer. Speeds up all API calls.")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**ERB members:** "
    + (f"{len(df)}" if board_loaded else "❌ File not found")
)
st.sidebar.markdown(
    "**Broader pool:** "
    + (f"{len(scopus_df):,} reviewers" if scopus_loaded else "not loaded")
)

# ══════════════════════════════════════════════════════════════════════════
# TOOL 1: REVIEWER FINDER
# ══════════════════════════════════════════════════════════════════════════
if tool == "🔍 Reviewer Finder":
    st.title("🔍 Reviewer Finder")
    st.markdown("Paste a paper's abstract and keywords to find the best-matched reviewers.")

    if not board_loaded:
        st.error(f"Board file `{BOARD_FILE}` not found. Place it in the same directory as this app.")
        st.stop()

    col1, col2 = st.columns([2, 1])

    with col1:
        abstract = st.text_area("Paper abstract", height=200,
                                placeholder="Paste the abstract here...")
        keywords = st.text_input("Paper keywords (comma-separated)",
                                 placeholder="e.g. crowdfunding, entrepreneurial finance, signaling")

    with col2:
        top_n = st.slider("Number of suggestions", 5, 25, 10)
        exclude_text = st.text_area("Exclude authors (one per line)",
                                    height=100,
                                    placeholder="e.g. author names to exclude\n(paper authors, conflicted reviewers)")
        exclude_names = [n.strip() for n in exclude_text.strip().split("\n") if n.strip()] if exclude_text else None

        if scopus_loaded:
            st.caption("Broader pool filters")
            selected_journals = st.multiselect(
                "Require pubs in journals",
                options=JOURNAL_ABBREVS,
                default=[],
                help="Only show broader-pool reviewers with ≥ min pubs in each selected journal",
            )
            min_pubs = st.slider("Minimum publications", 1, 20, 1, key="min_pubs")

    if st.button("🔎 Find Reviewers", type="primary", use_container_width=True):
        if not abstract.strip():
            st.warning("Please paste an abstract.")
        else:
            query = f"{abstract} {keywords} {keywords} {keywords}"

            # ── Section A: ERB Board Members ──────────────────────────────
            erb_results = find_reviewers(query, df, vectorizer, tfidf_matrix,
                                         top_n=top_n, exclude=exclude_names)

            st.markdown("### ERB Board Members")
            if erb_results.empty:
                st.info("No strong ERB matches found.")
            else:
                st.caption(f"Top {len(erb_results)} matches from the editorial board")
                for rank, (_, row) in enumerate(erb_results.iterrows(), 1):
                    score = row["Relevance"]
                    with st.container():
                        c1, c2 = st.columns([3, 1])
                        with c1:
                            st.markdown(f"**{rank}. {row['Name']}**")
                            st.caption(row["Location"])
                            expertise = str(row.get("Areas_of_Expertise", ""))
                            if expertise and expertise != "nan":
                                st.markdown(f"*{expertise}*")
                        with c2:
                            st.metric("Score", f"{score:.3f}")
                            st.progress(min(score * 3, 1.0))
                        st.divider()

            # ── Section B: Broader Reviewer Pool ──────────────────────────
            broader_results = pd.DataFrame()
            if scopus_loaded:
                st.markdown("### Broader Reviewer Pool")
                broader_results = find_reviewers(
                    query, scopus_df, scopus_vec, scopus_matrix,
                    top_n=top_n * 3, exclude=exclude_names,
                )
                if not broader_results.empty:
                    # Exclude names already shown in ERB section
                    erb_shown = set(erb_results["Name"].str.lower().str.strip()) if not erb_results.empty else set()
                    broader_results = broader_results[
                        ~broader_results["Name"].str.lower().str.strip().isin(erb_shown)
                    ]
                    # Apply journal filters
                    for j in (selected_journals if scopus_loaded else []):
                        col_name = f"Pubs_{j}"
                        if col_name in broader_results.columns:
                            broader_results = broader_results[broader_results[col_name] >= min_pubs]
                    # Limit to top_n after filtering
                    broader_results = broader_results.head(top_n)

                if broader_results.empty:
                    st.info("No broader-pool matches (try relaxing journal filters).")
                else:
                    st.caption(f"Top {len(broader_results)} matches from the broader Scopus pool")
                    for rank, (_, row) in enumerate(broader_results.iterrows(), 1):
                        score = row["Relevance"]
                        name = str(row["Name"])
                        location = str(row.get("Location", ""))
                        if location == "nan":
                            location = ""
                        expertise = str(row.get("Areas_of_Expertise", ""))
                        if expertise == "nan":
                            expertise = ""
                        erb_badge = " *(ERB)*" if row.get("Is_ERB", False) else ""

                        # Journal breakdown — only non-zero journals
                        journal_parts = []
                        for jcol in JOURNAL_COLS:
                            val = row.get(jcol, 0)
                            if pd.notna(val) and int(val) > 0:
                                journal_parts.append(f"{jcol.replace('Pubs_', '')}: {int(val)}")
                        journal_str = " | ".join(journal_parts)

                        with st.container():
                            c1, c2 = st.columns([3, 1])
                            with c1:
                                st.markdown(f"**{rank}. {name}**{erb_badge}")
                                if location:
                                    st.caption(f"📍 {location}")
                                if expertise:
                                    st.markdown(f"*{expertise[:200]}*")
                                pubs = row.get("Total_Pubs", 0)
                                cites = row.get("Total_Citations", 0)
                                if pd.notna(pubs) and pd.notna(cites):
                                    st.caption(f"📊 {int(pubs)} pubs · {int(cites)} citations")
                                if journal_str:
                                    st.caption(f"📰 {journal_str}")
                            with c2:
                                st.metric("Score", f"{score:.3f}")
                                st.progress(min(score * 3, 1.0))
                            st.divider()

            # ── Combined CSV download ─────────────────────────────────────
            csv_parts = []
            if not erb_results.empty:
                erb_csv = erb_results[["Name", "Location", "Areas_of_Expertise",
                                       "Keywords", "Relevance"]].copy()
                erb_csv["Source"] = "ERB"
                csv_parts.append(erb_csv)
            if not broader_results.empty:
                cols = ["Name", "Location", "Areas_of_Expertise", "Keywords", "Relevance"]
                extra = ["Total_Pubs", "Total_Citations"] + JOURNAL_COLS
                cols += [c for c in extra if c in broader_results.columns]
                bp_csv = broader_results[cols].copy()
                bp_csv["Source"] = "Broader Pool"
                csv_parts.append(bp_csv)
            if csv_parts:
                combined_csv = pd.concat(csv_parts, ignore_index=True).to_csv(index=False)
                st.download_button("📥 Download all results as CSV", combined_csv,
                                   "reviewer_suggestions.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════
# TOOL 2: TOPIC EXPLORER
# ══════════════════════════════════════════════════════════════════════════
elif tool == "🌐 Topic Explorer":
    st.title("🌐 Topic Explorer")
    st.markdown("Search Semantic Scholar to explore research trends, find key papers, and identify expertise gaps.")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📑 Paper Search", "📈 Trend Analysis", "🌱 Emerging Scholars",
        "📚 Landmark Papers", "🗺️ Topic Map", "🧭 Board Gap Finder",
    ])

    # ── Tab 1: Paper Search ────────────────────────────────────────────────
    with tab1:
        st.markdown("Search for recent papers on a topic and see who's publishing what.")
        search_col1, search_col2 = st.columns([2, 1])

        with search_col1:
            search_query = st.text_input("Search topic",
                                         placeholder="e.g. entrepreneurial ecosystems digital platforms")
        with search_col2:
            year_from = st.number_input("From year", 2015, 2026, 2020)
            year_to = st.number_input("To year", 2015, 2026, 2025)
            max_results = st.slider("Max papers", 10, 100, 50, key="search_max")

        if st.button("🔎 Search Papers", key="search_papers"):
            if not search_query:
                st.warning("Enter a search topic.")
            else:
                with st.spinner("Querying Semantic Scholar..."):
                    papers = s2_search_papers(search_query, year_from, year_to,
                                              max_results, api_key)

                if papers is None:
                    pass  # error already shown by s2_search_papers
                elif not papers:
                    st.info("No papers found. Try different keywords.")
                else:
                    st.success(f"Found {len(papers)} papers")
                    st.session_state["last_papers"] = papers

                    for p in papers[:30]:
                        with st.expander(
                            f"**{p.get('title', 'Untitled')}** "
                            f"({p.get('year', '?')}) — "
                            f"Citations: {p.get('citationCount', 0)}"
                        ):
                            authors = ", ".join(
                                a.get("name", "") for a in (p.get("authors") or [])[:6]
                            )
                            if len(p.get("authors") or []) > 6:
                                authors += " et al."
                            st.caption(authors)

                            journal = p.get("journal") or {}
                            if journal.get("name"):
                                st.caption(f"📰 {journal['name']}")

                            abstract = p.get("abstract")
                            if abstract:
                                st.markdown(abstract[:500] + ("..." if len(abstract) > 500 else ""))

                            if board_loaded and abstract:
                                query = f"{p.get('title', '')} {abstract}"
                                matches = find_reviewers(query, df, vectorizer,
                                                         tfidf_matrix, top_n=3)
                                if not matches.empty:
                                    st.markdown("**Closest board members:**")
                                    for _, m in matches.iterrows():
                                        st.caption(
                                            f"→ {m['Name']} ({m['Location']}) "
                                            f"— score {m['Relevance']:.3f}"
                                        )

    # ── Tab 2: Trend Analysis ──────────────────────────────────────────────
    with tab2:
        st.markdown(
            "Search a broad topic area and see which keywords and themes "
            "dominate recent publications — useful for spotting emerging trends "
            "or planning special issues."
        )

        trend_query = st.text_input("Topic area to analyse",
                                    placeholder="e.g. entrepreneurship sustainability",
                                    key="trend_query")

        trend_col1, trend_col2 = st.columns(2)
        with trend_col1:
            trend_year_from = st.number_input("From year", 2015, 2026, 2020,
                                              key="trend_from")
        with trend_col2:
            trend_year_to = st.number_input("To year", 2015, 2026, 2025,
                                            key="trend_to")

        if st.button("📈 Analyse Trends", key="analyse_trends"):
            if not trend_query:
                st.warning("Enter a topic area.")
            else:
                with st.spinner("Fetching papers and extracting keywords..."):
                    papers = s2_search_papers(trend_query, trend_year_from,
                                              trend_year_to, 100, api_key)

                if papers is None:
                    pass
                elif not papers:
                    st.info("No papers found.")
                else:
                    st.success(f"Analysed {len(papers)} papers")

                    keywords = extract_topic_keywords(papers, top_n=25)

                    if keywords:
                        import plotly.express as px
                        kw_df = pd.DataFrame(keywords, columns=["Keyword", "Frequency"])
                        kw_df = kw_df.sort_values("Frequency", ascending=True)
                        fig = px.bar(kw_df, x="Frequency", y="Keyword",
                                     orientation="h",
                                     title="Top Keywords in Recent Papers",
                                     height=max(400, len(kw_df) * 25))
                        fig.update_layout(yaxis=dict(dtick=1), showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)

                    years = [p.get("year") for p in papers if p.get("year")]
                    if years:
                        import plotly.express as px
                        year_df = pd.DataFrame({"Year": years})
                        year_counts = year_df["Year"].value_counts().sort_index()
                        fig2 = px.bar(x=year_counts.index, y=year_counts.values,
                                      labels={"x": "Year", "y": "Papers"},
                                      title="Publication Volume by Year")
                        st.plotly_chart(fig2, use_container_width=True)

                    st.markdown("### Most cited papers")
                    sorted_papers = sorted(papers,
                                           key=lambda p: p.get("citationCount", 0),
                                           reverse=True)
                    for p in sorted_papers[:10]:
                        st.markdown(
                            f"- **{p.get('title', 'Untitled')}** "
                            f"({p.get('year', '?')}) — "
                            f"{p.get('citationCount', 0)} citations"
                        )

    # ── Tab 3: Emerging Scholars ──────────────────────────────────────────
    with tab3:
        st.markdown(
            "Find rising researchers in a topic area, ranked by a score "
            "combining recent paper count and citation impact."
        )

        es_query = st.text_input(
            "Research topic",
            placeholder="e.g. crowdfunding entrepreneurship",
            key="es_query",
        )
        es_col1, es_col2 = st.columns(2)
        with es_col1:
            es_recency = st.slider("Recency window (years)", 1, 10, 5, key="es_recency")
        with es_col2:
            es_top_n = st.slider("Number of scholars", 5, 50, 20, key="es_topn")

        if st.button("🌱 Find Emerging Scholars", key="find_scholars"):
            if not es_query:
                st.warning("Enter a research topic.")
            else:
                with st.spinner("Searching Semantic Scholar for recent authors..."):
                    from datetime import datetime
                    cutoff = datetime.now().year - es_recency
                    papers = s2_search_papers(es_query, year_from=cutoff,
                                              limit=100, api_key=api_key)

                if papers is None:
                    pass
                elif not papers:
                    st.info("No papers found. Try different keywords.")
                else:
                    from collections import defaultdict
                    author_stats = defaultdict(lambda: {
                        "name": "", "paper_count": 0, "total_cites": 0, "author_id": ""
                    })
                    for p in papers:
                        cites = p.get("citationCount") or 0
                        for a in (p.get("authors") or []):
                            aid = a.get("authorId", "")
                            if not aid:
                                continue
                            s = author_stats[aid]
                            s["name"] = a.get("name", "")
                            s["author_id"] = aid
                            s["paper_count"] += 1
                            s["total_cites"] += cites

                    rows = []
                    for aid, s in author_stats.items():
                        if s["paper_count"] < 2:
                            continue
                        score = s["paper_count"] * math.log(s["total_cites"] + 1)
                        rows.append({
                            "Name": s["name"],
                            "Papers": s["paper_count"],
                            "Citations": s["total_cites"],
                            "Score": round(score, 1),
                            "S2_Profile": f"https://www.semanticscholar.org/author/{aid}",
                        })

                    if not rows:
                        st.info("No scholars found with 2+ papers.")
                    else:
                        es_df = pd.DataFrame(rows).nlargest(es_top_n, "Score").reset_index(drop=True)
                        es_df.index += 1
                        es_df.index.name = "Rank"
                        st.success(f"Found {len(es_df)} emerging scholars")
                        st.dataframe(
                            es_df,
                            column_config={
                                "S2_Profile": st.column_config.LinkColumn("S2 Profile"),
                            },
                            use_container_width=True,
                        )
                        csv = es_df.to_csv(index=False)
                        st.download_button("📥 Download CSV", csv,
                                           "emerging_scholars.csv", "text/csv",
                                           key="dl_scholars")

    # ── Tab 4: Landmark Papers ────────────────────────────────────────────
    with tab4:
        st.markdown(
            "Find the most-cited papers on a topic — useful for understanding "
            "the intellectual foundations of a research area."
        )

        lp_query = st.text_input(
            "Research topic",
            placeholder="e.g. digital entrepreneurship",
            key="lp_query",
        )
        lp_top_n = st.slider("Number of papers", 5, 50, 20, key="lp_topn")

        if st.button("📚 Find Landmark Papers", key="find_landmarks"):
            if not lp_query:
                st.warning("Enter a research topic.")
            else:
                with st.spinner("Fetching most-cited papers..."):
                    headers = {"x-api-key": api_key} if api_key else {}
                    params = {
                        "query": lp_query,
                        "limit": 100,
                        "fields": "title,year,citationCount,authors,journal,externalIds",
                        "sort": "citationCount:desc",
                        "fieldsOfStudy": "Business",
                    }
                    try:
                        r = requests.get(f"{S2_BASE}/paper/search", params=params,
                                         headers=headers, timeout=20)
                        r.raise_for_status()
                        papers = r.json().get("data", [])
                    except Exception as e:
                        st.error(f"S2 API error: {e}")
                        papers = []


                if not papers:
                    st.info("No papers found. Try different keywords.")
                else:
                    lp_rows = []
                    for p in papers[:lp_top_n]:
                        authors = p.get("authors") or []
                        first = authors[0]["name"] if authors else "Unknown"
                        et_al = " et al." if len(authors) > 1 else ""
                        doi = (p.get("externalIds") or {}).get("DOI", "")
                        journal_name = (p.get("journal") or {}).get("name", "")
                        lp_rows.append({
                            "Title": p.get("title", ""),
                            "Authors": f"{first}{et_al}",
                            "Year": p.get("year", ""),
                            "Citations": p.get("citationCount", 0),
                            "Journal": journal_name,
                            "DOI": doi,
                        })

                    st.success(f"Top {len(lp_rows)} most-cited papers")
                    for i, row in enumerate(lp_rows, 1):
                        doi_link = f" — [DOI](https://doi.org/{row['DOI']})" if row["DOI"] else ""
                        with st.expander(
                            f"**{i}. {row['Title']}** ({row['Year']}) — "
                            f"{row['Citations']:,} citations"
                        ):
                            st.markdown(f"**Authors:** {row['Authors']}")
                            if row["Journal"]:
                                st.markdown(f"**Journal:** {row['Journal']}")
                            if row["DOI"]:
                                st.markdown(f"**DOI:** [https://doi.org/{row['DOI']}](https://doi.org/{row['DOI']})")

                    lp_df = pd.DataFrame(lp_rows)
                    csv = lp_df.to_csv(index=False)
                    st.download_button("📥 Download CSV", csv,
                                       "landmark_papers.csv", "text/csv",
                                       key="dl_landmarks")

    # ── Tab 5: Topic Map ──────────────────────────────────────────────────
    with tab5:
        st.markdown(
            "Analyse a research topic to discover sub-topics and related concepts "
            "by extracting keywords from recent papers."
        )

        tm_query = st.text_input(
            "Research topic to map",
            placeholder="e.g. venture capital innovation",
            key="tm_query",
        )

        if st.button("🗺️ Generate Topic Map", key="gen_topicmap"):
            if not tm_query:
                st.warning("Enter a research topic.")
            else:
                with st.spinner("Fetching papers and extracting keywords..."):
                    from datetime import datetime
                    papers = s2_search_papers(tm_query,
                                              year_from=datetime.now().year - 5,
                                              limit=100, api_key=api_key)

                if papers is None:
                    pass
                elif not papers:
                    st.info("No papers found. Try different keywords.")
                else:
                    st.success(f"Analysed {len(papers)} papers")

                    # Extract keywords (same logic as topic_explorer.py)
                    stop = {
                        "the", "a", "an", "and", "or", "but", "in", "on", "at",
                        "to", "for", "of", "with", "by", "from", "as", "is", "was",
                        "are", "were", "this", "that", "we", "our", "their", "its",
                        "how", "what", "which", "who", "not", "no", "do", "does",
                        "did", "will", "would", "can", "could", "study", "paper",
                        "research", "article", "results", "analysis", "findings",
                        "evidence", "based", "using", "new", "approach", "case",
                        "role", "effect", "effects", "impact", "review", "may",
                        "also", "between", "through", "more", "than", "been",
                        "have", "has", "had", "about", "into", "over", "such",
                        "these", "those", "some", "other", "each", "both", "most",
                        "should",
                    }

                    all_text = []
                    for p in papers:
                        title = (p.get("title") or "").lower()
                        abstract = (p.get("abstract") or "").lower()
                        all_text.append(title + " " + title)
                        if abstract:
                            all_text.append(abstract)

                    combined = " ".join(all_text)
                    words = re.findall(r'[a-z][a-z-]+[a-z]', combined)
                    words = [w for w in words if w not in stop and len(w) > 2]

                    bigrams = Counter()
                    for i in range(len(words) - 1):
                        bg = f"{words[i]} {words[i+1]}"
                        if words[i] not in stop and words[i+1] not in stop:
                            bigrams[bg] += 1

                    unigrams = Counter(words)
                    query_words = set(tm_query.lower().split())

                    bg_rows = []
                    for term, count in bigrams.most_common(60):
                        if set(term.split()).issubset(query_words) or count < 3:
                            continue
                        bg_rows.append({"Term": term, "Type": "bigram", "Count": count})

                    ug_rows = []
                    for term, count in unigrams.most_common(100):
                        if term in query_words or count < 5:
                            continue
                        ug_rows.append({"Term": term, "Type": "unigram", "Count": count})

                    import plotly.express as px

                    if bg_rows:
                        bg_df = pd.DataFrame(bg_rows[:25]).sort_values("Count", ascending=True)
                        fig_bg = px.bar(
                            bg_df, x="Count", y="Term", orientation="h",
                            title="Top Sub-topics (bigrams)",
                            height=max(400, len(bg_df) * 25),
                        )
                        fig_bg.update_layout(yaxis=dict(dtick=1), showlegend=False)
                        st.plotly_chart(fig_bg, use_container_width=True)

                    if ug_rows:
                        ug_df = pd.DataFrame(ug_rows[:20]).sort_values("Count", ascending=True)
                        fig_ug = px.bar(
                            ug_df, x="Count", y="Term", orientation="h",
                            title="Top Concepts (unigrams)",
                            height=max(400, len(ug_df) * 25),
                        )
                        fig_ug.update_layout(yaxis=dict(dtick=1), showlegend=False)
                        st.plotly_chart(fig_ug, use_container_width=True)

                    all_rows = bg_rows + ug_rows
                    if all_rows:
                        tm_df = pd.DataFrame(all_rows)
                        csv = tm_df.to_csv(index=False)
                        st.download_button("📥 Download CSV", csv,
                                           "topic_map.csv", "text/csv",
                                           key="dl_topicmap")

    # ── Tab 6: Board Gap Finder ────────────────────────────────────────────
    with tab6:
        st.markdown(
            "Search for a topic and see how well your current editorial board "
            "covers it. Useful for identifying expertise gaps when considering "
            "new board appointments."
        )

        gap_query = st.text_input("Topic to check board coverage for",
                                  placeholder="e.g. artificial intelligence entrepreneurship",
                                  key="gap_query")

        if st.button("🧭 Check Coverage", key="check_coverage"):
            if not gap_query or not board_loaded:
                st.warning("Enter a topic and ensure the board file is loaded.")
            else:
                query = f"{gap_query} {gap_query} {gap_query}"
                scores = cosine_similarity(
                    vectorizer.transform([query.lower()]), tfidf_matrix
                ).flatten()

                result = df.copy()
                result["Relevance"] = scores

                threshold_strong = 0.10
                threshold_moderate = 0.05
                strong = (scores >= threshold_strong).sum()
                moderate = ((scores >= threshold_moderate) & (scores < threshold_strong)).sum()

                col1, col2, col3 = st.columns(3)
                col1.metric("Strong coverage (≥0.10)", strong)
                col2.metric("Moderate (0.05–0.10)", moderate)
                col3.metric("Weak/none (<0.05)", int((scores < threshold_moderate).sum()))

                if strong < 3:
                    st.warning(
                        f"⚠️ Only {strong} board members have strong expertise here. "
                        "Consider recruiting in this area."
                    )
                elif strong >= 5:
                    st.success(f"✅ Good coverage — {strong} board members well-matched.")

                top = result.nlargest(10, "Relevance")
                st.markdown("### Best-matched board members")
                for _, row in top.iterrows():
                    st.markdown(
                        f"- **{row['Name']}** ({row['Location']}) — "
                        f"score {row['Relevance']:.3f}"
                    )

                if api_key:
                    st.markdown("---")
                    st.markdown("### Potential new board members (from Semantic Scholar)")
                    st.caption("Prolific authors on this topic who are NOT currently on the board.")

                    with st.spinner("Searching for active researchers..."):
                        papers = s2_search_papers(gap_query, 2020, 2025, 100, api_key)

                    if papers:
                        author_counts = Counter()
                        author_info = {}
                        board_names_lower = set(df["Name"].str.lower().str.strip())

                        for p in papers:
                            for a in (p.get("authors") or []):
                                name = a.get("name", "")
                                if name and name.lower().strip() not in board_names_lower:
                                    author_counts[name] += 1
                                    author_info[name] = a.get("authorId", "")

                        st.markdown("**Frequently appearing authors** (not on board):")
                        for name, count in author_counts.most_common(15):
                            if count >= 2:
                                s2_id = author_info.get(name, "")
                                link = f" — [S2 profile](https://www.semanticscholar.org/author/{s2_id})" if s2_id else ""
                                st.markdown(f"- **{name}** — {count} papers{link}")


# ══════════════════════════════════════════════════════════════════════════
# TOOL 3: BOARD OVERVIEW
# ══════════════════════════════════════════════════════════════════════════
elif tool == "📊 Board Overview":
    st.title("📊 Board Overview")

    if not board_loaded:
        st.error(f"Board file `{BOARD_FILE}` not found.")
        st.stop()

    st.markdown(f"**{len(df)} editorial board members**")

    search = st.text_input("🔎 Filter by name, location, or expertise")
    display_df = df[["Name", "Location", "Areas_of_Expertise", "Keywords"]].copy()

    if search:
        mask = display_df.apply(
            lambda row: search.lower() in " ".join(str(v) for v in row).lower(),
            axis=1
        )
        display_df = display_df[mask]

    st.dataframe(display_df, use_container_width=True, height=500)

    st.markdown("### Most common expertise areas across the board")
    all_keywords = []
    for kw in df["Keywords"].dropna():
        all_keywords.extend([k.strip().lower() for k in str(kw).split(",")])

    kw_counts = Counter(all_keywords).most_common(30)
    if kw_counts:
        import plotly.express as px
        kw_df = pd.DataFrame(kw_counts, columns=["Keyword", "Count"])
        kw_df = kw_df.sort_values("Count", ascending=True)
        fig = px.bar(kw_df, x="Count", y="Keyword", orientation="h",
                     title="Board expertise concentration",
                     height=max(400, len(kw_df) * 25))
        fig.update_layout(yaxis=dict(dtick=1))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Geographic distribution")
    countries = []
    for loc in df["Location"].dropna():
        parts = str(loc).split(",")
        country = parts[-1].strip() if parts else "Unknown"
        countries.append(country)
    country_counts = Counter(countries).most_common(20)
    if country_counts:
        import plotly.express as px
        c_df = pd.DataFrame(country_counts, columns=["Country", "Count"])
        fig2 = px.pie(c_df, values="Count", names="Country",
                      title="Board members by country")
        st.plotly_chart(fig2, use_container_width=True)
