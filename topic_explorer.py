"""
S2 Topic Explorer — Semantic Scholar Research Intelligence Tool
================================================================
A suite of tools for editors and researchers to explore topics, track trends,
discover emerging scholars, and map the landscape of a research area.

Requirements:
    pip install pandas requests tqdm matplotlib

Usage:
    python topic_explorer.py

    Or import individual functions:
        from topic_explorer import TopicExplorer
        explorer = TopicExplorer(api_key="your_key")
        results = explorer.topic_trend("crowdfunding entrepreneurship", years=10)

Available explorations:
    1. Topic Trend        — publication volume over time for a research topic
    2. Emerging Scholars  — rising researchers in a topic (high recent output, newer to the field)
    3. Landmark Papers    — most-cited papers in a topic, by era
    4. Topic Map          — related sub-topics and how they cluster
    5. Board Gap Analysis — topics published in your journal vs. board expertise gaps
    6. Journal Benchmarking — who else is publishing in your space
"""

import os
import sys
import time
import json
import re
from collections import Counter, defaultdict
from datetime import datetime

import pandas as pd
import requests
from tqdm import tqdm

# ── Configuration ──────────────────────────────────────────────────────────
S2_BASE = "https://api.semanticscholar.org/graph/v1"
HEADERS = {}
RATE_DELAY = 1.0

api_key = os.environ.get("S2_API_KEY")
if api_key:
    HEADERS["x-api-key"] = api_key
else:
    RATE_DELAY = 3.2
    print("⚠  No S2_API_KEY found. Using free tier (slower rate limits).")
    print("   Set the environment variable S2_API_KEY for faster queries.\n")


def _get(url, params=None):
    """Rate-limited GET request to S2 API."""
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=20)
        time.sleep(RATE_DELAY)
        if r.status_code == 429:
            print("  ⏳ Rate limited, waiting 60s...")
            time.sleep(60)
            r = requests.get(url, params=params, headers=HEADERS, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  ⚠ API error: {e}")
        return {}


# ═══════════════════════════════════════════════════════════════════════════
# 1. TOPIC TREND — publication volume over time
# ═══════════════════════════════════════════════════════════════════════════

def topic_trend(query: str, years: int = 10, save_csv: bool = True) -> pd.DataFrame:
    """
    Track how publication volume for a topic has changed over time.
    Shows year-by-year paper counts and top papers per year.
    """
    current_year = datetime.now().year
    start_year = current_year - years

    print(f"\n📈 Topic Trend: '{query}' ({start_year}–{current_year})")
    print("=" * 60)

    all_papers = []
    # S2 paper search with year filtering
    for year in tqdm(range(start_year, current_year + 1), desc="Fetching years"):
        params = {
            "query": query,
            "limit": 100,
            "fields": "title,year,citationCount,journal,authors",
            "year": str(year),
        }
        data = _get(f"{S2_BASE}/paper/search", params)
        papers = data.get("data", [])
        for p in papers:
            p["_year"] = year
        all_papers.extend(papers)

    if not all_papers:
        print("  No papers found.")
        return pd.DataFrame()

    df = pd.DataFrame(all_papers)
    df["citationCount"] = df["citationCount"].fillna(0).astype(int)

    # Year-by-year summary
    yearly = df.groupby("_year").agg(
        paper_count=("title", "count"),
        total_citations=("citationCount", "sum"),
        avg_citations=("citationCount", "mean"),
    ).reset_index().rename(columns={"_year": "year"})

    print("\nYear  | Papers | Total Cites | Avg Cites")
    print("-" * 48)
    for _, row in yearly.iterrows():
        bar = "█" * min(int(row["paper_count"] / 2), 40)
        print(f"{int(row['year'])}  | {int(row['paper_count']):>6} | {int(row['total_citations']):>11} | {row['avg_citations']:>9.1f}  {bar}")

    # Top paper per year
    print(f"\n🏆 Most-cited paper per year:")
    for year in range(start_year, current_year + 1):
        year_papers = df[df["_year"] == year]
        if year_papers.empty:
            continue
        top = year_papers.nlargest(1, "citationCount").iloc[0]
        authors = top.get("authors", [])
        first_author = authors[0]["name"] if authors else "Unknown"
        print(f"  {year}: [{top['citationCount']:>5} cites] {first_author} — {top['title'][:80]}")

    if save_csv:
        path = f"topic_trend_{query.replace(' ', '_')[:30]}.csv"
        yearly.to_csv(path, index=False)
        print(f"\n💾 Saved to {path}")

    return yearly


# ═══════════════════════════════════════════════════════════════════════════
# 2. EMERGING SCHOLARS — find rising researchers in a topic
# ═══════════════════════════════════════════════════════════════════════════

def emerging_scholars(query: str, top_n: int = 20, recency_years: int = 5,
                      save_csv: bool = True) -> pd.DataFrame:
    """
    Find researchers who are publishing actively and gaining traction in a topic.
    Prioritises recent output and citation velocity over lifetime h-index.
    """
    current_year = datetime.now().year
    cutoff = current_year - recency_years

    print(f"\n🌱 Emerging Scholars: '{query}' (active since {cutoff})")
    print("=" * 60)

    # Fetch recent papers on the topic
    params = {
        "query": query,
        "limit": 200,
        "fields": "title,year,citationCount,authors",
        "year": f"{cutoff}-",
    }
    data = _get(f"{S2_BASE}/paper/search", params)
    papers = data.get("data", [])

    if not papers:
        print("  No papers found.")
        return pd.DataFrame()

    # Count author appearances and aggregate citation impact
    author_stats = defaultdict(lambda: {
        "name": "", "paper_count": 0, "total_cites": 0,
        "papers": [], "author_id": ""
    })

    for p in papers:
        cites = p.get("citationCount") or 0
        title = p.get("title", "")
        for a in (p.get("authors") or []):
            aid = a.get("authorId", "")
            if not aid:
                continue
            s = author_stats[aid]
            s["name"] = a.get("name", "")
            s["author_id"] = aid
            s["paper_count"] += 1
            s["total_cites"] += cites
            s["papers"].append(title)

    # Score: papers × log(citations + 1) — rewards prolific + impactful
    import math
    rows = []
    for aid, s in author_stats.items():
        if s["paper_count"] < 2:
            continue
        score = s["paper_count"] * math.log(s["total_cites"] + 1)
        rows.append({
            "Name": s["name"],
            "S2_Author_ID": aid,
            "Recent_Papers": s["paper_count"],
            "Recent_Citations": s["total_cites"],
            "Emergence_Score": round(score, 1),
            "Sample_Titles": " | ".join(s["papers"][:3]),
        })

    df = pd.DataFrame(rows).nlargest(top_n, "Emergence_Score")

    print(f"\n{'Rank':<5} {'Name':<30} {'Papers':<8} {'Cites':<8} {'Score':<8}")
    print("-" * 65)
    for rank, (_, row) in enumerate(df.iterrows(), 1):
        print(f"{rank:<5} {row['Name']:<30} {row['Recent_Papers']:<8} "
              f"{row['Recent_Citations']:<8} {row['Emergence_Score']:<8}")

    if save_csv:
        path = f"emerging_scholars_{query.replace(' ', '_')[:30]}.csv"
        df.to_csv(path, index=False)
        print(f"\n💾 Saved to {path}")

    return df


# ═══════════════════════════════════════════════════════════════════════════
# 3. LANDMARK PAPERS — most influential papers in a topic
# ═══════════════════════════════════════════════════════════════════════════

def landmark_papers(query: str, top_n: int = 20,
                    save_csv: bool = True) -> pd.DataFrame:
    """
    Find the most-cited papers on a topic. Useful for understanding the
    intellectual foundations of a research area.
    """
    print(f"\n📚 Landmark Papers: '{query}'")
    print("=" * 60)

    params = {
        "query": query,
        "limit": 100,
        "fields": "title,year,citationCount,authors,journal,externalIds",
        "sort": "citationCount:desc",
    }
    data = _get(f"{S2_BASE}/paper/search", params)
    papers = data.get("data", [])

    if not papers:
        print("  No papers found.")
        return pd.DataFrame()

    rows = []
    for p in papers[:top_n]:
        authors = p.get("authors") or []
        first = authors[0]["name"] if authors else "Unknown"
        et_al = f" et al." if len(authors) > 1 else ""
        doi = (p.get("externalIds") or {}).get("DOI", "")
        journal_name = ""
        if p.get("journal"):
            journal_name = p["journal"].get("name", "")

        rows.append({
            "Title": p.get("title", ""),
            "Authors": f"{first}{et_al}",
            "Year": p.get("year", ""),
            "Citations": p.get("citationCount", 0),
            "Journal": journal_name,
            "DOI": doi,
        })

    df = pd.DataFrame(rows)

    for rank, (_, row) in enumerate(df.iterrows(), 1):
        print(f"\n  {rank}. [{row['Citations']:,} cites] {row['Authors']} ({row['Year']})")
        print(f"     {row['Title'][:90]}")
        if row["Journal"]:
            print(f"     {row['Journal']}")

    if save_csv:
        path = f"landmarks_{query.replace(' ', '_')[:30]}.csv"
        df.to_csv(path, index=False)
        print(f"\n💾 Saved to {path}")

    return df


# ═══════════════════════════════════════════════════════════════════════════
# 4. TOPIC MAP — discover sub-topics and related areas
# ═══════════════════════════════════════════════════════════════════════════

def topic_map(query: str, save_csv: bool = True) -> pd.DataFrame:
    """
    Analyse a research topic to discover sub-topics, related concepts,
    and emerging themes. Works by extracting and clustering keywords
    from recent papers on the topic.
    """
    print(f"\n🗺️  Topic Map: '{query}'")
    print("=" * 60)

    # Fetch a large sample of recent papers
    params = {
        "query": query,
        "limit": 200,
        "fields": "title,abstract,year",
        "year": f"{datetime.now().year - 5}-",
    }
    data = _get(f"{S2_BASE}/paper/search", params)
    papers = data.get("data", [])

    if not papers:
        print("  No papers found.")
        return pd.DataFrame()

    # Extract bigrams from titles and abstracts
    stop = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "was", "are", "were", "this",
        "that", "we", "our", "their", "its", "how", "what", "which", "who",
        "not", "no", "do", "does", "did", "will", "would", "can", "could",
        "study", "paper", "research", "article", "results", "analysis",
        "findings", "evidence", "based", "using", "new", "approach",
        "case", "role", "effect", "effects", "impact", "review",
        "may", "also", "between", "through", "more", "than", "been",
        "have", "has", "had", "about", "into", "over", "such", "these",
        "those", "some", "other", "each", "both", "most", "should",
    }

    all_text = []
    for p in papers:
        title = (p.get("title") or "").lower()
        abstract = (p.get("abstract") or "").lower()
        all_text.append(title + " " + title)  # weight titles
        if abstract:
            all_text.append(abstract)

    combined = " ".join(all_text)
    words = re.findall(r'[a-z][a-z-]+[a-z]', combined)
    words = [w for w in words if w not in stop and len(w) > 2]

    # Bigrams
    bigrams = Counter()
    for i in range(len(words) - 1):
        bg = f"{words[i]} {words[i+1]}"
        if words[i] not in stop and words[i+1] not in stop:
            bigrams[bg] += 1

    # Unigrams
    unigrams = Counter(words)

    # Filter out query terms themselves
    query_words = set(query.lower().split())

    print("\n📌 Top sub-topics / themes (bigrams):")
    print("-" * 50)
    rows = []
    rank = 0
    for term, count in bigrams.most_common(60):
        term_words = set(term.split())
        if term_words.issubset(query_words):
            continue
        if count < 3:
            continue
        rank += 1
        if rank <= 25:
            bar = "█" * min(count // 2, 30)
            print(f"  {rank:>2}. {term:<35} ({count:>3}) {bar}")
        rows.append({"term": term, "type": "bigram", "count": count})

    print("\n📌 Top concepts (unigrams, excluding query terms):")
    print("-" * 50)
    rank = 0
    for term, count in unigrams.most_common(100):
        if term in query_words:
            continue
        if count < 5:
            continue
        rank += 1
        if rank <= 20:
            bar = "█" * min(count // 3, 30)
            print(f"  {rank:>2}. {term:<25} ({count:>4}) {bar}")
        rows.append({"term": term, "type": "unigram", "count": count})

    df = pd.DataFrame(rows)
    if save_csv and not df.empty:
        path = f"topic_map_{query.replace(' ', '_')[:30]}.csv"
        df.to_csv(path, index=False)
        print(f"\n💾 Saved to {path}")

    return df


# ═══════════════════════════════════════════════════════════════════════════
# 5. BOARD GAP ANALYSIS — find expertise gaps on your editorial board
# ═══════════════════════════════════════════════════════════════════════════

def board_gap_analysis(board_file: str, journal_topics: list[str],
                       save_csv: bool = True) -> pd.DataFrame:
    """
    Compare the topics being published in your journal's domain against
    the expertise of your editorial board. Highlights topics where you
    may be under-covered.
    """
    print(f"\n🔍 Board Gap Analysis")
    print("=" * 60)

    # Load board
    df = pd.read_excel(board_file)
    board_text = " ".join(
        str(row.get("Areas_of_Expertise", "")) + " " + str(row.get("Keywords", ""))
        for _, row in df.iterrows()
    ).lower()

    print(f"  Loaded {len(df)} board members")
    print(f"  Checking {len(journal_topics)} topic areas\n")

    results = []
    for topic in tqdm(journal_topics, desc="Checking topics"):
        # Count how many board members cover this topic
        topic_lower = topic.lower()
        topic_words = set(topic_lower.split())
        board_coverage = sum(
            1 for _, row in df.iterrows()
            if any(w in str(row.get("Keywords", "")).lower()
                   for w in topic_words)
        )

        # Check recent publication volume via S2
        params = {"query": topic, "limit": 1, "fields": "title",
                  "year": f"{datetime.now().year - 3}-"}
        data = _get(f"{S2_BASE}/paper/search", params)
        pub_volume = data.get("total", 0)

        results.append({
            "Topic": topic,
            "Board_Members_Covering": board_coverage,
            "Recent_S2_Papers": pub_volume,
            "Gap_Flag": "⚠ GAP" if board_coverage < 2 and pub_volume > 50 else "",
        })

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values("Board_Members_Covering")

    print(f"\n{'Topic':<40} {'Board':<8} {'S2 Papers':<12} {'Status'}")
    print("-" * 70)
    for _, row in result_df.iterrows():
        print(f"{row['Topic']:<40} {row['Board_Members_Covering']:<8} "
              f"{row['Recent_S2_Papers']:<12} {row['Gap_Flag']}")

    if save_csv:
        path = "board_gap_analysis.csv"
        result_df.to_csv(path, index=False)
        print(f"\n💾 Saved to {path}")

    return result_df


# ═══════════════════════════════════════════════════════════════════════════
# INTERACTIVE MENU
# ═══════════════════════════════════════════════════════════════════════════

def interactive():
    print("\n" + "=" * 60)
    print("  S2 TOPIC EXPLORER")
    print("  Semantic Scholar Research Intelligence")
    print("=" * 60)

    while True:
        print("\n  1. Topic Trend         — publication volume over time")
        print("  2. Emerging Scholars   — rising researchers in a topic")
        print("  3. Landmark Papers     — most-cited foundational works")
        print("  4. Topic Map           — sub-topics and related concepts")
        print("  5. Board Gap Analysis  — expertise gaps on your board")
        print("  q. Quit\n")

        choice = input("Choose (1-5, q): ").strip()

        if choice == "q":
            break
        elif choice == "1":
            q = input("Topic to track: ").strip()
            y = input("How many years back? (default 10): ").strip()
            topic_trend(q, years=int(y) if y.isdigit() else 10)
        elif choice == "2":
            q = input("Topic to search: ").strip()
            emerging_scholars(q)
        elif choice == "3":
            q = input("Topic to search: ").strip()
            landmark_papers(q)
        elif choice == "4":
            q = input("Topic to map: ").strip()
            topic_map(q)
        elif choice == "5":
            bf = input(f"Board file (default: ERB_ETP_enriched.xlsx): ").strip()
            bf = bf or "ERB_ETP_enriched.xlsx"
            print("Enter journal topics to check (one per line, empty line to finish):")
            topics = []
            while True:
                t = input("  > ").strip()
                if not t:
                    break
                topics.append(t)
            if topics:
                board_gap_analysis(bf, topics)
        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    interactive()
