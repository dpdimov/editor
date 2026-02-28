"""
ERB Editorial Board — Semantic Scholar Enrichment Tool
======================================================
Searches Semantic Scholar for each board member, retrieves their recent
publications, and extracts actual research keywords to supplement/replace
the Google Scholar profile keywords.

Requirements:
    pip install pandas openpyxl requests tqdm

Usage:
    python enrich_board.py ERB_ETP_final.xlsx

Output:
    ERB_ETP_enriched.xlsx  — original data plus new columns:
        - S2_Author_ID       : Semantic Scholar author ID (for future lookups)
        - S2_Paper_Count      : number of papers found
        - S2_Recent_Titles    : titles of up to 20 most recent papers
        - S2_Extracted_Keywords: keywords extracted from paper titles/abstracts
        - S2_Match_Confidence : how confident the name+affiliation match is

Notes:
    • Semantic Scholar's free tier allows ~100 requests / 5 min (no key needed).
      The script includes automatic rate-limiting.
    • If you have an S2 API key, set the environment variable S2_API_KEY.
    • Some names may return no results or ambiguous matches — the script
      flags these so you can review manually.
"""

import os
import sys
import time
import re
import json
from collections import Counter
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# ── Configuration ──────────────────────────────────────────────────────────
S2_BASE = "https://api.semanticscholar.org/graph/v1"
RATE_LIMIT_DELAY = 3.2          # seconds between requests (safe for free tier)
MAX_PAPERS = 20                  # papers to retrieve per author
RECENT_YEARS = 10                # only consider papers from last N years

HEADERS = {}
api_key = os.environ.get("S2_API_KEY")
if api_key:
    HEADERS["x-api-key"] = api_key
    RATE_LIMIT_DELAY = 1.0       # faster with a key

# ── Stop words for keyword extraction ──────────────────────────────────────
STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "this",
    "that", "these", "those", "it", "its", "we", "our", "their", "they",
    "how", "what", "which", "who", "whom", "when", "where", "why",
    "not", "no", "nor", "so", "if", "then", "than", "too", "very",
    "just", "about", "above", "after", "again", "all", "also", "am",
    "any", "because", "before", "between", "both", "each", "few",
    "further", "here", "into", "more", "most", "other", "out", "over",
    "same", "some", "such", "through", "under", "until", "up", "while",
    "study", "research", "paper", "article", "analysis", "results",
    "findings", "evidence", "approach", "effect", "effects", "role",
    "case", "using", "based", "new", "among", "across", "towards",
    "toward", "within", "upon", "during", "however", "whether",
    "examining", "exploring", "understanding", "investigate",
    "investigates", "investigated", "examine", "examines", "examined",
    "explore", "explores", "explored", "impact", "impacts", "implications",
    "review", "literature", "perspective", "perspectives", "context",
    "does", "make", "one", "two", "three", "first", "second",
}


def search_author(name: str, affiliation: str) -> list[dict]:
    """Search S2 for an author by name, return candidate matches."""
    params = {"query": name, "limit": 5,
              "fields": "name,affiliations,paperCount,hIndex"}
    try:
        r = requests.get(f"{S2_BASE}/author/search", params=params,
                         headers=HEADERS, timeout=15)
        r.raise_for_status()
        return r.json().get("data", [])
    except Exception as e:
        print(f"  ⚠ Search failed for {name}: {e}")
        return []


def pick_best_match(candidates: list[dict], name: str, affiliation: str) -> tuple:
    """Pick the best author match based on name similarity and affiliation overlap."""
    if not candidates:
        return None, "no_results"

    aff_lower = affiliation.lower() if affiliation else ""
    name_lower = name.lower()

    scored = []
    for c in candidates:
        score = 0
        c_name = (c.get("name") or "").lower()
        c_affs = " ".join(c.get("affiliations") or []).lower()

        # Name matching
        if c_name == name_lower:
            score += 10
        elif name_lower.split()[-1] in c_name:
            score += 5

        # Affiliation matching — check for university name overlap
        aff_words = set(re.findall(r'\w+', aff_lower)) - {"university", "of", "the", "and", "school", "college"}
        c_aff_words = set(re.findall(r'\w+', c_affs))
        overlap = aff_words & c_aff_words
        score += len(overlap) * 3

        # Prefer authors with more papers (likely the right person)
        score += min((c.get("paperCount") or 0) / 20, 3)

        scored.append((score, c))

    scored.sort(key=lambda x: -x[0])
    best_score, best = scored[0]

    if best_score >= 5:
        return best, "high"
    elif best_score >= 2:
        return best, "medium"
    else:
        return best, "low"


def get_author_papers(author_id: str) -> list[dict]:
    """Retrieve recent papers for a given S2 author ID."""
    params = {
        "fields": "title,abstract,year,citationCount",
        "limit": MAX_PAPERS,
        "sort": "year:desc"
    }
    try:
        r = requests.get(f"{S2_BASE}/author/{author_id}/papers", params=params,
                         headers=HEADERS, timeout=15)
        r.raise_for_status()
        return r.json().get("data", [])
    except Exception as e:
        print(f"  ⚠ Paper fetch failed for {author_id}: {e}")
        return []


def extract_keywords_from_papers(papers: list[dict], top_n: int = 15) -> list[str]:
    """Extract distinctive keywords from paper titles and abstracts using simple TF approach."""
    from datetime import datetime
    current_year = datetime.now().year

    all_text = []
    for p in papers:
        year = p.get("year") or 0
        if year < current_year - RECENT_YEARS:
            continue
        title = p.get("title") or ""
        abstract = p.get("abstract") or ""
        # Weight titles more heavily
        all_text.append(title.lower() + " " + title.lower())
        if abstract:
            all_text.append(abstract.lower())

    combined = " ".join(all_text)

    # Extract bigrams and unigrams
    words = re.findall(r'[a-z][a-z-]+[a-z]', combined)
    words = [w for w in words if w not in STOP_WORDS and len(w) > 2]

    # Unigrams
    unigram_counts = Counter(words)

    # Bigrams
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
    bigram_counts = Counter(bigrams)

    # Combine: prefer bigrams that occur 2+ times, then top unigrams
    keywords = []
    for bg, count in bigram_counts.most_common(30):
        if count >= 2 and bg.split()[0] not in STOP_WORDS and bg.split()[1] not in STOP_WORDS:
            keywords.append(bg)
        if len(keywords) >= top_n // 2:
            break

    for ug, count in unigram_counts.most_common(50):
        if ug not in " ".join(keywords) and count >= 2:
            keywords.append(ug)
        if len(keywords) >= top_n:
            break

    return keywords


def process_board(input_path: str) -> pd.DataFrame:
    """Main pipeline: read board list, enrich via S2, return enriched DataFrame."""
    df = pd.read_excel(input_path)
    print(f"\n📋 Loaded {len(df)} board members from {input_path}\n")

    results = {
        "S2_Author_ID": [],
        "S2_Paper_Count": [],
        "S2_Recent_Titles": [],
        "S2_Extracted_Keywords": [],
        "S2_Match_Confidence": [],
    }

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Looking up authors"):
        name = row["Name"]
        affiliation = row.get("Location", "")

        # Step 1: Search for author
        candidates = search_author(name, affiliation)
        time.sleep(RATE_LIMIT_DELAY)

        # Step 2: Pick best match
        match, confidence = pick_best_match(candidates, name, affiliation)

        if match is None:
            results["S2_Author_ID"].append("")
            results["S2_Paper_Count"].append(0)
            results["S2_Recent_Titles"].append("")
            results["S2_Extracted_Keywords"].append("")
            results["S2_Match_Confidence"].append("no_results")
            continue

        author_id = match["authorId"]

        # Step 3: Get papers
        papers = get_author_papers(author_id)
        time.sleep(RATE_LIMIT_DELAY)

        # Step 4: Extract keywords
        keywords = extract_keywords_from_papers(papers)
        titles = [p.get("title", "") for p in papers[:10]]

        results["S2_Author_ID"].append(author_id)
        results["S2_Paper_Count"].append(len(papers))
        results["S2_Recent_Titles"].append(" | ".join(titles))
        results["S2_Extracted_Keywords"].append(", ".join(keywords))
        results["S2_Match_Confidence"].append(confidence)

    for col, vals in results.items():
        df[col] = vals

    return df


def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else "ERB_ETP_final.xlsx"
    if not Path(input_path).exists():
        print(f"❌ File not found: {input_path}")
        sys.exit(1)

    df = process_board(input_path)

    output_path = input_path.replace(".xlsx", "_enriched.xlsx")
    df.to_excel(output_path, index=False)

    # Summary
    high = (df["S2_Match_Confidence"] == "high").sum()
    med = (df["S2_Match_Confidence"] == "medium").sum()
    low = (df["S2_Match_Confidence"] == "low").sum()
    none = (df["S2_Match_Confidence"] == "no_results").sum()

    print(f"\n✅ Saved enriched file to: {output_path}")
    print(f"\n📊 Match summary:")
    print(f"   High confidence:  {high}")
    print(f"   Medium confidence: {med}")
    print(f"   Low confidence:   {low}")
    print(f"   No results:       {none}")
    print(f"\n⏱  Estimated time was ~{len(df) * RATE_LIMIT_DELAY * 2 / 60:.0f} minutes")
    print(f"   (2 API calls per author × {RATE_LIMIT_DELAY}s rate limit)")
    print(f"\n💡 Review 'medium' and 'low' confidence matches manually.")
    print(f"   'no_results' authors may need manual Scopus/Google Scholar lookup.")


if __name__ == "__main__":
    main()
