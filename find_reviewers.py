"""
ERB Reviewer Finder Tool
=========================
Given a paper's abstract and keywords, ranks editorial board members by
relevance using TF-IDF cosine similarity.

Works with both the original spreadsheet and the enriched version
(if you've run enrich_board.py first).

Requirements:
    pip install pandas openpyxl scikit-learn

Usage — Interactive mode:
    python find_reviewers.py ERB_ETP_final.xlsx

Usage — Command-line mode:
    python find_reviewers.py ERB_ETP_final.xlsx --abstract "This paper examines..." --keywords "crowdfunding, entrepreneurial finance"

Usage — Batch mode (for multiple papers):
    python find_reviewers.py ERB_ETP_final.xlsx --batch papers.csv

    where papers.csv has columns: paper_id, title, abstract, keywords
"""

import sys
import re
import argparse
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_board(path: str) -> pd.DataFrame:
    """Load the board member spreadsheet and build text profiles."""
    df = pd.read_excel(path)

    profiles = []
    for _, row in df.iterrows():
        parts = []
        # Areas of expertise (weighted by repeating)
        aoe = str(row.get("Areas_of_Expertise", ""))
        if aoe and aoe != "nan":
            parts.append(aoe)
            parts.append(aoe)  # double-weight

        # Keywords (weighted)
        kw = str(row.get("Keywords", ""))
        if kw and kw != "nan":
            parts.append(kw)
            parts.append(kw)  # double-weight

        # If enriched data exists, include it
        s2_kw = str(row.get("S2_Extracted_Keywords", ""))
        if s2_kw and s2_kw != "nan":
            parts.append(s2_kw)
            parts.append(s2_kw)  # double-weight — these are from actual papers

        s2_titles = str(row.get("S2_Recent_Titles", ""))
        if s2_titles and s2_titles != "nan":
            parts.append(s2_titles)

        profiles.append(" ".join(parts).lower())

    df["_profile"] = profiles
    return df


def build_matcher(df: pd.DataFrame):
    """Build TF-IDF matrix from board member profiles."""
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),       # unigrams + bigrams
        stop_words="english",
        min_df=1,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(df["_profile"])
    return vectorizer, tfidf_matrix


def find_reviewers(query_text: str, df: pd.DataFrame, vectorizer, tfidf_matrix,
                   top_n: int = 15, exclude_names: list = None) -> pd.DataFrame:
    """Rank board members by similarity to the query text."""
    query_vec = vectorizer.transform([query_text.lower()])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    df = df.copy()
    df["Relevance_Score"] = scores

    if exclude_names:
        exclude_lower = [n.lower().strip() for n in exclude_names]
        df = df[~df["Name"].str.lower().str.strip().isin(exclude_lower)]

    results = df.nlargest(top_n, "Relevance_Score")
    results = results[results["Relevance_Score"] > 0.01]  # filter noise

    return results[["Name", "Location", "Areas_of_Expertise", "Keywords",
                     "Relevance_Score"]].copy()


def format_results(results: pd.DataFrame) -> str:
    """Pretty-print the results."""
    lines = []
    lines.append(f"\n{'='*80}")
    lines.append(f"  TOP {len(results)} REVIEWER MATCHES")
    lines.append(f"{'='*80}\n")

    for rank, (_, row) in enumerate(results.iterrows(), 1):
        score_bar = "█" * int(row["Relevance_Score"] * 50)
        lines.append(f"  {rank:2d}. {row['Name']}")
        lines.append(f"      {row['Location']}")
        lines.append(f"      Score: {row['Relevance_Score']:.3f} {score_bar}")
        expertise = str(row.get("Areas_of_Expertise", ""))
        if expertise and expertise != "nan":
            lines.append(f"      Expertise: {expertise[:100]}...")
        lines.append("")

    return "\n".join(lines)


def interactive_mode(df, vectorizer, tfidf_matrix):
    """Run an interactive loop for finding reviewers."""
    print("\n" + "="*60)
    print("  ERB REVIEWER FINDER — Interactive Mode")
    print("="*60)
    print(f"\n  Loaded {len(df)} board members.")
    print("  Type 'quit' to exit.\n")

    while True:
        print("-" * 60)
        abstract = input("\n📝 Paste the paper ABSTRACT (or 'quit'):\n> ").strip()
        if abstract.lower() in ("quit", "exit", "q"):
            break

        keywords = input("\n🔑 Enter paper KEYWORDS (comma-separated):\n> ").strip()

        exclude = input("\n🚫 Names to EXCLUDE (e.g. authors, comma-separated, or press Enter):\n> ").strip()
        exclude_names = [n.strip() for n in exclude.split(",")] if exclude else None

        top_n_str = input("\n📊 How many suggestions? (default 10):\n> ").strip()
        top_n = int(top_n_str) if top_n_str.isdigit() else 10

        # Build query from abstract + keywords (keywords weighted)
        query = f"{abstract} {keywords} {keywords} {keywords}"

        results = find_reviewers(query, df, vectorizer, tfidf_matrix,
                                 top_n=top_n, exclude_names=exclude_names)
        print(format_results(results))

        # Offer to save
        save = input("💾 Save results to CSV? (y/n): ").strip().lower()
        if save == "y":
            out_path = "reviewer_suggestions.csv"
            results.to_csv(out_path, index=False)
            print(f"   Saved to {out_path}")


def batch_mode(df, vectorizer, tfidf_matrix, batch_path: str, top_n: int = 10):
    """Process multiple papers from a CSV file."""
    papers = pd.read_csv(batch_path)
    print(f"\n📋 Processing {len(papers)} papers from {batch_path}\n")

    all_results = []
    for _, paper in papers.iterrows():
        paper_id = paper.get("paper_id", paper.get("title", "unknown"))
        abstract = str(paper.get("abstract", ""))
        keywords = str(paper.get("keywords", ""))
        query = f"{abstract} {keywords} {keywords} {keywords}"

        results = find_reviewers(query, df, vectorizer, tfidf_matrix, top_n=top_n)
        results.insert(0, "Paper", paper_id)
        all_results.append(results)

    combined = pd.concat(all_results, ignore_index=True)
    out_path = batch_path.replace(".csv", "_reviewers.csv")
    combined.to_csv(out_path, index=False)
    print(f"\n✅ Saved batch results to: {out_path}")
    return combined


def main():
    parser = argparse.ArgumentParser(description="ERB Reviewer Finder")
    parser.add_argument("board_file", help="Path to the board member Excel file")
    parser.add_argument("--abstract", help="Paper abstract text")
    parser.add_argument("--keywords", help="Paper keywords (comma-separated)")
    parser.add_argument("--exclude", help="Author names to exclude (comma-separated)")
    parser.add_argument("--top", type=int, default=10, help="Number of suggestions")
    parser.add_argument("--batch", help="Path to a CSV of papers for batch processing")
    args = parser.parse_args()

    if not Path(args.board_file).exists():
        print(f"❌ File not found: {args.board_file}")
        sys.exit(1)

    print("🔄 Loading board members and building index...")
    df = load_board(args.board_file)
    vectorizer, tfidf_matrix = build_matcher(df)
    print(f"✅ Index built: {len(df)} members, {tfidf_matrix.shape[1]} features")

    if args.batch:
        batch_mode(df, vectorizer, tfidf_matrix, args.batch, top_n=args.top)
    elif args.abstract:
        keywords = args.keywords or ""
        query = f"{args.abstract} {keywords} {keywords} {keywords}"
        exclude_names = [n.strip() for n in args.exclude.split(",")] if args.exclude else None
        results = find_reviewers(query, df, vectorizer, tfidf_matrix,
                                 top_n=args.top, exclude_names=exclude_names)
        print(format_results(results))
    else:
        interactive_mode(df, vectorizer, tfidf_matrix)


if __name__ == "__main__":
    main()
