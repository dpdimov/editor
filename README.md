# ERB Editorial Toolkit

A suite of tools for journal editors: reviewer matching, board enrichment, and research topic intelligence.

---

## Files

| File | Purpose |
|------|---------|
| `app.py` | **Streamlit web app** — shareable reviewer finder for all editors |
| `find_reviewers.py` | Command-line reviewer finder (interactive, single, or batch mode) |
| `enrich_board.py` | Enriches board member keywords via Semantic Scholar publications |
| `topic_explorer.py` | Research intelligence: trends, emerging scholars, topic maps, board gaps |

---

## 1. Streamlit App — Reviewer Finder for the Editorial Team

### Setup

```bash
pip install streamlit pandas openpyxl scikit-learn
```

### Run locally

Place `ERB_ETP_enriched.xlsx` in the same folder as `app.py`, then:

```bash
streamlit run app.py
```

Opens in your browser. Editors paste an abstract, enter keywords, exclude author names, and get ranked reviewer suggestions.

### Deploy for the team

**Option A — Streamlit Community Cloud (free, easiest):**
1. Push `app.py`, `ERB_ETP_enriched.xlsx`, and a `requirements.txt` to a GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo and deploy
4. Share the URL with your editors

`requirements.txt`:
```
streamlit
pandas
openpyxl
scikit-learn
```

**Option B — Internal server:**
```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

**Option C — Point to a different board file:**
```bash
BOARD_FILE=/path/to/your/enriched_file.xlsx streamlit run app.py
```

---

## 2. Topic Explorer — Research Intelligence

### Setup

```bash
pip install pandas requests tqdm matplotlib
export S2_API_KEY=your_key_here
```

### Run

```bash
python topic_explorer.py
```

### What it does

**Topic Trend** — Track how a research area has grown or declined over time. Useful for spotting whether a submitted paper is riding a wave or entering a saturated space.

```
Topic to track: entrepreneurial ecosystems
How many years back? 10
→ Year-by-year paper counts, citation averages, top paper per year
```

**Emerging Scholars** — Find rising researchers who are publishing actively in a topic but may not yet be on your radar. Great for refreshing your reviewer pool or identifying potential new board members.

```
Topic to search: digital entrepreneurship
→ Ranked list of researchers by recent output + citation momentum
```

**Landmark Papers** — The most-cited foundational works in a topic. Useful as a quick orientation when handling a paper in a sub-field you're less familiar with, or for checking whether a submission is citing the right foundations.

```
Topic to search: effectuation theory
→ Top 20 papers by citation count with authors, years, journals
```

**Topic Map** — Shows what sub-topics and related concepts cluster around a research area. Helpful for understanding the intellectual neighbourhood of a paper, or for planning special issues.

```
Topic to map: social entrepreneurship
→ Ranked bigrams and concepts from recent papers in the area
```

**Board Gap Analysis** — Compares a list of topics against your editorial board's expertise. Flags topics where you have active publication volume in the field but fewer than 2 board members covering it.

```
Enter topics: crowdfunding, digital platforms, AI entrepreneurship, ...
→ Table showing board coverage vs. publication activity per topic
```

---

## Typical Workflows

### Day-to-day: finding reviewers for a new submission
1. Open the Streamlit app (or run `find_reviewers.py`)
2. Paste the abstract and keywords
3. Exclude the authors
4. Send invitations to the top-ranked board members

### Quarterly: refreshing the data
1. Re-run `enrich_board.py` to capture new publications
2. Replace the enriched file in the app directory

### Annually: strategic planning
1. Use **Topic Trend** to check which areas are growing
2. Use **Board Gap Analysis** to spot expertise gaps
3. Use **Emerging Scholars** to identify candidates for board expansion
4. Use **Landmark Papers** to orient on new sub-fields

---

## Notes

- All S2 tools respect rate limits automatically. With an API key, queries run ~3× faster.
- The enrichment script takes ~20–30 minutes for ~300 board members.
- All exploration tools save results as CSV files for further analysis.
- The Streamlit app caches the board data and TF-IDF index, so it loads fast after the first run.
