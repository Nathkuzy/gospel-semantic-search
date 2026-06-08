# Semantic Search in the Four Gospels

Semantic search and interactive graph visualization of hidden connections in the KJV corpus of the four Gospels (Matthew, Mark, Luke, John).

## How it works

1. **DataLoader** — parses `bible_kjv.txt` into a structured corpus (3779 verses)
2. **EmbeddingGenerator** — encodes all verses with `all-mpnet-base-v2` (768-dim vectors), caches result to `matrix_cache.npy`
3. **SemanticAnalyzer** — computes cosine similarity between the query vector and the corpus, selects top-k verses, builds a weighted adjacency matrix
4. **GraphVisualizer** — renders an interactive radial graph as a self-contained HTML file via Pyvis

## Setup

```bash
pip install -r requirements.txt
```

Get the corpus (runs once, downloads from Project Gutenberg):

```bash
python download_kjv.py
```

## Usage

Interactive mode:
```bash
python main.py
```

Single query:
```bash
python main.py --query "love your enemies"
```

All options:
```bash
python main.py --help
```

Key flags:
- `--top_k` — number of most relevant verses to retrieve (default: 100)
- `--threshold` — edge weight cutoff after nonlinear scaling (default: 0.3)
- `--show_cluster_edges` — show inter-verse edges in the graph
- `--full_bible` — use the full Bible instead of just the four Gospels
- `--no_open` — don't auto-open the result in a browser

## Output

The result is saved as `output.html` and opened automatically in the browser. Each node is color-coded by Gospel (Matthew=red, Mark=blue, Luke=green, John=orange). Node size and edge length reflect cosine similarity to the query.

## Dataset

King James Version Bible — [Project Gutenberg #10](https://www.gutenberg.org/cache/epub/10/pg10.txt)

## Requirements

- Python 3.9+
- See `requirements.txt`

On first run without a cache file, embedding generation takes ~100 seconds on CPU. Subsequent runs load from `matrix_cache.npy` in under 1 second.
