import argparse
import os
import sys
import webbrowser
from pathlib import Path

from data_loader import DataLoader
from embedding_generator import EmbeddingGenerator
from semantic_analyzer import SemanticAnalyzer
from graph_visualizer import GraphVisualizer


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CORPUS_PATH = PROJECT_ROOT / "bible_kjv.txt"
DEFAULT_CACHE_PATH = PROJECT_ROOT / "matrix_cache.npy"
DEFAULT_OUTPUT_HTML = PROJECT_ROOT / "output.html"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Semantic search and graph visualization in the four Gospels (KJV)"
    )
    parser.add_argument("--corpus", type=str, default=str(DEFAULT_CORPUS_PATH))
    parser.add_argument("--cache", type=str, default=str(DEFAULT_CACHE_PATH))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_HTML))
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--exponent", type=int, default=3)
    parser.add_argument("--full_bible", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--show_cluster_edges", action="store_true")
    parser.add_argument("--no_open", action="store_true")
    return parser.parse_args()


def run_pipeline(args: argparse.Namespace) -> None:
    print("[1/4] Завантаження корпусу...")
    loader = DataLoader(gospels_only=not args.full_bible)

    if not os.path.exists(args.corpus):
        print(f"Файл корпусу не знайдено: {args.corpus}")
        print("Запустіть: python download_kjv.py")
        sys.exit(1)

    loader.load_file(args.corpus)
    verses = loader.flatten_verses()
    texts = [v["text"] for v in verses]
    print(f"      {len(verses)} віршів з {len(loader.parsed_corpus)} книг")

    print("[2/4] Завантаження моделі MPNet...")
    embedder = EmbeddingGenerator(cache_path=args.cache, device=args.device)

    print("[3/4] Векторизація корпусу...")
    matrix = embedder.generate_embeddings(texts, batch_size=32)
    print(f"      матриця {matrix.shape}, {embedder.memory_footprint_mb():.2f} МБ")

    analyzer = SemanticAnalyzer(
        similarity_threshold=args.threshold,
        exponent_k=args.exponent,
    )
    visualizer = GraphVisualizer(show_inter_verse_edges=args.show_cluster_edges)

    queries = [args.query] if args.query else None

    print("[4/4] Семантичний аналіз та візуалізація.")
    while True:
        if queries is None:
            try:
                query = input("\nЗапит (або 'exit'): ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            if query.lower() in {"exit", "quit", "q"}:
                break
            if not query:
                continue
        else:
            if not queries:
                break
            query = queries.pop(0)
            print(f"\nЗапит: {query}")

        query_vector = embedder.encode_query(query)
        adj_matrix, top_indices, top_scores = analyzer.analyze_similarity(
            query_vector, matrix, top_k=args.top_k
        )

        nodes_payload = []
        for idx, score in zip(top_indices, top_scores):
            v = verses[int(idx)]
            nodes_payload.append({
                "book": v["book"],
                "chapter": v["chapter"],
                "verse": v["verse"],
                "uid": v["uid"],
                "text": v["text"],
                "similarity": float(score),
            })

        visualizer.generate_layout(
            nodes_payload, adj_matrix,
            output_filename=args.output,
            query_text=query,
        )
        print(f"      збережено {args.output} ({len(nodes_payload)} вузлів, {adj_matrix.nnz // 2} ребер)")

        if not args.no_open:
            webbrowser.open(f"file://{os.path.abspath(args.output)}")

        if queries is not None and not queries:
            break


def main() -> None:
    args = parse_arguments()
    run_pipeline(args)


if __name__ == "__main__":
    main()
