import gc
import os
import time

import numpy as np

from data_loader import DataLoader
from semantic_analyzer import SemanticAnalyzer
from graph_visualizer import GraphVisualizer

CORPUS_PATH = "bible_kjv.txt"
CACHE_PATH = "matrix_cache.npy"


def measure(label, func, repeats=10):
    samples = []
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter()
        func()
        samples.append((time.perf_counter() - start) * 1000.0)
    samples.sort()
    print(f"{label:42s} min {samples[0]:9.3f}  mean {sum(samples)/len(samples):9.3f}  "
          f"median {samples[len(samples)//2]:9.3f}  max {samples[-1]:9.3f}  (n={repeats})")
    return sum(samples) / len(samples)


def peak_rss_mb():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return 0.0


def main():
    print("=" * 86)
    print("СТАТИСТИКА ВИКОНАННЯ ПРОГРАМИ")
    print("=" * 86)

    print("\n[1] ЗАВАНТАЖЕННЯ ТА СТРУКТУРУВАННЯ КОРПУСУ")

    def load_corpus():
        loader = DataLoader(gospels_only=True)
        loader.load_file(CORPUS_PATH)
        loader.flatten_verses()

    measure("Парсинг корпусу та побудова дерева", load_corpus, repeats=10)

    loader = DataLoader(gospels_only=True)
    loader.load_file(CORPUS_PATH)
    verses = loader.flatten_verses()
    print(f"   Віршів Євангелій: {len(verses)}")

    print("\n[2] ЗАВАНТАЖЕННЯ КЕШУ МАТРИЦІ ЕМБЕДДИНГІВ")

    def load_cache():
        cached = np.load(CACHE_PATH, mmap_mode="r")
        _ = np.array(cached, dtype=np.float32)

    measure("Читання matrix_cache.npy", load_cache, repeats=10)

    matrix = np.array(np.load(CACHE_PATH, mmap_mode="r"), dtype=np.float32)
    print(f"   Розмірність: {matrix.shape}, {matrix.nbytes / (1024*1024):.2f} МБ")

    print("\n[3] СЕМАНТИЧНИЙ АНАЛІЗ")
    rng = np.random.default_rng(42)
    query_indices = rng.integers(0, matrix.shape[0], size=30)
    counter = {"i": 0}

    def run_analysis():
        analyzer = SemanticAnalyzer(similarity_threshold=0.3, exponent_k=3)
        qv = matrix[query_indices[counter["i"] % len(query_indices)]]
        counter["i"] += 1
        analyzer.analyze_similarity(qv, matrix, top_k=100)

    measure("analyze_similarity (top_k=100)", run_analysis, repeats=30)

    print("\n[4] ГРАФОВА ВІЗУАЛІЗАЦІЯ")
    analyzer = SemanticAnalyzer(similarity_threshold=0.3, exponent_k=3)
    adj, top_idx, top_scores = analyzer.analyze_similarity(matrix[0], matrix, top_k=100)
    nodes_payload = []
    for idx, score in zip(top_idx, top_scores):
        v = verses[int(idx)]
        nodes_payload.append({
            "book": v["book"], "chapter": v["chapter"], "verse": v["verse"],
            "text": v["text"], "similarity": float(score),
        })

    import tempfile
    bench_html = os.path.join(tempfile.gettempdir(), "_bench_tmp.html")

    def run_visual():
        viz = GraphVisualizer()
        viz.generate_layout(nodes_payload, adj, output_filename=bench_html, query_text="benchmark")

    measure("Побудова HTML-графа", run_visual, repeats=10)
    if os.path.exists(bench_html):
        print(f"   Розмір HTML: {os.path.getsize(bench_html) / 1024:.1f} КБ")
        try:
            os.remove(bench_html)
        except OSError:
            pass

    print("\n[5] МАСШТАБОВАНІСТЬ ЗА РОЗМІРОМ КОРПУСУ")
    for size in (500, 1000, 2000, 3000, matrix.shape[0]):
        sub = matrix[:size]

        def run_sub():
            a = SemanticAnalyzer(similarity_threshold=0.3, exponent_k=3)
            a.analyze_similarity(sub[0], sub, top_k=100)

        measure(f"Корпус {size:5d} віршів", run_sub, repeats=15)

    print("\n[6] МАСШТАБОВАНІСТЬ ЗА top_k")
    for k in (25, 50, 100, 200, 400):
        def run_k():
            a = SemanticAnalyzer(similarity_threshold=0.3, exponent_k=3)
            a.analyze_similarity(matrix[0], matrix, top_k=k)

        measure(f"top_k = {k:3d}", run_k, repeats=15)

    print("\n[7] СПОЖИВАННЯ ПАМ'ЯТІ")
    print(f"   Пік RSS: {peak_rss_mb():.2f} МБ")
    print("=" * 86)


if __name__ == "__main__":
    main()
