import time

from data_loader import DataLoader
from embedding_generator import EmbeddingGenerator

print("=" * 70)
print("ЗАМІР ТРАНСФОРМЕРНИХ ЕТАПІВ")
print("=" * 70)

start = time.perf_counter()
embedder = EmbeddingGenerator()
model_load = time.perf_counter() - start
print(f"Завантаження моделі MPNet: {model_load:.3f} с")
print(f"Пристрій: {embedder._device}")

# розігрів
embedder.encode_query("warmup query")

queries = [
    "love your enemies", "the kingdom of heaven", "forgiveness of sins",
    "faith and healing", "a good shepherd", "the light of the world",
]
samples = []
for q in queries * 5:
    start = time.perf_counter()
    embedder.encode_query(q)
    samples.append((time.perf_counter() - start) * 1000.0)
samples.sort()
print(f"Кодування одного запиту: min {samples[0]:.1f} ms  "
      f"mean {sum(samples)/len(samples):.1f} ms  "
      f"median {samples[len(samples)//2]:.1f} ms  "
      f"max {samples[-1]:.1f} ms  (n={len(samples)})")

loader = DataLoader(gospels_only=True)
loader.load_file("bible_kjv.txt")
verses = loader.flatten_verses()
sample_texts = [v["text"] for v in verses[:256]]

start = time.perf_counter()
embedder._model.encode(sample_texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
batch_time = time.perf_counter() - start
throughput = 256 / batch_time
print(f"Векторизація 256 віршів: {batch_time:.3f} с ({throughput:.1f} віршів/с)")
print(f"Оцінка повної векторизації ({len(verses)} віршів): {len(verses)/throughput:.1f} с")
print("=" * 70)
