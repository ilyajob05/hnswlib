import os
import hnswlib
import numpy as np

"""
TurboQuant example: build, search, save/load, and rerank with mmap'd raw vectors.

TurboQuant compresses each vector from dim*4 bytes (float32) to 2*dim+12 bytes,
reducing memory ~1.9x for dim=128.  Re-ranking with exact L2 on mmap'd original
vectors restores recall to near-baseline levels.
"""

dim = 512
num_elements = 100000
num_queries = 200
k = 10

# Generate random data
np.random.seed(42)
data = np.float32(np.random.randn(num_elements, dim))
queries = np.float32(np.random.randn(num_queries, dim))

# --- Build TQ index --------------------------------------------------------
print(f"Building TQ index: {num_elements} vectors, dim={dim}")
tq = hnswlib.TQIndex(dim=dim, bits_per_coord=8)
tq.build(data, M=16, ef_construction=200, num_threads=8)

print(f"  element_count = {tq.element_count}")
print(f"  code_size     = {tq.code_size} bytes/vector "
      f"(vs {dim * 4} bytes for float32, "
      f"{dim * 4 / tq.code_size:.1f}x compression)")

# --- TQ search (compressed distances) --------------------------------------
tq.ef = 64
labels, distances = tq.knn_query(queries, k=k, num_threads=8)

# Measure recall against brute-force
bf = hnswlib.BFIndex(space='l2', dim=dim)
bf.init_index(max_elements=num_elements)
bf.add_items(data)
gt_labels, _ = bf.knn_query(queries, k=k, num_threads=8)

recall_tq = np.mean([
    len(set(labels[i]) & set(gt_labels[i])) / k
    for i in range(num_queries)
])
print(f"\nTQ search recall@{k}: {recall_tq:.4f}")

# --- Save index + raw vectors for rerank -----------------------------------
index_path = "tq_example.bin"
raw_path = "tq_example.tqrv"

print(f"\nSaving index to '{index_path}', raw vectors to '{raw_path}'")
tq.save(index_path, raw_path=raw_path, raw_data=data)
del tq

# --- Load and rerank -------------------------------------------------------
print(f"Loading index from '{index_path}' with raw vectors")
tq2 = hnswlib.TQIndex(dim=dim, bits_per_coord=8)
tq2.load(index_path, raw_path=raw_path)

print(f"  has_raw_vectors = {tq2.has_raw_vectors}")

# Search with rerank: TQ search with ef=100, then exact L2 rerank to top-k
labels_rr, distances_rr = tq2.knn_query_rerank(queries, k=k, ef=100, num_threads=8)

recall_rr = np.mean([
    len(set(labels_rr[i]) & set(gt_labels[i])) / k
    for i in range(num_queries)
])
print(f"TQ + rerank recall@{k}: {recall_rr:.4f}")

# --- L2 baseline for comparison --------------------------------------------
hnsw = hnswlib.Index(space='l2', dim=dim)
hnsw.init_index(max_elements=num_elements, M=16, ef_construction=200)
hnsw.add_items(data)
hnsw.set_ef(64)
labels_l2, _ = hnsw.knn_query(queries, k=k, num_threads=8)

recall_l2 = np.mean([
    len(set(labels_l2[i]) & set(gt_labels[i])) / k
    for i in range(num_queries)
])
print(f"L2 baseline recall@{k}: {recall_l2:.4f}")

# --- Summary ----------------------------------------------------------------
print(f"\n=== Summary (recall@{k}) ===")
print(f"  L2 baseline (ef=64):       {recall_l2:.4f}")
print(f"  TQ8 compressed (ef=64):    {recall_tq:.4f}")
print(f"  TQ8 + rerank (ef=100):     {recall_rr:.4f}")

# Cleanup
os.remove(index_path)
os.remove(raw_path)
