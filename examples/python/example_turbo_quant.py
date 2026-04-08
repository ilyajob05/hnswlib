import os
import time
import hnswlib
import numpy as np

"""
TurboQuant example: build, search, save/load, rerank, and performance benchmark.

TurboQuant compresses each vector from dim*4 bytes (float32) to 2*dim+12 bytes,
reducing memory ~1.9x for dim=128.  Re-ranking with exact L2 on mmap'd original
vectors restores recall to near-baseline levels.

Key parameters:
  - ef: controls search quality for both TQ and L2 (higher = better recall, slower)
  - rerank_ef: number of TQ candidates to re-rank with exact L2 distances.
    Should be >= ef. Good starting point: 2x-5x of k.
    Default (0) = use current ef value.
"""

dim = 512
num_elements = 100000
num_queries = 10000
k = 10
ef = 200            # same ef for fair comparison between L2 and TQ
rerank_ef = 200     # candidates to re-rank (>= ef for best results)
M = 16
ef_construction = 200
num_threads = 8

np.random.seed(42)
data = np.float32(np.random.randn(num_elements, dim))
queries = np.float32(np.random.randn(num_queries, dim))

# --- Ground truth (brute-force) --------------------------------------------
print("Computing ground truth (brute-force)...")
bf = hnswlib.BFIndex(space='l2', dim=dim)
bf.init_index(max_elements=num_elements)
bf.add_items(data)
gt_labels, _ = bf.knn_query(queries, k=k, num_threads=num_threads)
del bf

def recall(labels):
    return np.mean([
        len(set(labels[i]) & set(gt_labels[i])) / k
        for i in range(num_queries)
    ])

# --- L2 baseline -----------------------------------------------------------
print(f"\n--- L2 baseline (ef={ef}) ---")
t0 = time.time()
hnsw = hnswlib.Index(space='l2', dim=dim)
hnsw.init_index(max_elements=num_elements, M=M, ef_construction=ef_construction)
hnsw.add_items(data, num_threads=num_threads)
build_l2 = time.time() - t0
print(f"  Build: {build_l2:.2f}s")

hnsw.set_ef(ef)
t0 = time.time()
labels_l2, _ = hnsw.knn_query(queries, k=k, num_threads=num_threads)
search_l2 = time.time() - t0
recall_l2 = recall(labels_l2)
qps_l2 = num_queries / search_l2
print(f"  Search: {search_l2 * 1000:.1f}ms ({qps_l2:.0f} QPS), recall@{k}={recall_l2:.4f}")
del hnsw

# --- TQ build --------------------------------------------------------------
print(f"\n--- TQ8 (ef={ef}) ---")
t0 = time.time()
tq = hnswlib.TQIndex(dim=dim, bits_per_coord=8)
tq.build(data, M=M, ef_construction=ef_construction, num_threads=num_threads)
build_tq = time.time() - t0
print(f"  Build: {build_tq:.2f}s")
print(f"  Code size: {tq.code_size} B/vec (vs {dim * 4} B, "
      f"{dim * 4 / tq.code_size:.1f}x compression)")

# --- TQ search (no rerank) ------------------------------------------------
tq.ef = ef
t0 = time.time()
labels_tq, _ = tq.knn_query(queries, k=k, num_threads=num_threads)
search_tq = time.time() - t0
recall_tq = recall(labels_tq)
qps_tq = num_queries / search_tq
print(f"  Search: {search_tq * 1000:.1f}ms ({qps_tq:.0f} QPS), recall@{k}={recall_tq:.4f}")

# --- Save + Load with raw vectors -----------------------------------------
index_path = "tq_example.bin"
raw_path = "tq_example.tqrv"
tq.save(index_path, raw_path=raw_path, raw_data=data)
del tq

tq = hnswlib.TQIndex(dim=dim, bits_per_coord=8)
tq.load(index_path, raw_path=raw_path)

# --- TQ + rerank -----------------------------------------------------------
# ef controls TQ graph traversal, rerank_ef controls how many candidates
# are re-ranked with exact L2. Set ef >= rerank_ef for efficiency.
print(f"\n--- TQ8 + rerank (ef={ef}, rerank_ef={rerank_ef}) ---")
tq.ef = ef
t0 = time.time()
labels_rr, _ = tq.knn_query_rerank(queries, k=k, rerank_ef=rerank_ef, num_threads=num_threads)
search_rr = time.time() - t0
recall_rr = recall(labels_rr)
qps_rr = num_queries / search_rr
print(f"  Search: {search_rr * 1000:.1f}ms ({qps_rr:.0f} QPS), recall@{k}={recall_rr:.4f}")

# --- Summary ---------------------------------------------------------------
print(f"\n{'='*55}")
print(f"{'Method':<30} {'recall@'+str(k):<12} {'QPS':<10} {'Build'}")
print(f"{'-'*55}")
print(f"{'L2 (ef='+str(ef)+')':<30} {recall_l2:<12.4f} {qps_l2:<10.0f} {build_l2:.2f}s")
print(f"{'TQ8 (ef='+str(ef)+')':<30} {recall_tq:<12.4f} {qps_tq:<10.0f} {build_tq:.2f}s")
print(f"{'TQ8+rerank (rerank_ef='+str(rerank_ef)+')':<30} {recall_rr:<12.4f} {qps_rr:<10.0f} {build_tq:.2f}s")
print(f"{'='*55}")
print(f"\nMemory: {tq.code_size} B/vec TQ vs {dim*4} B/vec L2 "
      f"({dim*4/tq.code_size:.1f}x compression)")

# Cleanup
del tq
os.remove(index_path)
os.remove(raw_path)
