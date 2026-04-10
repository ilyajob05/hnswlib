"""
TurboQuant sweep example.

Builds an L2 float32 baseline and a set of TQ indices (one per bits_per_coord),
then sweeps over (ef, rerank_ef) combinations and prints a comparison table.

Edit PARAM_GRID below to pick which configurations to benchmark.
"""

import os
import time

import hnswlib
import numpy as np

# ---------------------------------------------------------------------------
# Dataset / graph parameters
# ---------------------------------------------------------------------------

DIM = 128
NUM_ELEMENTS = 100_000
NUM_QUERIES = 1_000
K = 10
M = 16
EF_CONSTRUCTION = 200
NUM_THREADS = 8
SEED = 42

# ---------------------------------------------------------------------------
# Sweep table. Each row is one benchmark run.
#   method:    "l2" | "tq" | "tq_rerank" | "tq_l2graph_rerank"
#     - "l2":                pure float32 L2 HNSW
#     - "tq":                TQ graph + TQ asymmetric search (no rerank)
#     - "tq_rerank":         TQ graph + TQ search + L2 rerank (from inmem raw)
#     - "tq_l2graph_rerank": L2 graph + TQ search + L2 rerank
#                            (best recall/memory tradeoff on SIFT1M)
#   bits:      4 or 8 (ignored for "l2")
#   ef:        HNSW search ef
#   rerank_ef: shortlist size for rerank (ignored unless *_rerank)
#
# Indices are built once per (method, bits) — sweeping ef is cheap.
# ---------------------------------------------------------------------------

PARAM_GRID = [
    # L2 baseline sweep
    dict(method="l2",        bits=None, ef=32,  rerank_ef=0),
    dict(method="l2",        bits=None, ef=64,  rerank_ef=0),
    dict(method="l2",        bits=None, ef=128, rerank_ef=0),
    dict(method="l2",        bits=None, ef=200, rerank_ef=0),

    # TQ8 pure search
    dict(method="tq",        bits=8,    ef=64,  rerank_ef=0),
    dict(method="tq",        bits=8,    ef=128, rerank_ef=0),
    dict(method="tq",        bits=8,    ef=200, rerank_ef=0),

    # TQ8 + L2 rerank
    dict(method="tq_rerank", bits=8,    ef=64,  rerank_ef=128),
    dict(method="tq_rerank", bits=8,    ef=128, rerank_ef=256),
    dict(method="tq_rerank", bits=8,    ef=200, rerank_ef=400),

    # TQ4 pure search
    dict(method="tq",        bits=4,    ef=128, rerank_ef=0),
    dict(method="tq",        bits=4,    ef=200, rerank_ef=0),

    # TQ4 + L2 rerank
    dict(method="tq_rerank", bits=4,    ef=128, rerank_ef=256),
    dict(method="tq_rerank", bits=4,    ef=200, rerank_ef=400),
    dict(method="tq_rerank", bits=4,    ef=500, rerank_ef=1000),

    # L2-graph + TQ4 search + L2 rerank (winning preset on SIFT1M)
    dict(method="tq_l2graph_rerank", bits=4, ef=64,  rerank_ef=128),
    dict(method="tq_l2graph_rerank", bits=4, ef=128, rerank_ef=256),
    dict(method="tq_l2graph_rerank", bits=4, ef=200, rerank_ef=400),

    # L2-graph + TQ8 search + L2 rerank
    dict(method="tq_l2graph_rerank", bits=8, ef=64,  rerank_ef=128),
    dict(method="tq_l2graph_rerank", bits=8, ef=128, rerank_ef=256),
]


def build_l2(data):
    t0 = time.time()
    idx = hnswlib.Index(space="l2", dim=DIM)
    idx.init_index(max_elements=NUM_ELEMENTS, M=M, ef_construction=EF_CONSTRUCTION)
    idx.add_items(data, num_threads=NUM_THREADS)
    return idx, time.time() - t0


def build_tq(data, bits):
    t0 = time.time()
    idx = hnswlib.TQIndex(dim=DIM, bits_per_coord=bits)
    idx.build(data, M=M, ef_construction=EF_CONSTRUCTION, num_threads=NUM_THREADS)
    return idx, time.time() - t0


def build_tq_l2graph(data, bits):
    t0 = time.time()
    idx = hnswlib.TQIndex(dim=DIM, bits_per_coord=bits)
    idx.build_from_l2(
        data,
        M=M,
        ef_construction=EF_CONSTRUCTION,
        keep_raw=True,
        num_threads=NUM_THREADS,
    )
    return idx, time.time() - t0


def run_sweep(data, queries, gt_labels, grid):
    # Cache one index per (method, bits); sweeping ef/rerank_ef is cheap.
    cache = {}
    results = []

    def get_index(method, bits):
        # Cache key groups methods that can share an index:
        #   "l2"                 — shared by all "l2" rows
        #   ("tq", bits)         — shared by "tq" and "tq_rerank"
        #   ("tq_l2g", bits)     — separate: graph topology differs
        if method == "l2":
            key = ("l2",)
        elif method in ("tq", "tq_rerank"):
            key = ("tq", bits)
        elif method == "tq_l2graph_rerank":
            key = ("tq_l2g", bits)
        else:
            raise ValueError(f"unknown method: {method}")

        if key not in cache:
            if key[0] == "l2":
                print("[build] L2 float32...")
                cache[key] = build_l2(data)
            elif key[0] == "tq":
                print(f"[build] TQ{bits} (TQ graph)...")
                cache[key] = build_tq(data, bits)
            else:
                print(f"[build] TQ{bits} (L2 graph + TQ search)...")
                cache[key] = build_tq_l2graph(data, bits)
            print(f"        done in {cache[key][1]:.2f}s")
        return cache[key]

    for row in grid:
        method = row["method"]
        bits = row["bits"]
        ef = row["ef"]
        rerank_ef = row["rerank_ef"]

        idx, build_time = get_index(method, bits)

        if method == "l2":
            idx.set_ef(ef)
            t0 = time.time()
            labels, _ = idx.knn_query(queries, k=K, num_threads=NUM_THREADS)
            dt = time.time() - t0
            label = f"L2"
        elif method == "tq":
            idx.ef = ef
            t0 = time.time()
            labels, _ = idx.knn_query(queries, k=K, num_threads=NUM_THREADS)
            dt = time.time() - t0
            label = f"TQ{bits}"
        elif method == "tq_rerank":
            idx.ef = ef
            t0 = time.time()
            labels, _ = idx.knn_query_rerank(
                queries, k=K, rerank_ef=rerank_ef, num_threads=NUM_THREADS
            )
            dt = time.time() - t0
            label = f"TQ{bits}+rerank"
        elif method == "tq_l2graph_rerank":
            idx.ef = ef
            t0 = time.time()
            labels, _ = idx.knn_query_rerank(
                queries, k=K, rerank_ef=rerank_ef, num_threads=NUM_THREADS
            )
            dt = time.time() - t0
            label = f"TQ{bits}+L2graph+rr"
        else:
            raise ValueError(f"unknown method: {method}")

        rec = np.mean(
            [
                len(set(labels[i]) & set(gt_labels[i])) / K
                for i in range(NUM_QUERIES)
            ]
        )
        qps = NUM_QUERIES / dt
        us_per_q = dt / NUM_QUERIES * 1e6

        results.append(
            dict(
                label=label,
                ef=ef,
                rerank_ef=rerank_ef,
                recall=rec,
                qps=qps,
                us=us_per_q,
                build=build_time,
            )
        )

    return results, cache


def print_table(results):
    hdr = f"{'Method':<16}{'ef':>6}{'rerank_ef':>12}{'recall@K':>12}{'QPS':>12}{'us/q':>10}{'build_s':>10}"
    print()
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(
            f"{r['label']:<16}{r['ef']:>6}{r['rerank_ef']:>12}"
            f"{r['recall']:>12.4f}{r['qps']:>12.0f}{r['us']:>10.1f}{r['build']:>10.2f}"
        )
    print("=" * len(hdr))


def main():
    np.random.seed(SEED)
    data = np.float32(np.random.randn(NUM_ELEMENTS, DIM))
    queries = np.float32(np.random.randn(NUM_QUERIES, DIM))

    print(f"Computing brute-force GT ({NUM_QUERIES} x {NUM_ELEMENTS}, d={DIM})...")
    t0 = time.time()
    bf = hnswlib.BFIndex(space="l2", dim=DIM)
    bf.init_index(max_elements=NUM_ELEMENTS)
    bf.add_items(data)
    gt_labels, _ = bf.knn_query(queries, k=K, num_threads=NUM_THREADS)
    del bf
    print(f"  done in {time.time() - t0:.1f}s")

    results, cache = run_sweep(data, queries, gt_labels, PARAM_GRID)
    print_table(results)

    tq_any = next(
        (v[0] for k, v in cache.items() if k[0] in ("tq", "tq_l2g")),
        None,
    )
    if tq_any is not None:
        print(
            f"\nMemory: TQ code_size={tq_any.code_size} B/vec "
            f"vs L2 {DIM * 4} B/vec "
            f"({DIM * 4 / tq_any.code_size:.1f}x compression)"
        )


if __name__ == "__main__":
    main()
