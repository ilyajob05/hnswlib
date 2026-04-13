"""
TurboQuant Grid Search — Systematic parameter sweep on real datasets.

Goal: find optimal (bits, M, ef_construction, ef) for max recall without rerank,
      compare against L2 baseline and rerank variants.

Datasets:
  - bigann_sift128: SIFT descriptors, dim=128
  - glove100_128:   GloVe 100d padded to 128
  - glove200_256:   GloVe 200d padded to 256

Usage:
  python test_tq_grid_search.py [--base-dir /path/to/hnswlib] [--max-elements 100000]
  python test_tq_grid_search.py --datasets bigann_sift128  # single dataset
"""

import argparse
import csv
import os
import struct
import time

import hnswlib
import numpy as np

# ---------------------------------------------------------------------------
# Parameter grid — edit this to control the sweep
# ---------------------------------------------------------------------------

# Build configurations: (bits, M, ef_construction)
BUILD_GRID = [
    # bits  M   ef_c
    (4, 16, 100),
    (4, 16, 200),
    (4, 16, 400),
    (4, 32, 400),
    (4, 32, 800),
    (4, 48, 600),
    (8, 16, 100),
    (8, 16, 200),
    (8, 16, 400),
    (8, 32, 400),
    (8, 32, 800),
    (8, 48, 600),
]

# Search ef values to sweep for each build config
SEARCH_EFS = [64, 128, 256, 512, 1024]

# L2 baseline build params
L2_BUILDS = [
    # M   ef_c
    (16, 100),
    (16, 200),
    (32, 200),
    (32, 400),
]

# L2 baseline search ef
L2_SEARCH_EFS = [64, 128, 256, 512]

# Rerank configurations: (bits, M, ef_c, search_ef, rerank_ef)
RERANK_GRID = [
    # bits  M   ef_c   ef   rr_ef
    (4, 16, 200, 64, 128),
    (4, 16, 200, 128, 256),
    (4, 16, 200, 256, 512),
    (4, 16, 200, 512, 1024),
    (8, 16, 200, 64, 128),
    (8, 16, 200, 128, 256),
    (8, 16, 200, 256, 512),
    (8, 16, 200, 512, 1024),
]

# Use corrected build distance (QJL cross-term) — set to True to include
INCLUDE_CORRECTED_BUILD = False  # negligible effect in prior tests

K = 100  # search for top-100, compute R@1, R@10, R@100
NUM_THREADS = 6


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_bvecs(path, max_elements=0):
    vecs = []
    with open(path, "rb") as f:
        while True:
            buf = f.read(4)
            if len(buf) < 4:
                break
            dim = struct.unpack("<i", buf)[0]
            raw = f.read(dim)
            if len(raw) < dim:
                break
            vecs.append(np.frombuffer(raw, dtype=np.uint8).astype(np.float32))
            if max_elements > 0 and len(vecs) >= max_elements:
                break
    return np.array(vecs) if vecs else None


def load_fvecs(path, max_elements=0):
    vecs = []
    with open(path, "rb") as f:
        while True:
            buf = f.read(4)
            if len(buf) < 4:
                break
            dim = struct.unpack("<i", buf)[0]
            raw = f.read(dim * 4)
            if len(raw) < dim * 4:
                break
            vecs.append(np.frombuffer(raw, dtype=np.float32).copy())
            if max_elements > 0 and len(vecs) >= max_elements:
                break
    return np.array(vecs) if vecs else None


def load_ivecs(path, max_elements=0):
    vecs = []
    with open(path, "rb") as f:
        while True:
            buf = f.read(4)
            if len(buf) < 4:
                break
            dim = struct.unpack("<i", buf)[0]
            raw = f.read(dim * 4)
            if len(raw) < dim * 4:
                break
            vecs.append(np.frombuffer(raw, dtype=np.int32).copy())
            if max_elements > 0 and len(vecs) >= max_elements:
                break
    return np.array(vecs) if vecs else None


def pad_to_power_of_2(data, target_dim=None):
    n, d = data.shape
    if target_dim is None:
        target_dim = 1
        while target_dim < d:
            target_dim <<= 1
    if target_dim == d:
        return data
    padded = np.zeros((n, target_dim), dtype=np.float32)
    padded[:, :d] = data
    return padded


# ---------------------------------------------------------------------------
# Dataset definitions
# ---------------------------------------------------------------------------

DATASETS = {
    "bigann_sift128": {
        "base": "bigann/bigann_base.bvecs",
        "query": "bigann/bigann_query.bvecs",
        "gt": "bigann/gnd/idx_1M.ivecs",
        "loader": "bvecs",
        "dim": 128,
    },
    "glove100_128": {
        "base": "glove/glove100_128/base.fvecs",
        "query": "glove/glove100_128/query.fvecs",
        "gt": "glove/glove100_128/gt.ivecs",
        "loader": "fvecs",
        "dim": 128,
    },
    "glove200_256": {
        "base": "glove/glove200_256/base.fvecs",
        "query": "glove/glove200_256/query.fvecs",
        "gt": "glove/glove200_256/gt.ivecs",
        "loader": "fvecs",
        "dim": 256,
    },
}


def load_dataset(name, base_dir, max_elements=0):
    ds = DATASETS[name]
    loader = load_bvecs if ds["loader"] == "bvecs" else load_fvecs

    base_path = os.path.join(base_dir, ds["base"])
    query_path = os.path.join(base_dir, ds["query"])
    gt_path = os.path.join(base_dir, ds["gt"])

    print(f"  Loading {name}...")
    data = loader(base_path, max_elements=max_elements)
    queries = loader(query_path)
    gt = load_ivecs(gt_path)

    if data is None:
        print(f"  ERROR: cannot load {base_path}")
        return None

    dim = data.shape[1]
    target_dim = ds["dim"]
    if dim != target_dim:
        data = pad_to_power_of_2(data, target_dim)
        queries = pad_to_power_of_2(queries, target_dim)

    print(f"    data: {data.shape}, queries: {queries.shape}, "
          f"gt: {gt.shape if gt is not None else 'N/A'}")
    return data, queries, gt, target_dim


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def recall_at_k(labels, gt_labels, k):
    """Compute recall@k. Returns None if labels has fewer than k columns."""
    if labels.shape[1] < k or gt_labels.shape[1] < k:
        return None
    n = min(len(labels), len(gt_labels))
    return np.mean([
        len(set(labels[i][:k]) & set(gt_labels[i][:k])) / k
        for i in range(n)
    ])


def compute_recalls(labels, gt_labels):
    """Return (R@1, R@10, R@100) — None for any that can't be computed."""
    return (recall_at_k(labels, gt_labels, 1),
            recall_at_k(labels, gt_labels, 10),
            recall_at_k(labels, gt_labels, 100))


def fmt_recall(val):
    """Format recall value or 'N/A'."""
    return f"{val:>7.4f}" if val is not None else "    N/A"


def compute_bruteforce_gt(data, queries, k, num_threads=8):
    dim = data.shape[1]
    bf = hnswlib.BFIndex(space="l2", dim=dim)
    bf.init_index(max_elements=len(data))
    bf.add_items(data)
    labels, _ = bf.knn_query(queries, k=k, num_threads=num_threads)
    del bf
    return labels


# def measure_search(idx, queries, k, ef, num_threads):
#     """Set ef, run knn_query, return (labels, qps). k is clamped to ef."""
#     actual_k = min(k, ef)
#     if hasattr(idx, 'ef'):
#         idx.ef = ef
#     else:
#         idx.set_ef(ef)
#     t0 = time.time()
#     labels, _ = idx.knn_query(queries, k=actual_k, num_threads=num_threads)
#     dt = time.time() - t0
#     qps = len(queries) / dt
#     return labels, qps, actual_k


def measure_search(idx, queries, k, ef, num_threads):
    actual_k = min(k, ef)
    idx.set_ef(ef)

    t0 = time.time()
    labels, _ = idx.knn_query(queries, k=actual_k, num_threads=num_threads)
    dt = time.time() - t0

    qps = len(queries) / dt
    latency_ms = (dt / len(queries)) * 1000  # Average latency per query in ms
    return labels, qps, latency_ms


def measure_rerank(idx, queries, k, ef, rerank_ef, num_threads):
    """Set ef, run knn_query_rerank, return (labels, qps). k clamped to rerank_ef."""
    actual_k = min(k, rerank_ef)
    idx.ef = ef
    t0 = time.time()
    labels, _ = idx.knn_query_rerank(queries, k=actual_k, rerank_ef=rerank_ef,
                                     num_threads=num_threads)
    dt = time.time() - t0
    qps = len(queries) / dt
    return labels, qps, actual_k


# ---------------------------------------------------------------------------
# Grid search runner
# ---------------------------------------------------------------------------

def run_grid_search(data, queries, gt_labels, dim, dataset_name, results):
    N = len(data)
    N_Q = len(queries)

    # ---- L2 baselines ----
    for M, ef_c in L2_BUILDS:
        label = f"L2 M={M}"
        print(f"\n  Building {label} (ef_c={ef_c})...")
        t0 = time.time()
        idx = hnswlib.Index(space="l2", dim=dim)
        idx.init_index(max_elements=N, M=M, ef_construction=ef_c)
        idx.add_items(data, num_threads=NUM_THREADS)
        build_s = time.time() - t0
        print(f"    built in {build_s:.1f}s")

        for ef in L2_SEARCH_EFS:
            labels, qps, actual_k = measure_search(idx, queries, K, ef, NUM_THREADS)
            r1, r10, r100 = compute_recalls(labels, gt_labels)
            row = dict(dataset=dataset_name, method="l2", bits=0, M=M, ef_c=ef_c,
                       ef=ef, rerank_ef=0, recall_1=r1, recall_10=r10, recall_100=r100,
                       qps=qps, build_s=build_s,
                       mem_per_vec=dim * 4)
            results.append(row)
            print(
                f"    ef={ef:>5}  R@1={fmt_recall(r1)}  R@10={fmt_recall(r10)}  R@100={fmt_recall(r100)}  QPS={qps:>8.0f}")
        del idx

    # ---- TQ builds (no rerank) ----
    build_modes = [("tq", False)]
    if INCLUDE_CORRECTED_BUILD:
        build_modes.append(("tq_corr", True))

    for method_label, use_corr in build_modes:
        for bits, M, ef_c in BUILD_GRID:
            tag = f"TQ{bits} M={M} efc={ef_c}"
            if use_corr:
                tag += " corr"
            print(f"\n  Building {tag}...")
            t0 = time.time()
            idx = hnswlib.TQIndex(dim=dim, bits_per_coord=bits)
            idx.build(data, M=M, ef_construction=ef_c,
                      num_threads=NUM_THREADS, use_corrected_build=use_corr)
            build_s = time.time() - t0
            code_sz = idx.code_size
            print(f"    built in {build_s:.1f}s, code_size={code_sz}B")

            for ef in SEARCH_EFS:
                labels, qps, actual_k = measure_search(idx, queries, K, ef, NUM_THREADS)
                r1, r10, r100 = compute_recalls(labels, gt_labels)
                row = dict(dataset=dataset_name, method=method_label,
                           bits=bits, M=M, ef_c=ef_c,
                           ef=ef, rerank_ef=0, recall_1=r1, recall_10=r10, recall_100=r100,
                           qps=qps, build_s=build_s, mem_per_vec=code_sz)
                results.append(row)
                print(
                    f"    ef={ef:>5}  R@1={fmt_recall(r1)}  R@10={fmt_recall(r10)}  R@100={fmt_recall(r100)}  QPS={qps:>8.0f}")
            del idx

    # ---- L2-graph + TQ search (no rerank) ----
    # Build once per bits, use for both TQ search and rerank
    rerank_by_bits = {}
    for bits in sorted(set(b for b, _, _, _, _ in RERANK_GRID)):
        rerank_by_bits[bits] = []
    for bits, M_rr, ef_c_rr, ef_rr, rr_ef in RERANK_GRID:
        rerank_by_bits[bits].append((ef_rr, rr_ef))

    for bits in sorted(set(b for b, _, _ in BUILD_GRID)):
        print(f"\n  Building L2graph+TQ{bits} (M=16, ef_c=200)...")
        t0 = time.time()
        idx = hnswlib.TQIndex(dim=dim, bits_per_coord=bits)
        idx.build_from_l2(data, M=16, ef_construction=200,
                          keep_raw=True, num_threads=NUM_THREADS)
        build_s = time.time() - t0
        code_sz = idx.code_size
        print(f"    built in {build_s:.1f}s")

        # TQ search only
        for ef in SEARCH_EFS:
            labels, qps, actual_k = measure_search(idx, queries, K, ef, NUM_THREADS)
            r1, r10, r100 = compute_recalls(labels, gt_labels)
            row = dict(dataset=dataset_name, method="l2g_tq",
                       bits=bits, M=16, ef_c=200,
                       ef=ef, rerank_ef=0, recall_1=r1, recall_10=r10, recall_100=r100,
                       qps=qps, build_s=build_s, mem_per_vec=code_sz)
            results.append(row)
            print(
                f"    tq{bits} ef={ef:>5}  R@1={fmt_recall(r1)}  R@10={fmt_recall(r10)}  R@100={fmt_recall(r100)}  QPS={qps:>8.0f}")

        # Rerank
        for ef_rr, rr_ef in rerank_by_bits.get(bits, []):
            labels, qps, actual_k = measure_rerank(idx, queries, K, ef_rr, rr_ef, NUM_THREADS)
            r1, r10, r100 = compute_recalls(labels, gt_labels)
            row = dict(dataset=dataset_name, method="l2g_rerank",
                       bits=bits, M=16, ef_c=200,
                       ef=ef_rr, rerank_ef=rr_ef, recall_1=r1, recall_10=r10, recall_100=r100,
                       qps=qps, build_s=build_s,
                       mem_per_vec=code_sz + dim * 4)
            results.append(row)
            print(
                f"    rr{bits} ef={ef_rr:>4} rr={rr_ef:>4}  R@1={fmt_recall(r1)}  R@10={fmt_recall(r10)}  R@100={fmt_recall(r100)}  QPS={qps:>8.0f}")

        del idx


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def print_summary(results):
    print(f"\n{'=' * 125}")
    print("SUMMARY — sorted by dataset, then recall@10 descending")
    print(f"{'=' * 125}")
    hdr = (f"{'Dataset':<18} {'Method':<12} {'bits':>4} {'M':>3} {'efc':>5} "
           f"{'ef':>5} {'rr_ef':>5} {'R@1':>7} {'R@10':>7} {'R@100':>7} {'QPS':>9} "
           f"{'build':>7} {'B/vec':>6}")
    print(hdr)
    print("-" * 125)

    from itertools import groupby
    sorted_results = sorted(results, key=lambda r_: (r_["dataset"], -(r_["recall_10"] or 0)))
    for ds, group in groupby(sorted_results, key=lambda r_: r_["dataset"]):
        for r in group:
            print(f"{r['dataset']:<18} {r['method']:<12} {r['bits']:>4} {r['M']:>3} "
                  f"{r['ef_c']:>5} {r['ef']:>5} {r['rerank_ef']:>5} "
                  f"{fmt_recall(r['recall_1'])} {fmt_recall(r['recall_10'])} {fmt_recall(r['recall_100'])} {r['qps']:>9.0f} "
                  f"{r['build_s']:>7.1f} {r.get('mem_per_vec', 0):>6}")
        print()


def print_best_per_dataset(results):
    """Print best non-rerank config per dataset."""
    print(f"\n{'=' * 80}")
    print("BEST non-rerank configs (highest R@10 with QPS > 5000)")
    print(f"{'=' * 80}")

    from itertools import groupby
    sorted_results = sorted(results, key=lambda r: r["dataset"])
    for ds, group in groupby(sorted_results, key=lambda r: r["dataset"]):
        group = list(group)
        # Filter: no rerank, QPS > 5000
        no_rr = [r for r in group if r["rerank_ef"] == 0 and r["qps"] > 5000 and r["recall_10"] is not None]
        if not no_rr:
            continue
        # Best by recall@10
        best = max(no_rr, key=lambda r: (r["recall_10"] or 0))
        print(f"\n  {ds}:")
        print(f"    method={best['method']} bits={best['bits']} M={best['M']} "
              f"ef_c={best['ef_c']} ef={best['ef']}")
        print(f"    R@1={fmt_recall(best['recall_1'])}  R@10={fmt_recall(best['recall_10'])}  "
              f"R@100={fmt_recall(best['recall_100'])}  "
              f"QPS={best['qps']:.0f}  B/vec={best.get('mem_per_vec', '?')}")

        # Also show best rerank for comparison
        with_rr = [r for r in group if r["rerank_ef"] > 0]
        if with_rr:
            best_rr = max(with_rr, key=lambda r: (r["recall_10"] or 0))
            print(f"    (rerank best: method={best_rr['method']} bits={best_rr['bits']} "
                  f"ef={best_rr['ef']} rr={best_rr['rerank_ef']} "
                  f"R@10={fmt_recall(best_rr['recall_10'])} R@100={fmt_recall(best_rr['recall_100'])} "
                  f"QPS={best_rr['qps']:.0f})")


def analyze_efficiency(results):
    print(f"\n{'=' * 80}")
    print("EFFICIENCY ANALYSIS (Target Recall vs Latency)")
    print(f"{'=' * 80}")

    # Group by dataset
    from collections import defaultdict
    ds_groups = defaultdict(list)
    for r in results:
        ds_groups[r['dataset']].append(r)

    for ds, data in ds_groups.items():
        print(f"\nDataset: {ds}")
        print(f"{'Target R@10':<12} | {'Method':<12} | {'Latency (ms)':<12} | {'QPS':<10} | {'Bits':<5}")
        print("-" * 65)

        # Target recall levels
        for target in [0.5, 0.8, 0.9, 0.95, 0.99]:
            # Filter configs that reached the target
            candidates = [r for r in data if (r['recall_10'] or 0) >= target]

            if candidates:
                # Find the fastest (highest QPS)
                best = max(candidates, key=lambda x: x['qps'])
                latency = 1000 / best['qps']  # Latency per query in ms
                print(
                    f"{target:<12} | {best['method']:<12} | {latency:<12.3f} | {best['qps']:<10.0f} | {best['bits']:<5}")
            else:
                print(f"{target:<12} | {'N/A':<12} | {'-':<12} | {'-':<10} | {'-'}")








def plot_turboquant_comparison(csv_path: str = "tq_grid_search_results.csv", save_prefix: str = "tq_final"):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams['figure.figsize'] = (18, 12)

    """
    Professional TurboQuant results visualization with full method comparison.
    """
    df = pd.read_csv(csv_path)
    df['bits'] = df['bits'].astype(int)

    # Map method names to human-readable labels
    method_map = {
        'l2': 'L2 Baseline',
        'tq': 'TurboQuant',
        'l2g_tq': 'L2-graph + TQ',
        'l2g_rerank': 'L2-graph + Rerank'
    }
    df['method'] = df['method'].map(method_map)

    # ====================== 1. Pareto Front (main plot) ======================
    plt.figure(figsize=(14, 9))

    # Quantized methods only
    quant_df = df[df['method'].isin(['TurboQuant', 'L2-graph + TQ', 'L2-graph + Rerank'])].copy()

    sns.scatterplot(
        data=quant_df,
        x='qps',
        y='recall_10',
        hue='bits',
        style='method',
        size='M',
        palette='viridis',
        sizes=(80, 450),
        alpha=0.9,
        edgecolor='black',
        linewidth=0.8
    )

    # L2 Baseline as a horizontal reference line
    l2_max = df[df['method'] == 'L2 Baseline']['recall_10'].max()
    plt.axhline(y=l2_max, color='red', linestyle='--', linewidth=2.5,
                label=f'L2 Baseline (max recall@10 = {l2_max:.4f})')

    plt.title('Pareto Front: Recall@10 vs Speed (TurboQuant vs L2)', fontsize=16, pad=20)
    plt.xlabel('Queries Per Second (log scale)')
    plt.ylabel('Recall@10')
    plt.xscale('log')
    plt.legend(title='Configuration', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_pareto.png", dpi=300, bbox_inches='tight')
    plt.show()

    # ====================== 2. Full dashboard (4 panels) ======================
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])

    # A. Recall@10 vs efSearch (line plot)
    ax1 = fig.add_subplot(gs[0, 0])
    sns.lineplot(data=quant_df, x='ef', y='recall_10', hue='bits', style='M',
                 markers=True, linewidth=2.5, markersize=8, ax=ax1)
    ax1.set_title('Recall@10 vs efSearch')
    ax1.set_xlabel('efSearch')
    ax1.set_ylabel('Recall@10')

    # B. Recall vs QPS (scatter)
    ax2 = fig.add_subplot(gs[0, 1])
    sns.scatterplot(data=quant_df, x='qps', y='recall_10',
                    hue='bits', style='method', size='M',
                    palette='viridis', sizes=(60, 400), alpha=0.85, ax=ax2)
    ax2.set_xscale('log')
    ax2.set_title('Recall@10 vs Speed')
    ax2.set_xlabel('QPS (log)')
    ax2.set_ylabel('Recall@10')

    # C. TurboQuant heatmap
    ax3 = fig.add_subplot(gs[0, 2])
    tq_pivot = df[df['method'] == 'TurboQuant'].pivot_table(
        values='recall_10', index='M', columns='ef', aggfunc='max')
    sns.heatmap(tq_pivot, annot=True, fmt='.3f', cmap='viridis',
                cbar_kws={'label': 'Recall@10'}, ax=ax3)
    ax3.set_title('Recall@10 Heatmap (TurboQuant)')

    # D. All methods comparison
    ax4 = fig.add_subplot(gs[1, :])
    sns.barplot(data=df, x='bits', y='recall_10', hue='method', palette='Set2', ax=ax4)
    ax4.axhline(y=l2_max, color='red', linestyle='--', linewidth=2.5, label='L2 Baseline')
    ax4.set_title('Recall@10 by Method and Bits')
    ax4.set_ylabel('Recall@10')
    ax4.legend(title='Method')

    plt.suptitle('TurboQuant Grid Search — Full Comparison Dashboard', fontsize=18, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_dashboard.png", dpi=300, bbox_inches='tight')
    plt.show()

    # ====================== 3. Top-15 best configurations ======================
    top = df.nlargest(15, 'recall_10')
    print("\n🔥 TOP-15 best configurations by Recall@10:")
    print(top[['dataset', 'method', 'bits', 'M', 'ef', 'recall_10', 'qps', 'mem_per_vec']]
          .round(4).to_string(index=False))

    print(f"\n✅ Plots saved:\n   • {save_prefix}_pareto.png\n   • {save_prefix}_dashboard.png")









# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TQ Grid Search")
    parser.add_argument("--base-dir", default="/Users/ilia/PRJ/nmslib/hnswlib")
    parser.add_argument("--max-elements", type=int, default=100_000)
    parser.add_argument("--datasets", nargs="*",
                        default=["bigann_sift128", "glove100_128", "glove200_256"])
    parser.add_argument("--output", default="tq_grid_search_results.csv")
    args = parser.parse_args()

    # Print parameter grid for reference
    print("=" * 60)
    print("Parameter Grid")
    print("=" * 60)
    print(f"\nBuild configs ({len(BUILD_GRID)}):")
    print(f"  {'bits':>4} {'M':>4} {'ef_c':>6}")
    for bits, M, ef_c in BUILD_GRID:
        print(f"  {bits:>4} {M:>4} {ef_c:>6}")
    print(f"\nSearch ef: {SEARCH_EFS}")
    print(f"L2 baselines: {L2_BUILDS}")
    print(f"L2 search ef: {L2_SEARCH_EFS}")
    print(f"Rerank configs ({len(RERANK_GRID)}):")
    for bits, M, ef_c, ef, rr_ef in RERANK_GRID:
        print(f"  bits={bits} M={M} ef_c={ef_c} ef={ef} rr={rr_ef}")
    print(f"Include corrected build: {INCLUDE_CORRECTED_BUILD}")
    total_runs = (len(L2_BUILDS) * len(L2_SEARCH_EFS) +
                  len(BUILD_GRID) * len(SEARCH_EFS) * (2 if INCLUDE_CORRECTED_BUILD else 1) +
                  len(set(b for b, _, _ in BUILD_GRID)) * len(SEARCH_EFS) +
                  len(RERANK_GRID))
    print(f"Total search runs: ~{total_runs}")
    print()

    results = []

    for ds_name in args.datasets:
        if ds_name not in DATASETS:
            print(f"Unknown dataset: {ds_name}")
            continue

        print(f"\n{'=' * 70}")
        print(f"Dataset: {ds_name}")
        print(f"{'=' * 70}")

        loaded = load_dataset(ds_name, args.base_dir, max_elements=args.max_elements)
        if loaded is None:
            continue

        data, queries, gt, dim = loaded

        if gt is not None and np.max(gt) < len(data):
            gt_labels = gt
            print(f"  Using pre-computed GT")
        else:
            print(f"  Recomputing GT (brute force, K={K})...")
            t0 = time.time()
            gt_labels = compute_bruteforce_gt(data, queries, K, NUM_THREADS)
            print(f"    done in {time.time() - t0:.1f}s")

        run_grid_search(data, queries, gt_labels, dim, ds_name, results)

    # Write CSV
    if results:
        csv_path = os.path.join(args.base_dir, args.output)
        fields = ["dataset", "method", "bits", "M", "ef_c", "ef", "rerank_ef",
                  "recall_1", "recall_10", "recall_100", "qps", "build_s", "mem_per_vec"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {csv_path}")

        analyze_efficiency(results)

        print_summary(results)
        print_best_per_dataset(results)

        plot_turboquant_comparison(args.output)
    else:
        assert True


if __name__ == "__main__":
    main()
