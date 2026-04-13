"""
TurboQuant Distance Quality Diagnostics

Tests:
  1. Distance correlation (Pearson, Spearman) for distBuild, distBuildCorrected, distSearch vs L2
  2. Top-K rank stability
  3. Recall comparison: TQ-graph vs TQ-graph-corrected vs L2-graph, with varying M/ef_construction
  4. Lloyd-Max table validation & post-RHT distribution analysis
  5. Bit-width sweep (4, 8)
"""

import time
import numpy as np
from scipy import stats
import hnswlib

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
DIM = 128           # power of 2, smaller = faster diagnostics
N = 20_000          # dataset size
N_QUERIES = 500
N_PAIRS = 10_000    # for distance correlation
K = 10
SEED = 42
NUM_THREADS = 8

np.random.seed(SEED)


def generate_data():
    data = np.float32(np.random.randn(N, DIM))
    queries = np.float32(np.random.randn(N_QUERIES, DIM))
    return data, queries


def compute_gt(data, queries, k):
    bf = hnswlib.BFIndex(space="l2", dim=DIM)
    bf.init_index(max_elements=len(data))
    bf.add_items(data)
    labels, _ = bf.knn_query(queries, k=k, num_threads=NUM_THREADS)
    del bf
    return labels


def recall_at_k(labels, gt_labels, k):
    return np.mean([
        len(set(labels[i]) & set(gt_labels[i])) / k
        for i in range(len(labels))
    ])


# =========================================================================
# Test 1: Distance Correlation
# =========================================================================
def test_distance_correlation(data, bits):
    print(f"\n{'='*60}")
    print(f"Test 1: Distance Correlation (bits={bits}, dim={DIM}, N={N})")
    print(f"{'='*60}")

    idx = hnswlib.TQIndex(dim=DIM, bits_per_coord=bits)
    idx.build(data, M=16, ef_construction=200, num_threads=NUM_THREADS)

    pairs = np.random.randint(0, N, size=(N_PAIRS, 2)).astype(np.int64)
    mask = pairs[:, 0] != pairs[:, 1]
    pairs = pairs[mask]

    t0 = time.time()
    dists = idx.compute_distances(pairs, data)
    dt = time.time() - t0
    print(f"  Computed {len(pairs)} distances in {dt:.2f}s")

    d_l2 = dists["l2"]
    d_build = dists["dist_build"]
    d_build_corr = dists["dist_build_corrected"]
    d_search = dists["dist_search"]

    for name, d_tq in [("distBuild", d_build),
                        ("distBuildCorrected", d_build_corr),
                        ("distSearch", d_search)]:
        pearson_r, pearson_p = stats.pearsonr(d_l2, d_tq)
        spearman_r, spearman_p = stats.spearmanr(d_l2, d_tq)

        rel_err = np.abs(d_tq - d_l2) / np.maximum(d_l2, 1e-10)
        p50, p90, p99 = np.percentile(rel_err, [50, 90, 99])

        # Bias: mean(d_tq - d_l2) / mean(d_l2)
        bias = np.mean(d_tq - d_l2) / np.mean(d_l2)

        print(f"\n  L2 vs {name}:")
        print(f"    Pearson:  r={pearson_r:.4f}  (p={pearson_p:.2e})")
        print(f"    Spearman: r={spearman_r:.4f}  (p={spearman_p:.2e})")
        print(f"    Rel error: mean={rel_err.mean():.4f}  p50={p50:.4f}  p90={p90:.4f}  p99={p99:.4f}")
        print(f"    Bias (mean shift): {bias:+.4f}  ({bias*100:+.2f}%)")

    return idx


# =========================================================================
# Test 2: Top-K Rank Stability
# =========================================================================
def test_rank_stability(data, queries, gt_labels, bits):
    print(f"\n{'='*60}")
    print(f"Test 2: Top-K Rank Stability (bits={bits})")
    print(f"{'='*60}")

    idx = hnswlib.TQIndex(dim=DIM, bits_per_coord=bits)
    idx.build(data, M=16, ef_construction=200, num_threads=NUM_THREADS)

    idx.ef = K * 4
    labels_tq, _ = idx.knn_query(queries, k=K * 2, num_threads=NUM_THREADS)

    stability = np.mean([
        len(set(gt_labels[i][:K]) & set(labels_tq[i][:K*2])) / K
        for i in range(N_QUERIES)
    ])
    print(f"  Top-{K} in Top-{K*2} stability: {stability:.4f}")

    labels_tq_k, _ = idx.knn_query(queries, k=K, num_threads=NUM_THREADS)
    rec = recall_at_k(labels_tq_k, gt_labels, K)
    print(f"  Recall@{K} (ef={K*4}): {rec:.4f}")


# =========================================================================
# Test 3: Recall comparison — build modes and parameters
# =========================================================================
def test_recall_comparison(data, queries, gt_labels, bits):
    print(f"\n{'='*60}")
    print(f"Test 3: Recall Comparison — Build Modes (bits={bits})")
    print(f"{'='*60}")

    configs = [
        ("TQ-graph (plain)",     16, 200, False, [64, 128, 256]),
        ("TQ-graph (corrected)", 16, 200, True,  [64, 128, 256]),
        ("TQ-graph (plain M=32)",     32, 400, False, [128, 256, 512]),
        ("TQ-graph (corrected M=32)", 32, 400, True,  [128, 256, 512]),
    ]

    hdr = f"  {'Config':<30} {'ef':>5} {'R@1':>8} {'R@10':>8} {'QPS':>10} {'build_s':>8}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for label, M, ef_c, use_corr, ef_list in configs:
        t0 = time.time()
        idx = hnswlib.TQIndex(dim=DIM, bits_per_coord=bits)
        idx.build(data, M=M, ef_construction=ef_c,
                  num_threads=NUM_THREADS, use_corrected_build=use_corr)
        build_time = time.time() - t0

        for ef in ef_list:
            idx.ef = ef
            t0 = time.time()
            labels, _ = idx.knn_query(queries, k=K, num_threads=NUM_THREADS)
            dt = time.time() - t0
            labels1, _ = idx.knn_query(queries, k=1, num_threads=NUM_THREADS)

            r1 = recall_at_k(labels1, gt_labels, 1)
            r10 = recall_at_k(labels, gt_labels, K)
            qps = N_QUERIES / dt

            print(f"  {label:<30} {ef:>5} {r1:>8.4f} {r10:>8.4f} {qps:>10.0f} {build_time:>8.2f}")

    # L2-graph baselines
    print(f"\n  --- L2-graph baselines ---")
    idx_l2g = hnswlib.TQIndex(dim=DIM, bits_per_coord=bits)
    t0 = time.time()
    idx_l2g.build_from_l2(data, M=16, ef_construction=200, keep_raw=True, num_threads=NUM_THREADS)
    build_time = time.time() - t0

    for ef, rerank_ef in [(64, 128), (128, 256)]:
        idx_l2g.ef = ef
        t0 = time.time()
        labels, _ = idx_l2g.knn_query_rerank(queries, k=K, rerank_ef=rerank_ef, num_threads=NUM_THREADS)
        dt = time.time() - t0
        r10 = recall_at_k(labels, gt_labels, K)
        qps = N_QUERIES / dt
        print(f"  {'L2graph+rerank':<30} {ef:>5} {'':>8} {r10:>8.4f} {qps:>10.0f} {build_time:>8.2f}")

    for ef in [64, 128, 256]:
        idx_l2g.ef = ef
        t0 = time.time()
        labels, _ = idx_l2g.knn_query(queries, k=K, num_threads=NUM_THREADS)
        dt = time.time() - t0
        r10 = recall_at_k(labels, gt_labels, K)
        qps = N_QUERIES / dt
        print(f"  {'L2graph+TQsearch':<30} {ef:>5} {'':>8} {r10:>8.4f} {qps:>10.0f} {build_time:>8.2f}")


# =========================================================================
# Test 4: Lloyd-Max Table & Post-RHT Distribution Analysis
# =========================================================================
def test_lloyd_max_and_rht(data, bits):
    print(f"\n{'='*60}")
    print(f"Test 4: Lloyd-Max Table & Post-RHT Distribution (bits={bits})")
    print(f"{'='*60}")

    idx = hnswlib.TQIndex(dim=DIM, bits_per_coord=bits)
    idx.build(data[:1000], M=4, ef_construction=50, num_threads=1)  # minimal build

    # Get Lloyd-Max table
    table = idx.get_lloyd_max_table()
    centroids = np.array(table["centroids"])
    boundaries = np.array(table["boundaries"])
    bits_sq = table["bits_sq"]
    num_levels = table["num_levels"]

    print(f"\n  Lloyd-Max table (SQ bits={bits_sq}, levels={num_levels}):")
    print(f"    Centroids:  {centroids}")
    print(f"    Boundaries: {boundaries}")

    # Verify symmetry
    mid = num_levels // 2
    sym_err = np.max(np.abs(centroids[:mid] + centroids[num_levels-1:mid-1:-1]))
    print(f"    Symmetry error: {sym_err:.2e} (should be ~0)")

    # Verify centroids are conditional means of N(0,1)
    from scipy.stats import norm
    print(f"\n  Verifying centroids against theoretical N(0,1) conditional means:")
    all_bounds = np.concatenate([[-np.inf], boundaries, [np.inf]])
    for i in range(num_levels):
        lo, hi = all_bounds[i], all_bounds[i+1]
        # E[X | lo < X < hi] for N(0,1)
        denom = norm.cdf(hi) - norm.cdf(lo)
        if denom > 1e-15:
            theoretical = (norm.pdf(lo) - norm.pdf(hi)) / denom
        else:
            theoretical = 0.5 * (lo + hi)
        err = abs(centroids[i] - theoretical)
        if err > 1e-4:
            print(f"    Level {i}: centroid={centroids[i]:.6f} theory={theoretical:.6f} ERR={err:.6f}")
    print(f"    (only mismatches > 1e-4 shown)")

    # Get post-RHT coordinates
    rht_coords = idx.get_post_rht_coords(data, max_vectors=5000)
    all_coords = rht_coords.flatten()

    print(f"\n  Post-RHT distribution (N={len(rht_coords)} vectors, {len(all_coords)} coords):")
    print(f"    mean:    {all_coords.mean():.6f}  (expect ~0)")
    print(f"    std:     {all_coords.std():.6f}   (expect ~1)")
    print(f"    skewness: {stats.skew(all_coords):.6f}  (expect ~0)")
    print(f"    kurtosis: {stats.kurtosis(all_coords):.6f}  (expect ~0 for Gaussian)")

    # Normality test (D'Agostino-Pearson)
    k2, p_normal = stats.normaltest(all_coords[:100000])  # limit sample for speed
    print(f"    Normality test: K²={k2:.2f}, p={p_normal:.2e}")
    if p_normal < 0.01:
        print(f"    WARNING: post-RHT distribution significantly deviates from N(0,1)")
    else:
        print(f"    OK: post-RHT distribution is consistent with N(0,1)")

    # Check quantization bin occupancy vs expected
    print(f"\n  Quantization bin occupancy (expected vs actual):")
    from scipy.stats import norm as norm_dist
    expected_probs = []
    for i in range(num_levels):
        lo, hi = all_bounds[i], all_bounds[i+1]
        expected_probs.append(norm_dist.cdf(hi) - norm_dist.cdf(lo))
    expected_probs = np.array(expected_probs)

    # Quantize all_coords manually
    bin_counts = np.zeros(num_levels)
    for i in range(num_levels - 1):
        if i == 0:
            mask = all_coords <= boundaries[0]
        else:
            mask = (all_coords > boundaries[i-1]) & (all_coords <= boundaries[i])
        bin_counts[i] = mask.sum()
    bin_counts[-1] = (all_coords > boundaries[-1]).sum()
    actual_probs = bin_counts / len(all_coords)

    hdr = f"    {'Bin':>4} {'Centroid':>10} {'Expected':>10} {'Actual':>10} {'Ratio':>8}"
    print(hdr)
    for i in range(num_levels):
        ratio = actual_probs[i] / expected_probs[i] if expected_probs[i] > 0 else 0
        print(f"    {i:>4} {centroids[i]:>10.4f} {expected_probs[i]:>10.4f} {actual_probs[i]:>10.4f} {ratio:>8.4f}")

    chi2, p_chi2 = stats.chisquare(bin_counts, f_exp=expected_probs * len(all_coords))
    print(f"    Chi² test: χ²={chi2:.2f}, p={p_chi2:.2e}")


# =========================================================================
# Main
# =========================================================================
def main():
    print(f"TurboQuant Diagnostics — dim={DIM}, N={N}, queries={N_QUERIES}")
    data, queries = generate_data()

    print("Computing ground truth...")
    gt_labels = compute_gt(data, queries, K)

    for bits in [4, 8]:
        test_distance_correlation(data, bits)
        test_rank_stability(data, queries, gt_labels, bits)
        test_recall_comparison(data, queries, gt_labels, bits)
        test_lloyd_max_and_rht(data, bits)


if __name__ == "__main__":
    main()
