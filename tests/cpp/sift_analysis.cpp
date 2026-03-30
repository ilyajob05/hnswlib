/**
 * sift_analysis.cpp — Analyze BigANN SIFT vector distribution properties
 * to understand why TurboQuant (randomized Hadamard + scalar quantization)
 * achieves lower recall on SIFT than on synthetic clustered data.
 *
 * Build:
 *   g++ -O3 -std=c++11 -o sift_analysis sift_analysis.cpp
 * Run:
 *   ./sift_analysis <path_to_bigann_base.bvecs> [max_vectors]
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <chrono>
#include <cassert>

// ============================================================
// bvecs loader
// ============================================================
struct BvecsData {
    std::vector<std::vector<uint8_t>> raw;   // raw uint8 vectors
    std::vector<std::vector<float>>   fvecs; // float-converted vectors
    int dim;
    size_t n;
};

BvecsData load_bvecs(const char* path, size_t max_n = 0) {
    BvecsData data;
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot open %s\n", path);
        exit(1);
    }

    // Read first dimension
    int32_t dim;
    if (fread(&dim, sizeof(int32_t), 1, f) != 1) {
        fprintf(stderr, "ERROR: Cannot read dimension\n");
        exit(1);
    }
    fseek(f, 0, SEEK_SET);

    data.dim = dim;
    printf("Dimension: %d\n", dim);

    // Get file size
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);

    size_t record_size = sizeof(int32_t) + dim;
    size_t total_n = fsize / record_size;
    data.n = (max_n > 0 && max_n < total_n) ? max_n : total_n;

    printf("Total vectors in file: %zu\n", total_n);
    printf("Loading: %zu vectors\n", data.n);

    data.raw.resize(data.n);
    data.fvecs.resize(data.n);

    std::vector<uint8_t> buf(record_size);
    for (size_t i = 0; i < data.n; i++) {
        if (fread(buf.data(), 1, record_size, f) != record_size) {
            fprintf(stderr, "ERROR: Short read at vector %zu\n", i);
            data.n = i;
            break;
        }
        int32_t d;
        memcpy(&d, buf.data(), sizeof(int32_t));
        assert(d == dim);

        data.raw[i].assign(buf.begin() + 4, buf.end());
        data.fvecs[i].resize(dim);
        for (int j = 0; j < dim; j++) {
            data.fvecs[i][j] = static_cast<float>(buf[4 + j]);
        }
    }
    fclose(f);
    return data;
}

// ============================================================
// Statistics helpers
// ============================================================
struct Stats {
    double min_val, max_val, mean, stddev, median;
    double p5, p25, p75, p95; // percentiles
};

Stats compute_stats(std::vector<double>& v) {
    Stats s;
    if (v.empty()) {
        memset(&s, 0, sizeof(s));
        return s;
    }
    std::sort(v.begin(), v.end());
    s.min_val = v.front();
    s.max_val = v.back();
    double sum = 0, sum2 = 0;
    for (double x : v) { sum += x; sum2 += x * x; }
    s.mean = sum / v.size();
    s.stddev = sqrt(sum2 / v.size() - s.mean * s.mean);
    s.median = v[v.size() / 2];
    s.p5  = v[(size_t)(v.size() * 0.05)];
    s.p25 = v[(size_t)(v.size() * 0.25)];
    s.p75 = v[(size_t)(v.size() * 0.75)];
    s.p95 = v[(size_t)(v.size() * 0.95)];
    return s;
}

void print_stats(const char* name, Stats& s) {
    printf("  %s: min=%.4f p5=%.4f p25=%.4f median=%.4f mean=%.4f p75=%.4f p95=%.4f max=%.4f stddev=%.4f\n",
           name, s.min_val, s.p5, s.p25, s.median, s.mean, s.p75, s.p95, s.max_val, s.stddev);
}

// ============================================================
// Analysis routines
// ============================================================

void analyze_value_distribution(const BvecsData& data) {
    printf("\n========================================\n");
    printf("1. RAW UINT8 VALUE DISTRIBUTION\n");
    printf("========================================\n");

    uint64_t histogram[256] = {};
    uint64_t total_values = 0;

    for (size_t i = 0; i < data.n; i++) {
        for (int j = 0; j < data.dim; j++) {
            histogram[data.raw[i][j]]++;
            total_values++;
        }
    }

    printf("Total coordinate values: %llu\n", (unsigned long long)total_values);
    printf("Fraction that are 0: %.4f%% (%llu)\n",
           100.0 * histogram[0] / total_values, (unsigned long long)histogram[0]);

    // Print histogram in buckets of 16
    printf("\nHistogram (buckets of 16):\n");
    for (int b = 0; b < 16; b++) {
        uint64_t count = 0;
        for (int j = b * 16; j < (b + 1) * 16; j++) count += histogram[j];
        double pct = 100.0 * count / total_values;
        printf("  [%3d-%3d]: %10llu (%6.2f%%) ", b*16, b*16+15,
               (unsigned long long)count, pct);
        int bar = (int)(pct * 2);
        for (int k = 0; k < bar; k++) printf("#");
        printf("\n");
    }

    // Find top-10 most common values
    std::vector<std::pair<uint64_t, int>> sorted_hist;
    for (int i = 0; i < 256; i++) sorted_hist.push_back({histogram[i], i});
    std::sort(sorted_hist.rbegin(), sorted_hist.rend());
    printf("\nTop 10 most common values:\n");
    for (int i = 0; i < 10; i++) {
        printf("  value=%3d: count=%10llu (%.2f%%)\n",
               sorted_hist[i].second, (unsigned long long)sorted_hist[i].first,
               100.0 * sorted_hist[i].first / total_values);
    }

    // Mean and variance of raw values
    double sum = 0, sum2 = 0;
    for (int i = 0; i < 256; i++) {
        sum += (double)i * histogram[i];
        sum2 += (double)i * i * histogram[i];
    }
    double mean = sum / total_values;
    double var = sum2 / total_values - mean * mean;
    printf("\nOverall: mean=%.2f, stddev=%.2f, var=%.2f\n", mean, sqrt(var), var);

    // Kurtosis of raw coordinate values
    double m4 = 0;
    for (int i = 0; i < 256; i++) {
        double diff = i - mean;
        m4 += diff * diff * diff * diff * histogram[i];
    }
    m4 /= total_values;
    double kurtosis = m4 / (var * var);  // excess kurtosis = kurtosis - 3
    printf("Kurtosis: %.4f (excess kurtosis: %.4f; Gaussian=0)\n", kurtosis, kurtosis - 3.0);
}

void analyze_sparsity(const BvecsData& data) {
    printf("\n========================================\n");
    printf("2. SPARSITY (fraction of zero coordinates)\n");
    printf("========================================\n");

    std::vector<double> zero_fractions(data.n);
    for (size_t i = 0; i < data.n; i++) {
        int zeros = 0;
        for (int j = 0; j < data.dim; j++) {
            if (data.raw[i][j] == 0) zeros++;
        }
        zero_fractions[i] = (double)zeros / data.dim;
    }

    Stats s = compute_stats(zero_fractions);
    print_stats("Zero fraction per vector", s);

    // Histogram of zero fractions
    printf("\nDistribution of zero fractions:\n");
    int buckets[11] = {};
    for (double z : zero_fractions) {
        int b = std::min((int)(z * 10), 10);
        buckets[b]++;
    }
    for (int b = 0; b <= 10; b++) {
        printf("  [%.1f-%.1f): %d (%.2f%%)\n",
               b * 0.1, (b + 1) * 0.1,
               buckets[b], 100.0 * buckets[b] / data.n);
    }
}

void analyze_norms(const BvecsData& data) {
    printf("\n========================================\n");
    printf("3. L2 NORM DISTRIBUTION (float vectors)\n");
    printf("========================================\n");

    std::vector<double> norms(data.n);
    for (size_t i = 0; i < data.n; i++) {
        double sq = 0;
        for (int j = 0; j < data.dim; j++) {
            sq += data.fvecs[i][j] * data.fvecs[i][j];
        }
        norms[i] = sqrt(sq);
    }

    Stats s = compute_stats(norms);
    print_stats("L2 norms", s);
    printf("  Coefficient of variation (std/mean): %.4f\n", s.stddev / s.mean);

    printf("\n  For comparison: if values were uniform in [0,255],\n");
    printf("  expected norm = sqrt(128 * (255^2/3)) = %.2f\n",
           sqrt(128.0 * 255.0 * 255.0 / 3.0));
}

void analyze_normalized(const BvecsData& data) {
    printf("\n========================================\n");
    printf("4. DISTRIBUTION AFTER L2 NORMALIZATION\n");
    printf("========================================\n");

    int dim = data.dim;

    // Compute per-coordinate statistics after normalization
    std::vector<double> coord_means(dim, 0);
    std::vector<double> coord_vars(dim, 0);
    std::vector<double> coord_kurtosis(dim, 0);

    // First pass: means
    for (size_t i = 0; i < data.n; i++) {
        double norm = 0;
        for (int j = 0; j < dim; j++) norm += data.fvecs[i][j] * data.fvecs[i][j];
        norm = sqrt(norm);
        if (norm < 1e-10) continue;
        for (int j = 0; j < dim; j++) {
            coord_means[j] += data.fvecs[i][j] / norm;
        }
    }
    for (int j = 0; j < dim; j++) coord_means[j] /= data.n;

    // Second pass: variance and 4th moment
    for (size_t i = 0; i < data.n; i++) {
        double norm = 0;
        for (int j = 0; j < dim; j++) norm += data.fvecs[i][j] * data.fvecs[i][j];
        norm = sqrt(norm);
        if (norm < 1e-10) continue;
        for (int j = 0; j < dim; j++) {
            double val = data.fvecs[i][j] / norm;
            double diff = val - coord_means[j];
            coord_vars[j] += diff * diff;
            coord_kurtosis[j] += diff * diff * diff * diff;
        }
    }
    for (int j = 0; j < dim; j++) {
        coord_vars[j] /= data.n;
        coord_kurtosis[j] /= data.n;
    }

    // Statistics of coordinate variances
    double mean_var = 0, min_var = 1e30, max_var = 0;
    for (int j = 0; j < dim; j++) {
        mean_var += coord_vars[j];
        if (coord_vars[j] < min_var) min_var = coord_vars[j];
        if (coord_vars[j] > max_var) max_var = coord_vars[j];
    }
    mean_var /= dim;

    printf("Per-coordinate variance after normalization:\n");
    printf("  min=%.6f, mean=%.6f, max=%.6f\n", min_var, mean_var, max_var);
    printf("  For uniform on unit sphere: expected var = 1/dim = %.6f\n", 1.0 / dim);
    printf("  Ratio actual/expected: %.4f\n", mean_var / (1.0 / dim));

    // Variance of the variances (how uneven are coordinates?)
    double var_of_var = 0;
    for (int j = 0; j < dim; j++) {
        double diff = coord_vars[j] - mean_var;
        var_of_var += diff * diff;
    }
    var_of_var /= dim;
    printf("  Std of per-coordinate variances: %.6f (ratio to mean: %.4f)\n",
           sqrt(var_of_var), sqrt(var_of_var) / mean_var);

    // Per-coordinate excess kurtosis
    std::vector<double> excess_kurt(dim);
    for (int j = 0; j < dim; j++) {
        if (coord_vars[j] > 1e-15) {
            excess_kurt[j] = coord_kurtosis[j] / (coord_vars[j] * coord_vars[j]) - 3.0;
        } else {
            excess_kurt[j] = 0;
        }
    }
    Stats ks = compute_stats(excess_kurt);
    printf("\nPer-coordinate excess kurtosis (after normalization):\n");
    print_stats("excess kurtosis", ks);

    // Print coordinate means — are some coordinates consistently larger?
    printf("\nPer-coordinate means after normalization (first 32 of %d):\n  ", dim);
    for (int j = 0; j < std::min(dim, 32); j++) {
        printf("%.4f ", coord_means[j]);
        if ((j + 1) % 8 == 0) printf("\n  ");
    }

    std::vector<double> cm(coord_means.begin(), coord_means.end());
    Stats ms = compute_stats(cm);
    printf("\n");
    print_stats("coord means", ms);
    printf("  For unit sphere: all means should be ~0\n");

    // Check how non-negative the data is
    printf("\nAll coordinates are non-negative (SIFT uses uint8): YES\n");
    printf("This means normalized vectors are confined to the positive orthant.\n");
    printf("They occupy only 1/2^128 of the sphere surface!\n");
}

void analyze_kurtosis_detail(const BvecsData& data) {
    printf("\n========================================\n");
    printf("5. KURTOSIS / GAUSSIANITY ANALYSIS\n");
    printf("========================================\n");

    int dim = data.dim;

    // Compute per-coordinate stats (raw float values)
    std::vector<double> means(dim, 0);
    for (size_t i = 0; i < data.n; i++)
        for (int j = 0; j < dim; j++)
            means[j] += data.fvecs[i][j];
    for (int j = 0; j < dim; j++) means[j] /= data.n;

    std::vector<double> vars(dim, 0);
    std::vector<double> m4s(dim, 0);
    std::vector<double> skews_num(dim, 0);
    for (size_t i = 0; i < data.n; i++) {
        for (int j = 0; j < dim; j++) {
            double d = data.fvecs[i][j] - means[j];
            vars[j] += d * d;
            skews_num[j] += d * d * d;
            m4s[j] += d * d * d * d;
        }
    }

    printf("Per-coordinate statistics (raw float, first 16 dims):\n");
    printf("  dim |   mean   |  stddev  | skewness | ex.kurt\n");
    printf("  ----|----------|----------|----------|--------\n");

    std::vector<double> all_excess_kurt(dim);
    std::vector<double> all_skewness(dim);
    for (int j = 0; j < dim; j++) {
        vars[j] /= data.n;
        skews_num[j] /= data.n;
        m4s[j] /= data.n;
        double sd = sqrt(vars[j]);
        double skew = (sd > 1e-10) ? skews_num[j] / (sd * sd * sd) : 0;
        double kurt = (vars[j] > 1e-10) ? m4s[j] / (vars[j] * vars[j]) - 3.0 : 0;
        all_excess_kurt[j] = kurt;
        all_skewness[j] = skew;
        if (j < 16) {
            printf("  %3d | %8.2f | %8.2f | %8.4f | %7.4f\n",
                   j, means[j], sd, skew, kurt);
        }
    }

    Stats ks = compute_stats(all_excess_kurt);
    Stats ss = compute_stats(all_skewness);
    printf("\nSummary across all %d coordinates:\n", dim);
    print_stats("excess kurtosis", ks);
    print_stats("skewness", ss);
    printf("\n  Interpretation:\n");
    printf("  - Gaussian: excess kurtosis=0, skewness=0\n");
    printf("  - Positive skewness: right-tailed (common for non-negative data)\n");
    printf("  - Negative kurtosis: lighter tails than Gaussian (platykurtic)\n");
    printf("  - Positive kurtosis: heavier tails (leptokurtic)\n");
}

void analyze_pairwise_distances(const BvecsData& data) {
    printf("\n========================================\n");
    printf("6. PAIRWISE L2 DISTANCE DISTRIBUTION\n");
    printf("========================================\n");

    size_t n_pairs = 10000;
    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> dist(0, data.n - 1);

    int dim = data.dim;

    // Raw L2 distances
    std::vector<double> raw_dists(n_pairs);
    // Normalized L2 distances (on unit sphere)
    std::vector<double> norm_dists(n_pairs);

    for (size_t p = 0; p < n_pairs; p++) {
        size_t i = dist(rng), j = dist(rng);
        while (j == i) j = dist(rng);

        double raw_sq = 0;
        double norm_i = 0, norm_j = 0;
        for (int k = 0; k < dim; k++) {
            double diff = data.fvecs[i][k] - data.fvecs[j][k];
            raw_sq += diff * diff;
            norm_i += data.fvecs[i][k] * data.fvecs[i][k];
            norm_j += data.fvecs[j][k] * data.fvecs[j][k];
        }
        raw_dists[p] = sqrt(raw_sq);

        double ni = sqrt(norm_i), nj = sqrt(norm_j);
        double norm_sq = 0;
        for (int k = 0; k < dim; k++) {
            double a = data.fvecs[i][k] / ni;
            double b = data.fvecs[j][k] / nj;
            norm_sq += (a - b) * (a - b);
        }
        norm_dists[p] = sqrt(norm_sq);
    }

    Stats rs = compute_stats(raw_dists);
    Stats ns = compute_stats(norm_dists);

    printf("Raw L2 distances (%zu random pairs):\n", n_pairs);
    print_stats("raw L2 dist", rs);
    printf("  Relative spread (std/mean): %.4f\n", rs.stddev / rs.mean);

    printf("\nNormalized L2 distances (unit vectors):\n");
    print_stats("norm L2 dist", ns);
    printf("  Relative spread (std/mean): %.4f\n", ns.stddev / ns.mean);

    printf("\n========================================\n");
    printf("7. COMPARISON WITH RANDOM UNIT SPHERE\n");
    printf("========================================\n");

    // For random unit vectors in dim d, E[||x-y||^2] = 2, so E[||x-y||] ~ sqrt(2)
    // More precisely, ||x-y||^2 = 2 - 2<x,y>, and <x,y> ~ N(0, 1/d) for large d
    // So Var(||x-y||^2) = 4*Var(<x,y>) = 4/d
    // E[||x-y||^2] = 2, Var(||x-y||^2) = 4/d
    double expected_sq_dist = 2.0;
    double var_sq_dist = 4.0 / dim;
    double expected_dist = sqrt(expected_sq_dist); // approx

    printf("Random unit vectors in dim=%d:\n", dim);
    printf("  E[||x-y||^2] = 2.0\n");
    printf("  E[||x-y||]   ~ sqrt(2) = %.4f\n", expected_dist);
    printf("  Var(||x-y||^2) = 4/d = %.6f\n", var_sq_dist);
    printf("  std(||x-y||^2) = 2/sqrt(d) = %.6f\n", 2.0 / sqrt(dim));
    printf("  Relative spread: std(||x-y||) / E[||x-y||] ~ 1/sqrt(d) = %.4f\n",
           1.0 / sqrt(dim));

    // For SIFT normalized
    double mean_sq = 0, var_sq = 0;
    for (double d : norm_dists) mean_sq += d * d;
    mean_sq /= norm_dists.size();
    for (double d : norm_dists) {
        double diff = d * d - mean_sq;
        var_sq += diff * diff;
    }
    var_sq /= norm_dists.size();

    printf("\nSIFT normalized vectors:\n");
    printf("  E[||x-y||]   = %.4f (vs sqrt(2)=%.4f)\n", ns.mean, expected_dist);
    printf("  E[||x-y||^2] = %.4f (vs 2.0)\n", mean_sq);
    printf("  Var(||x-y||^2) = %.6f (vs %.6f for random)\n", var_sq, var_sq_dist);
    printf("  std(||x-y||^2) = %.6f (vs %.6f for random)\n", sqrt(var_sq), 2.0/sqrt(dim));
    printf("  Relative spread: %.4f (vs %.4f for random)\n",
           ns.stddev / ns.mean, 1.0 / sqrt(dim));

    // Compute inner products between normalized vectors
    std::vector<double> ips(n_pairs);
    for (size_t p = 0; p < n_pairs; p++) {
        size_t i = dist(rng), j = dist(rng);
        while (j == i) j = dist(rng);
        double norm_i = 0, norm_j = 0, ip = 0;
        for (int k = 0; k < dim; k++) {
            norm_i += data.fvecs[i][k] * data.fvecs[i][k];
            norm_j += data.fvecs[j][k] * data.fvecs[j][k];
            ip += data.fvecs[i][k] * data.fvecs[j][k];
        }
        ips[p] = ip / (sqrt(norm_i) * sqrt(norm_j));
    }
    Stats ips_s = compute_stats(ips);
    printf("\nCosine similarity distribution (normalized inner products):\n");
    print_stats("cos sim", ips_s);
    printf("  For random unit sphere: mean ~0, std ~%.4f\n", 1.0 / sqrt(dim));
    printf("  SIFT vectors have HIGH positive cosine similarity (confined to positive orthant).\n");
    printf("  This means they are MUCH more concentrated than random unit vectors.\n");
}

void analyze_effective_dimension(const BvecsData& data) {
    printf("\n========================================\n");
    printf("8. EFFECTIVE DIMENSIONALITY & INTRINSIC STRUCTURE\n");
    printf("========================================\n");

    int dim = data.dim;
    size_t n = data.n;

    // Compute centroid
    std::vector<double> centroid(dim, 0);
    for (size_t i = 0; i < n; i++)
        for (int j = 0; j < dim; j++)
            centroid[j] += data.fvecs[i][j];
    for (int j = 0; j < dim; j++) centroid[j] /= n;

    // Compute variance per coordinate (centered)
    std::vector<double> coord_var(dim, 0);
    double total_var = 0;
    for (size_t i = 0; i < n; i++) {
        for (int j = 0; j < dim; j++) {
            double d = data.fvecs[i][j] - centroid[j];
            coord_var[j] += d * d;
        }
    }
    for (int j = 0; j < dim; j++) {
        coord_var[j] /= n;
        total_var += coord_var[j];
    }

    // Sort variances descending (approximate PCA eigenvalues for diagonal)
    std::vector<double> sorted_var(coord_var);
    std::sort(sorted_var.rbegin(), sorted_var.rend());

    printf("Total variance: %.2f\n", total_var);
    printf("Mean variance per coordinate: %.2f\n", total_var / dim);

    // Cumulative variance explained (axis-aligned, not PCA)
    printf("\nCumulative axis-aligned variance explained:\n");
    double cum = 0;
    int thresholds[] = {1, 5, 10, 20, 32, 64, 96, 128};
    int tidx = 0;
    for (int j = 0; j < dim && tidx < 8; j++) {
        cum += sorted_var[j];
        if (j + 1 == thresholds[tidx]) {
            printf("  Top %3d coords: %.2f%% of total variance\n",
                   j + 1, 100.0 * cum / total_var);
            tidx++;
        }
    }

    // Participation ratio: (sum lambda_i)^2 / sum lambda_i^2
    // This gives effective number of dimensions
    double sum_sq = 0;
    for (int j = 0; j < dim; j++) sum_sq += coord_var[j] * coord_var[j];
    double participation = (total_var * total_var) / sum_sq;
    printf("\nParticipation ratio (effective dimensionality): %.1f / %d\n",
           participation, dim);

    // Entropy of normalized variance (another measure)
    double entropy = 0;
    for (int j = 0; j < dim; j++) {
        double p = coord_var[j] / total_var;
        if (p > 1e-15) entropy -= p * log(p);
    }
    double max_entropy = log((double)dim);
    printf("Variance entropy: %.4f / %.4f (ratio: %.4f)\n",
           entropy, max_entropy, entropy / max_entropy);
}

void analyze_quantization_impact(const BvecsData& data) {
    printf("\n========================================\n");
    printf("9. QUANTIZATION ERROR ANALYSIS\n");
    printf("========================================\n");

    int dim = data.dim;
    size_t n = std::min(data.n, (size_t)50000);

    // Simulate scalar quantization to 4 bits (like TurboQuant)
    // For each vector after normalization, compute quantization error
    // Compare: (a) quantizing raw coordinates vs (b) quantizing normalized coordinates

    // Scenario A: quantize normalized vectors directly (no rotation)
    // Scenario B: apply random rotation then quantize (simulates Hadamard)

    // For each vector, normalize, then scalar-quantize to 4 bits (16 levels)
    // using min-max scaling per vector
    int n_bits = 4;
    int n_levels = (1 << n_bits);  // 16

    double total_err_direct = 0;
    double total_err_global = 0;
    int count = 0;

    // First compute global min/max after normalization
    double global_min = 1e30, global_max = -1e30;
    for (size_t i = 0; i < n; i++) {
        double norm = 0;
        for (int j = 0; j < dim; j++) norm += data.fvecs[i][j] * data.fvecs[i][j];
        norm = sqrt(norm);
        if (norm < 1e-10) continue;
        for (int j = 0; j < dim; j++) {
            double v = data.fvecs[i][j] / norm;
            if (v < global_min) global_min = v;
            if (v > global_max) global_max = v;
        }
    }

    printf("Normalized coordinate range: [%.6f, %.6f]\n", global_min, global_max);
    printf("Note: all values >= 0 since raw SIFT values are uint8 >= 0\n");

    // For direct per-vector quantization
    for (size_t i = 0; i < n; i++) {
        double norm = 0;
        for (int j = 0; j < dim; j++) norm += data.fvecs[i][j] * data.fvecs[i][j];
        norm = sqrt(norm);
        if (norm < 1e-10) continue;

        std::vector<double> nv(dim);
        double vmin = 1e30, vmax = -1e30;
        for (int j = 0; j < dim; j++) {
            nv[j] = data.fvecs[i][j] / norm;
            if (nv[j] < vmin) vmin = nv[j];
            if (nv[j] > vmax) vmax = nv[j];
        }

        // Per-vector min-max quantization
        double range = vmax - vmin;
        if (range < 1e-15) range = 1.0;
        double step = range / (n_levels - 1);

        double err_sq = 0;
        for (int j = 0; j < dim; j++) {
            int q = (int)round((nv[j] - vmin) / step);
            if (q < 0) q = 0;
            if (q >= n_levels) q = n_levels - 1;
            double recon = vmin + q * step;
            double e = nv[j] - recon;
            err_sq += e * e;
        }
        total_err_direct += err_sq;

        // Global min-max quantization
        double gstep = (global_max - global_min) / (n_levels - 1);
        double err_sq_g = 0;
        for (int j = 0; j < dim; j++) {
            int q = (int)round((nv[j] - global_min) / gstep);
            if (q < 0) q = 0;
            if (q >= n_levels) q = n_levels - 1;
            double recon = global_min + q * gstep;
            double e = nv[j] - recon;
            err_sq_g += e * e;
        }
        total_err_global += err_sq_g;

        count++;
    }

    double mse_direct = total_err_direct / count;
    double mse_global = total_err_global / count;
    printf("\n4-bit scalar quantization (16 levels):\n");
    printf("  Per-vector min-max: MSE = %.8f, RMSE = %.6f\n", mse_direct, sqrt(mse_direct));
    printf("  Global min-max:     MSE = %.8f, RMSE = %.6f\n", mse_global, sqrt(mse_global));

    // Compare: average pairwise distance squared
    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> udist(0, n - 1);
    double avg_dist_sq = 0;
    int npairs = 10000;
    for (int p = 0; p < npairs; p++) {
        size_t a = udist(rng), b = udist(rng);
        while (b == a) b = udist(rng);
        double ni = 0, nj = 0;
        for (int k = 0; k < dim; k++) {
            ni += data.fvecs[a][k] * data.fvecs[a][k];
            nj += data.fvecs[b][k] * data.fvecs[b][k];
        }
        ni = sqrt(ni); nj = sqrt(nj);
        double dsq = 0;
        for (int k = 0; k < dim; k++) {
            double d = data.fvecs[a][k]/ni - data.fvecs[b][k]/nj;
            dsq += d * d;
        }
        avg_dist_sq += dsq;
    }
    avg_dist_sq /= npairs;
    printf("\n  Average pairwise ||x-y||^2 (normalized): %.6f\n", avg_dist_sq);
    printf("  Quantization error / pairwise distance: %.4f (per-vec) / %.4f (global)\n",
           mse_direct / avg_dist_sq, mse_global / avg_dist_sq);
    printf("  => Quantization noise is %.1f%% (per-vec) / %.1f%% (global) of signal\n",
           100.0 * mse_direct / avg_dist_sq, 100.0 * mse_global / avg_dist_sq);
}

void analyze_dynamic_range(const BvecsData& data) {
    printf("\n========================================\n");
    printf("10. DYNAMIC RANGE & COORDINATE CONCENTRATION\n");
    printf("========================================\n");

    int dim = data.dim;
    size_t n = std::min(data.n, (size_t)50000);

    // After normalization, what fraction of the L2 norm is carried by
    // the top-K coordinates? (measures how "spiky" vs "flat" vectors are)
    double avg_top1 = 0, avg_top5 = 0, avg_top10 = 0, avg_top20 = 0;
    double avg_max_coord = 0;
    std::vector<double> max_coords(n);

    for (size_t i = 0; i < n; i++) {
        double norm = 0;
        for (int j = 0; j < dim; j++) norm += data.fvecs[i][j] * data.fvecs[i][j];
        norm = sqrt(norm);
        if (norm < 1e-10) continue;

        std::vector<double> sq(dim);
        for (int j = 0; j < dim; j++) {
            double v = data.fvecs[i][j] / norm;
            sq[j] = v * v;
        }
        std::sort(sq.rbegin(), sq.rend());

        max_coords[i] = sqrt(sq[0]);

        double sum1 = 0, sum5 = 0, sum10 = 0, sum20 = 0;
        for (int j = 0; j < dim; j++) {
            if (j < 1) sum1 += sq[j];
            if (j < 5) sum5 += sq[j];
            if (j < 10) sum10 += sq[j];
            if (j < 20) sum20 += sq[j];
        }
        avg_top1 += sum1;
        avg_top5 += sum5;
        avg_top10 += sum10;
        avg_top20 += sum20;
    }
    avg_top1 /= n; avg_top5 /= n; avg_top10 /= n; avg_top20 /= n;

    printf("Fraction of ||x||^2 carried by top-K coordinates (after normalization):\n");
    printf("  Top  1: %.4f (uniform: %.4f)\n", avg_top1, 1.0/dim);
    printf("  Top  5: %.4f (uniform: %.4f)\n", avg_top5, 5.0/dim);
    printf("  Top 10: %.4f (uniform: %.4f)\n", avg_top10, 10.0/dim);
    printf("  Top 20: %.4f (uniform: %.4f)\n", avg_top20, 20.0/dim);

    Stats ms = compute_stats(max_coords);
    printf("\nMax coordinate value (per normalized vector):\n");
    print_stats("max coord", ms);
    printf("  For uniform on sphere: expected max ~ sqrt(2*ln(d)/d) = %.4f\n",
           sqrt(2.0 * log(dim) / dim));
    printf("  Much larger max coord => spikier vectors => harder to quantize uniformly\n");
}

// ============================================================
// Main
// ============================================================
int main(int argc, char** argv) {
    const char* path = "../bigann/bigann_base.bvecs";
    size_t max_n = 0; // 0 = load all

    if (argc > 1) path = argv[1];
    if (argc > 2) max_n = atol(argv[2]);

    printf("==============================================\n");
    printf("  SIFT DATA DISTRIBUTION ANALYSIS\n");
    printf("  for TurboQuant recall investigation\n");
    printf("==============================================\n");
    printf("File: %s\n", path);

    auto t0 = std::chrono::high_resolution_clock::now();
    BvecsData data = load_bvecs(path, max_n);
    auto t1 = std::chrono::high_resolution_clock::now();
    double load_time = std::chrono::duration<double>(t1 - t0).count();
    printf("Load time: %.2f seconds\n", load_time);

    analyze_value_distribution(data);
    analyze_sparsity(data);
    analyze_norms(data);
    analyze_normalized(data);
    analyze_kurtosis_detail(data);
    analyze_pairwise_distances(data);
    analyze_effective_dimension(data);
    analyze_quantization_impact(data);
    analyze_dynamic_range(data);

    printf("\n========================================\n");
    printf("SUMMARY: WHY TURBOQUANT STRUGGLES ON SIFT\n");
    printf("========================================\n");
    printf("Key observations to check:\n");
    printf("1. SIFT vectors are non-negative (uint8) => after normalization,\n");
    printf("   vectors are confined to the positive orthant of the unit sphere.\n");
    printf("2. Coordinates are highly non-Gaussian (bounded, skewed, sparse).\n");
    printf("3. Randomized Hadamard transform assumes the input is roughly\n");
    printf("   spherically symmetric. Non-negative data violates this.\n");
    printf("4. High cosine similarity between pairs => small angular distances\n");
    printf("   => quantization noise may overwhelm the signal.\n");
    printf("5. If variance is concentrated in few coordinates, Hadamard helps\n");
    printf("   by spreading it, but the non-negativity/skewness remains.\n");
    printf("\n");

    return 0;
}
