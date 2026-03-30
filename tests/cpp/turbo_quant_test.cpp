/// turbo_quant_test.cpp — Correctness tests for TurboQuant Algorithm 2
/// Synthetic face-embedding-like data: d=128, L2-normalized vectors
///
/// Tests:
///   1. WHT orthogonality (energy preservation)
///   2. SQ distortion (MSE in rotated space)
///   3. Unbiasedness: E[⟨y, x̃⟩] = ⟨y, x⟩
///   4. Lossless identity: MSE + exact residual = exact IP
///   5. Asymmetric IP Pearson correlation
///   6. Recall@K — L2 ranking preservation
///   +  Memory footprint report (informational, no pass/fail)

#include "hnswlib/turbo_quant_space.h"

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <chrono>
#include <fstream>
#include <cstdio>

#ifdef __APPLE__
#include <mach/mach.h>
#endif

using namespace hnswlib::turboquant;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static float dot(const float* a, const float* b, size_t d) {
    float s = 0.0f;
    for (size_t i = 0; i < d; ++i) s += a[i] * b[i];
    return s;
}

static float l2_dist(const float* a, const float* b, size_t d) {
    float s = 0.0f;
    for (size_t i = 0; i < d; ++i) {
        float diff = a[i] - b[i];
        s += diff * diff;
    }
    return s;
}

static void normalize(float* v, size_t d) {
    float n = std::sqrt(dot(v, v, d));
    if (n > 1e-10f) {
        for (size_t i = 0; i < d; ++i) v[i] /= n;
    }
}

/// Generates clustered unit vectors simulating face descriptors.
static std::vector<std::vector<float>>
generateFaceEmbeddings(size_t n, size_t d, size_t num_clusters = 50,
                       uint64_t seed = 12345) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> gauss(0.0f, 1.0f);

    std::vector<std::vector<float>> centers(num_clusters, std::vector<float>(d));
    for (auto& c : centers) {
        for (auto& v : c) v = gauss(rng);
        normalize(c.data(), d);
    }

    std::uniform_int_distribution<size_t> cluster_dist(0, num_clusters - 1);
    std::vector<std::vector<float>> embeddings(n, std::vector<float>(d));
    for (size_t i = 0; i < n; ++i) {
        size_t cluster = cluster_dist(rng);
        for (size_t j = 0; j < d; ++j) {
            embeddings[i][j] = centers[cluster][j] + gauss(rng) * 0.3f;
        }
        normalize(embeddings[i].data(), d);
    }
    return embeddings;
}

// ---------------------------------------------------------------------------
// Test 1: WHT energy preservation
// ---------------------------------------------------------------------------
bool test_wht_energy() {
    std::cout << "=== Test 1: WHT energy preservation ===" << std::endl;

    constexpr size_t D = 128;
    std::mt19937_64 rng(42);
    std::normal_distribution<float> gauss(0.0f, 1.0f);

    uint64_t rot_seed = 42;
    auto rot_signs = generateSigns(D, rot_seed);
    float max_err = 0.0f;
    for (int trial = 0; trial < 100; ++trial) {
        std::vector<float> v(D);
        for (auto& x : v) x = gauss(rng);
        float energy_before = dot(v.data(), v.data(), D);

        std::vector<float> rotated(v);
        randomizedHadamard(rotated.data(), rot_signs.data(), D);
        float energy_after = dot(rotated.data(), rotated.data(), D);

        float rel_err = std::abs(energy_after - energy_before) / energy_before;
        max_err = std::max(max_err, rel_err);
    }

    constexpr float THRESHOLD = 1e-5f;
    bool pass = max_err < THRESHOLD;
    std::cout << "  Max relative energy error: " << std::scientific << max_err;
    if (!pass) {
        std::cout << "  FAIL (expected < " << THRESHOLD << ")";
    } else {
        std::cout << "  PASS";
    }
    std::cout << std::endl;
    return pass;
}

// ---------------------------------------------------------------------------
// Test 2: SQ distortion (MSE stage only, b-1 bits)
// ---------------------------------------------------------------------------
bool test_sq_distortion() {
    std::cout << "=== Test 2: SQ distortion (3-bit MSE stage) ===" << std::endl;

    constexpr size_t D = 128;
    constexpr size_t N = 1000;

    auto embeddings = generateFaceEmbeddings(N, D);
    uint64_t rot_seed = 42;
    uint64_t qjl_seed = 137;
    auto rot_signs = generateSigns(D, rot_seed);
    const TurboQuantEncoder enc(D, /*bits_per_coord=*/4, rot_seed, qjl_seed);  // 3-bit MSE + 1-bit QJL

    double total_mse = 0.0;
    for (size_t i = 0; i < N; ++i) {
        auto code = enc.encode(embeddings[i].data());

        // Reconstruct MSE part in rotated space
        std::vector<float> rotated(embeddings[i]);
        float inv_norm = 1.0f / code.norm_;
        for (size_t j = 0; j < D; ++j) rotated[j] *= inv_norm;
        randomizedHadamard(rotated.data(), rot_signs.data(), D);

        std::vector<float> recon(D);
        code.dequantizeBatch(recon.data(), D);
        float mse = 0.0f;
        for (size_t j = 0; j < D; ++j) {
            float diff = rotated[j] - recon[j];
            mse += diff * diff;
        }
        mse /= static_cast<float>(D);
        total_mse += mse;
    }
    double avg_mse = total_mse / N;

    // 3-bit Lloyd-Max on Gaussian: expected MSE ≈ 0.03 per paper
    constexpr double THRESHOLD = 0.05;
    bool pass = avg_mse < THRESHOLD;
    std::cout << "  Average MSE: " << std::fixed << std::setprecision(6)
              << avg_mse;
    if (!pass) {
        std::cout << "  FAIL (expected < " << THRESHOLD << ")";
    } else {
        std::cout << "  PASS";
    }
    std::cout << std::endl;
    return pass;
}

// ---------------------------------------------------------------------------
// Test 3: Unbiasedness — E[⟨y, x̃⟩] = ⟨y, x⟩
// Uses multiple random seeds to average over QJL randomness
// ---------------------------------------------------------------------------
bool test_unbiasedness() {
    std::cout << "=== Test 3: Unbiasedness of IP estimator ===" << std::endl;

    constexpr size_t D = 128;
    constexpr size_t NUM_PAIRS = 50;
    constexpr size_t NUM_SEEDS = 500;

    auto embeddings = generateFaceEmbeddings(NUM_PAIRS * 2, D);

    double total_bias = 0.0;
    double max_bias = 0.0;

    for (size_t p = 0; p < NUM_PAIRS; ++p) {
        const float* x = embeddings[2 * p].data();
        const float* y = embeddings[2 * p + 1].data();

        float exact_ip = dot(y, x, D);

        // Average over multiple QJL seeds to estimate E[⟨y, x̃⟩]
        double avg_approx = 0.0;
        for (size_t s = 0; s < NUM_SEEDS; ++s) {
            TurboQuantEncoder enc(D, 4, /*rot_seed=*/42,
                                  /*qjl_seed=*/1000 + s);
            auto code = enc.encode(x);
            float approx = enc.asymmetricInnerProduct(y, code);
            avg_approx += approx;
        }
        avg_approx /= NUM_SEEDS;

        double bias = std::abs(avg_approx - exact_ip);
        total_bias += bias;
        max_bias = std::max(max_bias, bias);
    }

    double avg_bias = total_bias / NUM_PAIRS;

    // Unbiased: average bias should be small relative to typical IP magnitude
    // For unit vectors with d=128, typical |IP| ≈ 0.1-0.3
    constexpr double THRESHOLD = 0.005;
    bool pass = avg_bias < THRESHOLD;
    std::cout << "  Average |bias| over " << NUM_SEEDS << " seeds: "
              << std::fixed << std::setprecision(6) << avg_bias;
    if (!pass) {
        std::cout << "  FAIL (expected < " << THRESHOLD << ")";
    } else {
        std::cout << "  PASS";
    }
    std::cout << std::endl;
    std::cout << "  Max |bias|: " << std::setprecision(6) << max_bias
              << std::endl;
    return pass;
}

// ---------------------------------------------------------------------------
// Test 3b: Lossless identity — MSE_IP + exact_residual_IP = exact_IP
// Deterministic check: no QJL randomness involved
// ---------------------------------------------------------------------------
bool test_lossless_identity() {
    std::cout << "=== Test 3b: MSE + residual = exact (lossless identity) ===" << std::endl;

    constexpr size_t D = 512;
    constexpr size_t N = 200000;

    auto embeddings = generateFaceEmbeddings(N, D);
    uint64_t rot_seed = 42;
    uint64_t qjl_seed = 137;
    auto rot_signs = generateSigns(D, rot_seed);

    double max_err = 0.0;
    const TurboQuantEncoder enc(D, 4, rot_seed, qjl_seed);

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = i + 1; j < std::min(N, i + 5); ++j) {
            const float* x = embeddings[i].data();
            const float* y = embeddings[j].data();
            float exact_ip = dot(x, y, D);

            auto code = enc.encode(x);

            // Compute MSE IP in rotated space
            float x_norm = code.norm_;
            float y_norm_sq = 0.0f;
            for (size_t k = 0; k < D; ++k) y_norm_sq += y[k] * y[k];
            float y_norm = std::sqrt(y_norm_sq);

            std::vector<float> y_rot(D);
            float y_inv = 1.0f / y_norm;
            for (size_t k = 0; k < D; ++k) y_rot[k] = y[k] * y_inv;
            randomizedHadamard(y_rot.data(), rot_signs.data(), D);

            // Rotate x
            std::vector<float> x_rot(D);
            float x_inv = 1.0f / x_norm;
            for (size_t k = 0; k < D; ++k) x_rot[k] = x[k] * x_inv;
            randomizedHadamard(x_rot.data(), rot_signs.data(), D);

            std::vector<float> cv(D);
            code.dequantizeBatch(cv.data(), D);
            float ip_mse = 0.0f;
            float ip_res = 0.0f;
            for (size_t k = 0; k < D; ++k) {
                ip_mse += y_rot[k] * cv[k];
                ip_res += y_rot[k] * (x_rot[k] - cv[k]);
            }
            float reconstructed = (ip_mse + ip_res) * x_norm * y_norm;
            double err = std::abs(reconstructed - exact_ip);
            max_err = std::max(max_err, err);
        }
    }

    // Should be exact up to float32 precision
    constexpr double THRESHOLD = 1e-4;
    bool pass = max_err < THRESHOLD;
    std::cout << "  Max |MSE+residual - exact|: " << std::scientific
              << max_err;
    if (!pass) {
        std::cout << "  FAIL (expected < " << THRESHOLD << ")";
    } else {
        std::cout << "  PASS";
    }
    std::cout << std::endl;
    return pass;
}

// ---------------------------------------------------------------------------
// Test 4: IP correlation (single seed, practical scenario)
// ---------------------------------------------------------------------------
bool test_ip_correlation() {
    std::cout << "=== Test 4: Asymmetric IP correlation ===" << std::endl;

    constexpr size_t D = 128;
    constexpr size_t N = 500;
    constexpr size_t NUM_QUERIES = 20;

    auto embeddings = generateFaceEmbeddings(N, D);
    uint64_t rot_seed = 42;
    uint64_t qjl_seed = 137;
    const TurboQuantEncoder enc(D, 4, rot_seed, qjl_seed);

    std::vector<TurboQuantCode> codes(N);
    for (size_t i = 0; i < N; ++i) {
        codes[i] = enc.encode(embeddings[i].data());
    }

    double sum_corr = 0.0;
    std::mt19937_64 rng(999);
    std::uniform_int_distribution<size_t> query_dist(0, N - 1);

    for (size_t q = 0; q < NUM_QUERIES; ++q) {
        size_t qi = query_dist(rng);
        const float* query = embeddings[qi].data();

        std::vector<float> exact_ips(N), approx_ips(N);
        for (size_t i = 0; i < N; ++i) {
            exact_ips[i] = dot(query, embeddings[i].data(), D);
            approx_ips[i] = enc.asymmetricInnerProduct(query, codes[i]);
        }

        // Pearson correlation
        float mean_e = 0, mean_a = 0;
        for (size_t i = 0; i < N; ++i) {
            mean_e += exact_ips[i];
            mean_a += approx_ips[i];
        }
        mean_e /= N; mean_a /= N;

        float cov = 0, var_e = 0, var_a = 0;
        for (size_t i = 0; i < N; ++i) {
            float de = exact_ips[i] - mean_e;
            float da = approx_ips[i] - mean_a;
            cov += de * da;
            var_e += de * de;
            var_a += da * da;
        }
        float corr = cov / (std::sqrt(var_e) * std::sqrt(var_a) + 1e-10f);
        sum_corr += corr;
    }

    double avg_corr = sum_corr / NUM_QUERIES;

    constexpr double THRESHOLD = 0.85;
    bool pass = avg_corr > THRESHOLD;
    std::cout << "  Average Pearson correlation: " << std::fixed
              << std::setprecision(4) << avg_corr;
    if (!pass) {
        std::cout << "  FAIL (expected > " << THRESHOLD << ")";
    } else {
        std::cout << "  PASS";
    }
    std::cout << std::endl;
    return pass;
}

// ---------------------------------------------------------------------------
// Test 5: Recall@K
// ---------------------------------------------------------------------------
bool test_recall_at_k() {
    std::cout << "=== Test 5: Recall@K (L2 ranking) ===" << std::endl;

    constexpr size_t D = 128;
    constexpr size_t N = 2000;
    constexpr size_t NUM_QUERIES = 50;
    constexpr size_t K = 10;

    auto embeddings = generateFaceEmbeddings(N, D, 100);
    uint64_t rot_seed = 42;
    uint64_t qjl_seed = 137;
    const TurboQuantEncoder enc(D, 4, rot_seed, qjl_seed);

    std::vector<TurboQuantCode> codes(N);
    for (size_t i = 0; i < N; ++i) {
        codes[i] = enc.encode(embeddings[i].data());
    }

    std::mt19937_64 rng(777);
    std::uniform_int_distribution<size_t> query_dist(0, N - 1);

    size_t total_hits = 0;

    for (size_t q = 0; q < NUM_QUERIES; ++q) {
        size_t qi = query_dist(rng);
        const float* query = embeddings[qi].data();

        // Exact top-K
        std::vector<std::pair<float, size_t>> exact_dists(N);
        for (size_t i = 0; i < N; ++i) {
            exact_dists[i] = {l2_dist(query, embeddings[i].data(), D), i};
        }
        std::partial_sort(exact_dists.begin(), exact_dists.begin() + K,
                          exact_dists.end());

        // Approximate top-K
        std::vector<std::pair<float, size_t>> approx_dists(N);
        for (size_t i = 0; i < N; ++i) {
            approx_dists[i] = {enc.asymmetricL2(query, codes[i]), i};
        }
        std::partial_sort(approx_dists.begin(), approx_dists.begin() + K,
                          approx_dists.end());

        // Count hits
        for (size_t i = 0; i < K; ++i) {
            for (size_t j = 0; j < K; ++j) {
                if (approx_dists[i].second == exact_dists[j].second) {
                    ++total_hits;
                    break;
                }
            }
        }
    }

    float recall = static_cast<float>(total_hits)
                 / static_cast<float>(NUM_QUERIES * K);

    constexpr float THRESHOLD = 0.60f;
    bool pass = recall > THRESHOLD;
    std::cout << "  Recall@" << K << ": " << std::fixed << std::setprecision(2)
              << (recall * 100.0f) << "%";
    if (!pass) {
        std::cout << "  FAIL (expected > " << (THRESHOLD * 100.0f) << "%)";
    } else {
        std::cout << "  PASS";
    }
    std::cout << std::endl;
    return pass;
}

// ---------------------------------------------------------------------------
// Memory footprint report (informational, no pass/fail)
// ---------------------------------------------------------------------------
void test_memory_footprint() {
    std::cout << "=== Memory footprint ===" << std::endl;

    constexpr size_t D = 128;
    uint64_t rot_seed = 42;
    uint64_t qjl_seed = 137;
    const TurboQuantEncoder enc(D, 4, rot_seed, qjl_seed);

    size_t raw_bytes = D * sizeof(float);
    size_t tq_bytes = enc.codeSizeBytes();

    std::cout << "  Dimension:       " << D << std::endl;
    std::cout << "  Bit budget:      " << enc.totalBits()
              << " (MSE: " << enc.mseBits() << " + QJL: 1)" << std::endl;
    std::cout << "  Raw float32:     " << raw_bytes << " bytes" << std::endl;
    std::cout << "  TurboQuant code: " << tq_bytes << " bytes" << std::endl;
    std::cout << "  Effective bpc:   " << std::fixed << std::setprecision(1)
              << enc.effectiveBitsPerCoord() << " bits/coord" << std::endl;
    std::cout << "  Compression:     " << std::setprecision(1)
              << static_cast<float>(raw_bytes) / tq_bytes << "x" << std::endl;
    std::cout << "  Per 1M vectors:  "
              << std::setprecision(0)
              << (raw_bytes * 1e6 / (1024.0 * 1024.0)) << " MB -> "
              << (tq_bytes * 1e6 / (1024.0 * 1024.0)) << " MB" << std::endl;

    // Tight-packed estimate (Phase 2 target)
    size_t tight_sq = (D * enc.mseBits() + 7) / 8;
    size_t tight_qjl = (D + 7) / 8;
    size_t tight_total = tight_sq + tight_qjl + sizeof(float) * 3;
    std::cout << "\n  Tight-packed estimate: " << tight_total
              << " bytes (" << std::setprecision(1)
              << static_cast<float>(raw_bytes) / tight_total << "x)"
              << std::endl;
}

// TurboQuantSpace is now in hnswlib/turbo_quant_space.h

// ---------------------------------------------------------------------------
// Test 6: Serialization round-trip
// ---------------------------------------------------------------------------
bool test_serialization_roundtrip() {
    std::cout << "=== Test 6: Serialization round-trip ===" << std::endl;

    constexpr size_t D = 128;
    constexpr size_t N = 100;

    auto embeddings = generateFaceEmbeddings(N, D);
    uint64_t rot_seed = 42;
    uint64_t qjl_seed = 137;
    const TurboQuantEncoder enc(D, 4, rot_seed, qjl_seed);

    double max_err = 0.0;
    bool all_match = true;

    for (size_t i = 0; i < N; ++i) {
        auto code = enc.encode(embeddings[i].data());

        // Serialize
        std::vector<char> buf(TurboQuantCode::codeSizeBytes(D));
        code.serializeTo(buf.data(), D);

        // Deserialize
        auto code3 = TurboQuantCode::deserializeFrom(
            buf.data(), D, code.boundaries(), code.numBoundaries(),
            code.centroids());

        // Check fields match
        if (code3.norm_ != code.norm_ || code3.gamma_ != code.gamma_
            || code3.sigma_ != code.sigma_) {
            all_match = false;
            break;
        }
        if (code3.sq_packed_ != code.sq_packed_
            || code3.qjl_signs_ != code.qjl_signs_) {
            all_match = false;
            break;
        }

        // Verify dequantized values match
        std::vector<float> recon1(D), recon2(D);
        code.dequantizeBatch(recon1.data(), D);
        code3.dequantizeBatch(recon2.data(), D);
        for (size_t j = 0; j < D; ++j) {
            double err = std::abs(recon1[j] - recon2[j]);
            max_err = std::max(max_err, err);
        }
    }

    bool pass = all_match && (max_err < 1e-10);
    std::cout << "  Fields match: " << (all_match ? "yes" : "NO")
              << ", max dequant err: " << std::scientific << max_err;
    if (!pass) {
        std::cout << "  FAIL";
    } else {
        std::cout << "  PASS";
    }
    std::cout << std::endl;
    return pass;
}

// ---------------------------------------------------------------------------
// Test 7: HNSW integration — TurboQuantSpace + HierarchicalNSW
// ---------------------------------------------------------------------------
bool test_hnsw_integration() {
    std::cout << "=== Test 7: HNSW integration (TurboQuantSpace) ===" << std::endl;

    constexpr size_t D = 128;
    constexpr size_t N = 5000;
    constexpr size_t NUM_QUERIES = 100;
    constexpr size_t K = 10;
    constexpr size_t M = 16;
    constexpr size_t EF_CONSTRUCTION = 200;

    auto embeddings = generateFaceEmbeddings(N, D, 100);

    uint64_t rot_seed = 42;
    uint64_t qjl_seed = 137;
    TurboQuantSpace space(D, 4, rot_seed, qjl_seed);

    hnswlib::HierarchicalNSW<float> hnsw(&space, N, M, EF_CONSTRUCTION);

    // Encode and insert
    std::vector<char> buf(space.codeSizeBytes());
    for (size_t i = 0; i < N; ++i) {
        space.encodeVector(embeddings[i].data(), buf.data());
        hnsw.addPoint(buf.data(), i);
    }

    hnsw.setEf(64);

    // Search and measure recall
    std::mt19937_64 rng(555);
    std::uniform_int_distribution<size_t> query_dist(0, N - 1);
    size_t total_hits = 0;

    for (size_t q = 0; q < NUM_QUERIES; ++q) {
        size_t qi = query_dist(rng);
        const float* query = embeddings[qi].data();

        // Exact top-K by brute-force L2
        std::vector<std::pair<float, size_t>> exact_dists(N);
        for (size_t i = 0; i < N; ++i) {
            exact_dists[i] = {l2_dist(query, embeddings[i].data(), D), i};
        }
        std::partial_sort(exact_dists.begin(), exact_dists.begin() + K,
                          exact_dists.end());

        // HNSW search with prepared query
        auto pq = space.prepareQuery(query);
        space.beginSearch(pq);
        auto result = hnsw.searchKnn(query, K);
        space.endSearch();

        // Collect HNSW results
        std::vector<size_t> hnsw_ids;
        while (!result.empty()) {
            hnsw_ids.push_back(result.top().second);
            result.pop();
        }

        // Count hits
        for (size_t i = 0; i < K; ++i) {
            for (size_t j = 0; j < hnsw_ids.size(); ++j) {
                if (hnsw_ids[j] == exact_dists[i].second) {
                    ++total_hits;
                    break;
                }
            }
        }
    }

    float recall = static_cast<float>(total_hits)
                 / static_cast<float>(NUM_QUERIES * K);

    // Memory comparison
    size_t raw_bytes = D * sizeof(float);
    size_t tq_bytes = space.codeSizeBytes();
    std::cout << "  Memory: " << raw_bytes << " B/vec (raw float) vs "
              << tq_bytes << " B/vec (TurboQuant) = "
              << std::fixed << std::setprecision(1)
              << static_cast<float>(raw_bytes) / tq_bytes << "x compression"
              << std::endl;

    constexpr float THRESHOLD = 0.45f;
    bool pass = recall > THRESHOLD;
    std::cout << "  Recall@" << K << ": " << std::fixed << std::setprecision(2)
              << (recall * 100.0f) << "%";
    if (!pass) {
        std::cout << "  FAIL (expected > " << (THRESHOLD * 100.0f) << "%)";
    } else {
        std::cout << "  PASS";
    }
    std::cout << std::endl;
    return pass;
}

// ---------------------------------------------------------------------------
// Test 8: Save/load index — verify persistence preserves search quality
// ---------------------------------------------------------------------------
bool test_save_load_index() {
    std::cout << "=== Test 8: Save/load index ===" << std::endl;

    constexpr size_t D = 128;
    constexpr size_t N = 2000;
    constexpr size_t NUM_QUERIES = 50;
    constexpr size_t K = 10;
    constexpr size_t M = 16;
    constexpr size_t EF_CONSTRUCTION = 200;

    auto embeddings = generateFaceEmbeddings(N, D, 50);

    uint64_t rot_seed = 42;
    uint64_t qjl_seed = 137;
    TurboQuantSpace space(D, 4, rot_seed, qjl_seed);

    // Build index
    hnswlib::HierarchicalNSW<float> hnsw(&space, N, M, EF_CONSTRUCTION);
    std::vector<char> buf(space.codeSizeBytes());
    for (size_t i = 0; i < N; ++i) {
        space.encodeVector(embeddings[i].data(), buf.data());
        hnsw.addPoint(buf.data(), i);
    }
    hnsw.setEf(64);

    // Search before save — collect results for comparison
    std::mt19937_64 rng(333);
    std::uniform_int_distribution<size_t> query_dist(0, N - 1);
    std::vector<size_t> query_indices(NUM_QUERIES);
    for (size_t q = 0; q < NUM_QUERIES; ++q) query_indices[q] = query_dist(rng);

    std::vector<std::vector<size_t>> results_before(NUM_QUERIES);
    for (size_t q = 0; q < NUM_QUERIES; ++q) {
        const float* query = embeddings[query_indices[q]].data();
        auto pq = space.prepareQuery(query);
        space.beginSearch(pq);
        auto result = hnsw.searchKnn(query, K);
        space.endSearch();
        while (!result.empty()) {
            results_before[q].push_back(result.top().second);
            result.pop();
        }
        std::sort(results_before[q].begin(), results_before[q].end());
    }

    // Save to file
    const std::string filename = "test_tq_index.bin";
    hnsw.saveIndex(filename);

    // Load into a new index (same space, fresh HierarchicalNSW)
    hnswlib::HierarchicalNSW<float> hnsw2(&space, filename);
    hnsw2.setEf(64);

    // Search after load — results must be identical
    bool all_match = true;
    size_t total_hits_vs_exact = 0;

    for (size_t q = 0; q < NUM_QUERIES; ++q) {
        const float* query = embeddings[query_indices[q]].data();

        auto pq = space.prepareQuery(query);
        space.beginSearch(pq);
        auto result = hnsw2.searchKnn(query, K);
        space.endSearch();

        std::vector<size_t> results_after;
        while (!result.empty()) {
            results_after.push_back(result.top().second);
            result.pop();
        }
        std::sort(results_after.begin(), results_after.end());

        if (results_after != results_before[q]) {
            all_match = false;
        }

        // Also check recall vs exact brute-force
        std::vector<std::pair<float, size_t>> exact_dists(N);
        for (size_t i = 0; i < N; ++i) {
            exact_dists[i] = {l2_dist(query, embeddings[i].data(), D), i};
        }
        std::partial_sort(exact_dists.begin(), exact_dists.begin() + K,
                          exact_dists.end());
        for (size_t i = 0; i < K; ++i) {
            for (size_t j = 0; j < results_after.size(); ++j) {
                if (results_after[j] == exact_dists[i].second) {
                    ++total_hits_vs_exact;
                    break;
                }
            }
        }
    }

    float recall = static_cast<float>(total_hits_vs_exact)
                 / static_cast<float>(NUM_QUERIES * K);

    // Clean up
    std::remove(filename.c_str());

    bool pass = all_match && (recall > 0.45f);
    std::cout << "  Results match before/after: "
              << (all_match ? "yes" : "NO") << std::endl;
    std::cout << "  Recall@" << K << " after load: " << std::fixed
              << std::setprecision(2) << (recall * 100.0f) << "%";
    if (!pass) {
        std::cout << "  FAIL";
    } else {
        std::cout << "  PASS";
    }
    std::cout << std::endl;
    return pass;
}

// ---------------------------------------------------------------------------
// Large-scale comparison: L2Space vs TurboQuantSpace
// 1M vectors, dim=1024.  Reports memory, file size, build time, search time,
// recall@10.
// ---------------------------------------------------------------------------

/// Returns resident set size (RSS) in bytes. macOS only; returns 0 elsewhere.
static size_t getCurrentRSS() {
#ifdef __APPLE__
    struct mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  (task_info_t)&info, &count) == KERN_SUCCESS) {
        return info.resident_size;
    }
#endif
    return 0;
}

static size_t fileSize(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    return f.good() ? static_cast<size_t>(f.tellg()) : 0;
}

/// Generates a single random unit vector (no storage of the full dataset).
static void generateRandomVector(float* out, size_t d, std::mt19937_64& rng) {
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    for (size_t i = 0; i < d; ++i) out[i] = gauss(rng);
    normalize(out, d);
}

void test_large_scale_comparison() {
    std::cout << "=== Large-scale comparison: L2Space vs TurboQuantSpace ==="
              << std::endl;

    constexpr size_t D = 1024;
    constexpr size_t N = 1000000;
    constexpr size_t NUM_QUERIES = 100;
    constexpr size_t K = 10;
    constexpr size_t M = 16;
    constexpr size_t EF_CONSTRUCTION = 100;
    constexpr size_t EF_SEARCH = 64;

    const std::string file_l2 = "bench_l2.bin";
    const std::string file_tq = "bench_tq.bin";

    // Pre-generate query vectors (small — 100 × 1024 floats = 400 KB)
    std::mt19937_64 qrng(99999);
    std::vector<std::vector<float>> queries(NUM_QUERIES,
        std::vector<float>(D));
    for (size_t q = 0; q < NUM_QUERIES; ++q) {
        generateRandomVector(queries[q].data(), D, qrng);
    }

    // -----------------------------------------------------------------------
    // Part A: Standard L2Space HNSW
    // -----------------------------------------------------------------------
    std::cout << "\n  --- L2Space (raw float32) ---" << std::endl;
    {
        hnswlib::L2Space l2space(D);
        size_t rss_before = getCurrentRSS();

        auto t0 = std::chrono::high_resolution_clock::now();
        hnswlib::HierarchicalNSW<float> hnsw(&l2space, N, M, EF_CONSTRUCTION);

        std::mt19937_64 rng(42);
        std::vector<float> vec(D);
        for (size_t i = 0; i < N; ++i) {
            generateRandomVector(vec.data(), D, rng);
            hnsw.addPoint(vec.data(), i);
            if ((i + 1) % 100000 == 0) {
                std::cout << "    inserted " << (i + 1) / 1000 << "K..."
                          << std::endl;
            }
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double build_sec = std::chrono::duration<double>(t1 - t0).count();

        size_t rss_after = getCurrentRSS();
        size_t rss_delta = (rss_after > rss_before) ? (rss_after - rss_before) : 0;

        // Save
        hnsw.saveIndex(file_l2);
        size_t fsize = fileSize(file_l2);

        // Search
        hnsw.setEf(EF_SEARCH);
        t0 = std::chrono::high_resolution_clock::now();
        for (size_t q = 0; q < NUM_QUERIES; ++q) {
            auto result = hnsw.searchKnn(queries[q].data(), K);
            (void)result;
        }
        t1 = std::chrono::high_resolution_clock::now();
        double search_sec = std::chrono::duration<double>(t1 - t0).count();

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "    Build time:    " << build_sec << " s" << std::endl;
        std::cout << "    RSS delta:     " << (rss_delta / (1024.0 * 1024.0))
                  << " MB" << std::endl;
        std::cout << "    data_size/vec: " << l2space.get_data_size()
                  << " B" << std::endl;
        std::cout << "    Index file:    " << (fsize / (1024.0 * 1024.0))
                  << " MB" << std::endl;
        std::cout << "    Search time:   " << (search_sec / NUM_QUERIES * 1000.0)
                  << " ms/query (" << NUM_QUERIES << " queries, ef="
                  << EF_SEARCH << ")" << std::endl;

        // Cleanup — release memory before TurboQuant build
        std::remove(file_l2.c_str());
    }

    // -----------------------------------------------------------------------
    // Part B: TurboQuantSpace HNSW
    // -----------------------------------------------------------------------
    std::cout << "\n  --- TurboQuantSpace (3-bit SQ + 1-bit QJL) ---"
              << std::endl;
    {
        uint64_t rot_seed = 42;
        uint64_t qjl_seed = 137;
        TurboQuantSpace tqspace(D, 4, rot_seed, qjl_seed);
        size_t rss_before = getCurrentRSS();

        auto t0 = std::chrono::high_resolution_clock::now();
        hnswlib::HierarchicalNSW<float> hnsw(&tqspace, N, M, EF_CONSTRUCTION);

        std::mt19937_64 rng(42);  // same seed → same data
        std::vector<float> vec(D);
        std::vector<char> buf(tqspace.codeSizeBytes());
        for (size_t i = 0; i < N; ++i) {
            generateRandomVector(vec.data(), D, rng);
            tqspace.encodeVector(vec.data(), buf.data());
            hnsw.addPoint(buf.data(), i);
            if ((i + 1) % 100000 == 0) {
                std::cout << "    inserted " << (i + 1) / 1000 << "K..."
                          << std::endl;
            }
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double build_sec = std::chrono::duration<double>(t1 - t0).count();

        size_t rss_after = getCurrentRSS();
        size_t rss_delta = (rss_after > rss_before) ? (rss_after - rss_before) : 0;

        // Save
        hnsw.saveIndex(file_tq);
        size_t fsize = fileSize(file_tq);

        // Search
        hnsw.setEf(EF_SEARCH);
        t0 = std::chrono::high_resolution_clock::now();
        for (size_t q = 0; q < NUM_QUERIES; ++q) {
            auto pq = tqspace.prepareQuery(queries[q].data());
            tqspace.beginSearch(pq);
            auto result = hnsw.searchKnn(queries[q].data(), K);
            tqspace.endSearch();
            (void)result;
        }
        t1 = std::chrono::high_resolution_clock::now();
        double search_sec = std::chrono::duration<double>(t1 - t0).count();

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "    Build time:    " << build_sec << " s" << std::endl;
        std::cout << "    RSS delta:     " << (rss_delta / (1024.0 * 1024.0))
                  << " MB" << std::endl;
        std::cout << "    data_size/vec: " << tqspace.get_data_size()
                  << " B" << std::endl;
        std::cout << "    Index file:    " << (fsize / (1024.0 * 1024.0))
                  << " MB" << std::endl;
        std::cout << "    Search time:   " << (search_sec / NUM_QUERIES * 1000.0)
                  << " ms/query (" << NUM_QUERIES << " queries, ef="
                  << EF_SEARCH << ")" << std::endl;

        // Recall: brute-force exact L2 over 1M for each query
        // Regenerate data with the same seed
        std::cout << "    Computing recall (brute-force)..." << std::endl;
        std::mt19937_64 rng2(42);
        // Store all vectors for brute-force — but 1M×1024 = 4GB is too much.
        // Instead, compute exact distances on-the-fly per query.
        size_t total_hits = 0;
        for (size_t q = 0; q < NUM_QUERIES; ++q) {
            const float* query = queries[q].data();

            // HNSW result
            auto pq = tqspace.prepareQuery(query);
            tqspace.beginSearch(pq);
            auto result = hnsw.searchKnn(query, K);
            tqspace.endSearch();
            std::vector<size_t> hnsw_ids;
            while (!result.empty()) {
                hnsw_ids.push_back(result.top().second);
                result.pop();
            }

            // Exact top-K: scan all vectors, keep a partial top-K heap
            std::vector<std::pair<float, size_t>> topk;
            topk.reserve(K + 1);
            std::mt19937_64 rng_scan(42);
            std::vector<float> scan_vec(D);
            for (size_t i = 0; i < N; ++i) {
                generateRandomVector(scan_vec.data(), D, rng_scan);
                float dist = l2_dist(query, scan_vec.data(), D);
                if (topk.size() < K) {
                    topk.push_back({dist, i});
                    if (topk.size() == K) {
                        std::make_heap(topk.begin(), topk.end());
                    }
                } else if (dist < topk.front().first) {
                    std::pop_heap(topk.begin(), topk.end());
                    topk.back() = {dist, i};
                    std::push_heap(topk.begin(), topk.end());
                }
            }

            for (size_t ti = 0; ti < topk.size(); ++ti) {
                for (size_t j = 0; j < hnsw_ids.size(); ++j) {
                    if (hnsw_ids[j] == topk[ti].second) {
                        ++total_hits;
                        break;
                    }
                }
            }
        }
        float recall = static_cast<float>(total_hits)
                     / static_cast<float>(NUM_QUERIES * K);
        std::cout << "    Recall@" << K << ":     " << std::setprecision(2)
                  << (recall * 100.0f) << "%" << std::endl;

        std::remove(file_tq.c_str());
    }
}

// ---------------------------------------------------------------------------
// Performance benchmark (informational, no pass/fail)
// ---------------------------------------------------------------------------
void test_benchmark() {
    std::cout << "=== Performance benchmark ===" << std::endl;

    constexpr size_t D = 128;
    constexpr size_t N = 10000;
    constexpr size_t NUM_QUERIES = 100;

    auto embeddings = generateFaceEmbeddings(N, D, 100);
    uint64_t rot_seed = 42;
    uint64_t qjl_seed = 137;
    const TurboQuantEncoder enc(D, 4, rot_seed, qjl_seed);

    // 1. Encode throughput
    std::vector<TurboQuantCode> codes(N);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; ++i) {
        codes[i] = enc.encode(embeddings[i].data());
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double encode_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

    // 2. DequantizeBatch throughput
    std::vector<float> recon(D);
    t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; ++i) {
        codes[i].dequantizeBatch(recon.data(), D);
    }
    t1 = std::chrono::high_resolution_clock::now();
    double dequant_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

    // 3. AsymmetricL2 (naive) — includes 2 RHT per comparison
    t0 = std::chrono::high_resolution_clock::now();
    volatile float sink = 0.0f;
    for (size_t q = 0; q < NUM_QUERIES; ++q) {
        for (size_t i = 0; i < N; ++i) {
            sink = enc.asymmetricL2(embeddings[q].data(), codes[i]);
        }
    }
    t1 = std::chrono::high_resolution_clock::now();
    double naive_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    size_t naive_total = NUM_QUERIES * N;

    // 4. Serialization round-trip
    std::vector<std::vector<char>> buffers(N,
        std::vector<char>(TurboQuantCode::codeSizeBytes(D)));
    t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; ++i) {
        codes[i].serializeTo(buffers[i].data(), D);
    }
    t1 = std::chrono::high_resolution_clock::now();
    double ser_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

    t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; ++i) {
        auto c = TurboQuantCode::deserializeFrom(
            buffers[i].data(), D,
            codes[0].boundaries(), codes[0].numBoundaries(),
            codes[0].centroids());
        (void)c;
    }
    t1 = std::chrono::high_resolution_clock::now();
    double deser_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

    // 5. Prepared-query L2 (fast path — no RHT per candidate)
    TurboQuantSpace space(D, 4, rot_seed, qjl_seed);
    // Encode into flat buffers via space
    std::vector<std::vector<char>> flat_codes(N,
        std::vector<char>(space.codeSizeBytes()));
    for (size_t i = 0; i < N; ++i) {
        space.encodeVector(embeddings[i].data(), flat_codes[i].data());
    }

    t0 = std::chrono::high_resolution_clock::now();
    auto dist_func = space.get_dist_func();
    auto* dist_param = space.get_dist_func_param();
    for (size_t q = 0; q < NUM_QUERIES; ++q) {
        auto pq = space.prepareQuery(embeddings[q].data());
        space.beginSearch(pq);
        for (size_t i = 0; i < N; ++i) {
            sink = dist_func(embeddings[q].data(), flat_codes[i].data(),
                             dist_param);
        }
        space.endSearch();
    }
    t1 = std::chrono::high_resolution_clock::now();
    double prepared_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    (void)sink;

    // Report
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "  encode:            " << std::setw(8)
              << (encode_us / N) << " us/vec  ("
              << static_cast<size_t>(N * 1e6 / encode_us) << " vec/s)" << std::endl;
    std::cout << "  dequantizeBatch:   " << std::setw(8)
              << (dequant_us / N) << " us/vec  ("
              << static_cast<size_t>(N * 1e6 / dequant_us) << " vec/s)" << std::endl;
    std::cout << "  asymmetricL2:      " << std::setw(8)
              << (naive_us / naive_total * 1000.0) << " ns/cmp  ("
              << static_cast<size_t>(naive_total * 1e6 / naive_us) << " cmp/s)"
              << std::endl;
    std::cout << "  serialize:         " << std::setw(8)
              << (ser_us / N * 1000.0) << " ns/op   ("
              << static_cast<size_t>(N * 1e6 / ser_us) << " op/s)" << std::endl;
    std::cout << "  deserialize:       " << std::setw(8)
              << (deser_us / N * 1000.0) << " ns/op   ("
              << static_cast<size_t>(N * 1e6 / deser_us) << " op/s)" << std::endl;
    std::cout << "  preparedL2:        " << std::setw(8)
              << (prepared_us / naive_total * 1000.0) << " ns/cmp  ("
              << static_cast<size_t>(naive_total * 1e6 / prepared_us) << " cmp/s)"
              << std::endl;
    double speedup = naive_us / prepared_us;
    std::cout << "  speedup (prepared vs naive): " << std::setprecision(2)
              << speedup << "x" << std::endl;
}

// ---------------------------------------------------------------------------
int main() {
    std::cout << "TurboQuant Algorithm 2 — Correctness Tests\n"
              << "d=128, b=4 (3-bit MSE + 1-bit QJL)\n"
              << "============================================\n" << std::endl;

    int passed = 0, total = 0;
    auto run = [&](bool result) {
        ++total; if (result) ++passed; std::cout << std::endl;
    };

    run(test_wht_energy());
    run(test_sq_distortion());
    run(test_unbiasedness());
    run(test_lossless_identity());
    run(test_ip_correlation());
    run(test_recall_at_k());
    run(test_serialization_roundtrip());
    run(test_hnsw_integration());
    run(test_save_load_index());

    test_memory_footprint();
    std::cout << std::endl;
    test_benchmark();
    std::cout << std::endl;
    test_large_scale_comparison();
    std::cout << std::endl;

    std::cout << "============================================" << std::endl;
    std::cout << "Results: " << passed << "/" << total << " tests passed"
              << std::endl;

    return (passed == total) ? 0 : 1;
}
