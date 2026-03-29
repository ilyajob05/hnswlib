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

#include "hnswlib/turbo_quant.h"

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>

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

    float max_err = 0.0f;
    for (int trial = 0; trial < 100; ++trial) {
        std::vector<float> v(D);
        for (auto& x : v) x = gauss(rng);
        float energy_before = dot(v.data(), v.data(), D);

        std::vector<float> rotated(v);
        randomizedHadamard(rotated.data(), D, 42);
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
    TurboQuantEncoder enc(D, /*bits_per_coord=*/4);  // 3-bit MSE + 1-bit QJL

    double total_mse = 0.0;
    for (size_t i = 0; i < N; ++i) {
        auto code = enc.encode(embeddings[i].data());

        // Reconstruct MSE part in rotated space
        std::vector<float> rotated(embeddings[i]);
        float inv_norm = 1.0f / code.norm_;
        for (size_t j = 0; j < D; ++j) rotated[j] *= inv_norm;
        randomizedHadamard(rotated.data(), D, enc.rotationSeed());

        float mse = 0.0f;
        for (size_t j = 0; j < D; ++j) {
            float recon = sqDequantize(code.sq_packed_[j], enc.centroids())
                          * code.sigma_;
            float diff = rotated[j] - recon;
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

    constexpr size_t D = 128;
    constexpr size_t N = 200;

    auto embeddings = generateFaceEmbeddings(N, D);
    uint64_t rot_seed = 42;

    double max_err = 0.0;
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = i + 1; j < std::min(N, i + 5); ++j) {
            const float* x = embeddings[i].data();
            const float* y = embeddings[j].data();
            float exact_ip = dot(x, y, D);

            TurboQuantEncoder enc(D, 4, rot_seed, /*qjl_seed=*/137);
            auto code = enc.encode(x);

            // Compute MSE IP in rotated space
            float x_norm = code.norm_;
            float y_norm_sq = 0.0f;
            for (size_t k = 0; k < D; ++k) y_norm_sq += y[k] * y[k];
            float y_norm = std::sqrt(y_norm_sq);

            std::vector<float> y_rot(D);
            float y_inv = 1.0f / y_norm;
            for (size_t k = 0; k < D; ++k) y_rot[k] = y[k] * y_inv;
            randomizedHadamard(y_rot.data(), D, rot_seed);

            // Rotate x
            std::vector<float> x_rot(D);
            float x_inv = 1.0f / x_norm;
            for (size_t k = 0; k < D; ++k) x_rot[k] = x[k] * x_inv;
            randomizedHadamard(x_rot.data(), D, rot_seed);

            float ip_mse = 0.0f;
            float ip_res = 0.0f;
            for (size_t k = 0; k < D; ++k) {
                float cv = sqDequantize(code.sq_packed_[k], enc.centroids())
                           * code.sigma_;
                ip_mse += y_rot[k] * cv;
                ip_res += y_rot[k] * (x_rot[k] - cv);
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
    TurboQuantEncoder enc(D, 4);

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
    TurboQuantEncoder enc(D, 4);

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
    TurboQuantEncoder enc(D, 4);

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

    test_memory_footprint();
    std::cout << std::endl;

    std::cout << "============================================" << std::endl;
    std::cout << "Results: " << passed << "/" << total << " tests passed"
              << std::endl;

    return (passed == total) ? 0 : 1;
}
