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

#include "hnswlib/hnswlib.h"
#include "hnswlib/turbo_quant.h"

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <chrono>

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

// ===========================================================================
// TurboQuantSpace — SpaceInterface adapter for HNSW integration (test-local)
// ===========================================================================

/// Parameter block passed as dist_func_param to the DISTFUNC.
struct TurboQuantDistParam {
    size_t dim;
    const float* centroids;
    const TurboQuantPreparedQuery* prepared_query;  ///< non-null = search mode
};

class TurboQuantSpace : public hnswlib::SpaceInterface<float> {
    size_t dim_;
    size_t code_size_;
    int mse_bits_;

    const float* boundaries_;
    int num_boundaries_;
    const float* centroids_;

    std::vector<float> rotation_signs_;
    std::vector<float> qjl_signs_;

    TurboQuantDistParam dist_param_;

    // -- Static distance functions -------------------------------------------

    static float distSearch(const TurboQuantPreparedQuery* pq,
                            const char* code_buf,
                            const TurboQuantDistParam* p) {
        const size_t dim = p->dim;
        const uint8_t* sq_packed = reinterpret_cast<const uint8_t*>(code_buf);
        const uint64_t* qjl_signs = reinterpret_cast<const uint64_t*>(
            code_buf + dim);
        const size_t qjl_bytes = ((dim + 63) / 64) * sizeof(uint64_t);
        const float* meta = reinterpret_cast<const float*>(
            code_buf + dim + qjl_bytes);
        const float x_norm  = meta[0];
        const float gamma   = meta[1];
        const float sigma   = meta[2];

        float ip_mse = 0.0f;
        for (size_t i = 0; i < dim; ++i) {
            ip_mse += pq->q_rot[i] * p->centroids[sq_packed[i]] * sigma;
        }

        float dot_qjl = 0.0f;
        for (size_t i = 0; i < dim; ++i) {
            const bool positive = (qjl_signs[i / 64] >> (i % 64)) & 1ULL;
            const float sign = positive ? 1.0f : -1.0f;
            dot_qjl += pq->s_q[i] * sign;
        }
        const float scale = std::sqrt(static_cast<float>(M_PI) / 2.0f)
                          / std::sqrt(static_cast<float>(dim));
        const float correction = scale * gamma * dot_qjl;

        const float ip = (ip_mse + correction) * x_norm * pq->q_norm;
        return std::max(0.0f, pq->q_norm_sq + x_norm * x_norm - 2.0f * ip);
    }

    static float distSymmetric(const char* buf_a, const char* buf_b,
                               const TurboQuantDistParam* p) {
        const size_t dim = p->dim;
        const size_t qjl_bytes = ((dim + 63) / 64) * sizeof(uint64_t);

        const uint8_t* sq_a = reinterpret_cast<const uint8_t*>(buf_a);
        const float* meta_a = reinterpret_cast<const float*>(
            buf_a + dim + qjl_bytes);
        const float norm_a  = meta_a[0];
        const float sigma_a = meta_a[2];

        const uint8_t* sq_b = reinterpret_cast<const uint8_t*>(buf_b);
        const float* meta_b = reinterpret_cast<const float*>(
            buf_b + dim + qjl_bytes);
        const float norm_b  = meta_b[0];
        const float sigma_b = meta_b[2];

        float ip_rot = 0.0f;
        for (size_t i = 0; i < dim; ++i) {
            ip_rot += (p->centroids[sq_a[i]] * sigma_a)
                    * (p->centroids[sq_b[i]] * sigma_b);
        }
        const float ip = ip_rot * norm_a * norm_b;
        return std::max(0.0f, norm_a * norm_a + norm_b * norm_b - 2.0f * ip);
    }

    static float turboQuantL2(const void* pVect1, const void* pVect2,
                              const void* param_ptr) {
        const auto* p = static_cast<const TurboQuantDistParam*>(param_ptr);
        const char* buf2 = static_cast<const char*>(pVect2);
        if (p->prepared_query != nullptr) {
            return distSearch(p->prepared_query, buf2, p);
        } else {
            const char* buf1 = static_cast<const char*>(pVect1);
            return distSymmetric(buf1, buf2, p);
        }
    }

    void encodeVectorImpl(const float* raw, char* out_buf) const {
        const size_t d = dim_;

        float norm_sq = 0.0f;
        for (size_t i = 0; i < d; ++i) norm_sq += raw[i] * raw[i];
        float norm = std::sqrt(norm_sq);
        float inv_norm = (norm > 1e-10f) ? (1.0f / norm) : 0.0f;

        std::vector<float> x_rot(d);
        for (size_t i = 0; i < d; ++i) x_rot[i] = raw[i] * inv_norm;
        randomizedHadamard(x_rot.data(), rotation_signs_.data(), d);

        float var = 0.0f;
        for (size_t i = 0; i < d; ++i) var += x_rot[i] * x_rot[i];
        float sigma = std::sqrt(var / static_cast<float>(d));
        if (sigma < 1e-10f) sigma = 1e-10f;
        float inv_sigma = 1.0f / sigma;

        std::vector<float> normalized(d);
        for (size_t i = 0; i < d; ++i) normalized[i] = x_rot[i] * inv_sigma;

        uint8_t* sq_out = reinterpret_cast<uint8_t*>(out_buf);
        for (size_t k = 0; k < d; ++k) {
            uint8_t idx = 0;
            for (int b = 0; b < num_boundaries_; ++b) {
                idx += (normalized[k] > boundaries_[b]);
            }
            sq_out[k] = idx;
        }

        std::vector<float> residual(d);
        for (size_t i = 0; i < d; ++i) {
            residual[i] = x_rot[i] - centroids_[sq_out[i]] * sigma;
        }
        float gamma_sq = 0.0f;
        for (size_t i = 0; i < d; ++i) gamma_sq += residual[i] * residual[i];
        float gamma = std::sqrt(gamma_sq);

        float inv_gamma = (gamma > 1e-10f) ? (1.0f / gamma) : 0.0f;
        std::vector<float> projected(d);
        for (size_t i = 0; i < d; ++i) projected[i] = residual[i] * inv_gamma;
        randomizedHadamard(projected.data(), qjl_signs_.data(), d);

        size_t num_words = (d + 63) / 64;
        uint64_t* qjl_out = reinterpret_cast<uint64_t*>(out_buf + d);
        for (size_t w = 0; w < num_words; ++w) qjl_out[w] = 0;
        for (size_t i = 0; i < d; ++i) {
            if (projected[i] >= 0.0f) {
                qjl_out[i / 64] |= (1ULL << (i % 64));
            }
        }

        size_t qjl_bytes = num_words * sizeof(uint64_t);
        float* meta = reinterpret_cast<float*>(out_buf + d + qjl_bytes);
        meta[0] = norm;
        meta[1] = gamma;
        meta[2] = sigma;
    }

 public:
    TurboQuantSpace(size_t dim, int bits_per_coord = 4,
                    uint64_t rot_seed = 42, uint64_t qjl_seed = 137)
        : dim_(dim)
        , code_size_(TurboQuantCode::codeSizeBytes(dim))
        , mse_bits_(bits_per_coord - 1)
        , boundaries_(nullptr)
        , num_boundaries_(0)
        , centroids_(nullptr)
    {
        if (mse_bits_ == 3) {
            boundaries_ = hnswlib::turboquant::detail::LM3.boundaries.data();
            num_boundaries_ = 7;
            centroids_ = hnswlib::turboquant::detail::LM3.centroids.data();
        } else if (mse_bits_ == 4) {
            boundaries_ = hnswlib::turboquant::detail::LM4.boundaries.data();
            num_boundaries_ = 15;
            centroids_ = hnswlib::turboquant::detail::LM4.centroids.data();
        }
        rotation_signs_ = generateSigns(dim_, rot_seed);
        qjl_signs_ = generateSigns(dim_, qjl_seed);

        dist_param_.dim = dim_;
        dist_param_.centroids = centroids_;
        dist_param_.prepared_query = nullptr;
    }

    size_t get_data_size() override { return code_size_; }
    hnswlib::DISTFUNC<float> get_dist_func() override { return &turboQuantL2; }
    void* get_dist_func_param() override { return &dist_param_; }

    size_t dim() const { return dim_; }
    size_t codeSizeBytes() const { return code_size_; }
    const float* rotationSigns() const { return rotation_signs_.data(); }
    const float* qjlSigns() const { return qjl_signs_.data(); }
    const float* boundaries() const { return boundaries_; }
    int numBoundaries() const { return num_boundaries_; }
    const float* centroids() const { return centroids_; }

    void encodeVector(const float* raw, char* out_buf) const {
        encodeVectorImpl(raw, out_buf);
    }

    TurboQuantPreparedQuery prepareQuery(const float* raw_query) const {
        TurboQuantPreparedQuery pq;
        const size_t d = dim_;

        pq.q_norm_sq = 0.0f;
        for (size_t i = 0; i < d; ++i)
            pq.q_norm_sq += raw_query[i] * raw_query[i];
        pq.q_norm = std::sqrt(pq.q_norm_sq);
        float q_inv = (pq.q_norm > 1e-10f) ? (1.0f / pq.q_norm) : 0.0f;

        pq.q_rot.resize(d);
        for (size_t i = 0; i < d; ++i) pq.q_rot[i] = raw_query[i] * q_inv;
        randomizedHadamard(pq.q_rot.data(), rotation_signs_.data(), d);

        pq.s_q = pq.q_rot;
        randomizedHadamard(pq.s_q.data(), qjl_signs_.data(), d);

        return pq;
    }

    void beginSearch(const TurboQuantPreparedQuery& pq) {
        dist_param_.prepared_query = &pq;
    }

    void endSearch() {
        dist_param_.prepared_query = nullptr;
    }
};

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

    std::cout << "============================================" << std::endl;
    std::cout << "Results: " << passed << "/" << total << " tests passed"
              << std::endl;

    return (passed == total) ? 0 : 1;
}
