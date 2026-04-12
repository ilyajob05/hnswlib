#pragma once
/// turbo_quant_space.h — TurboQuant SpaceInterface adapter + compress/save
/// utilities
///
/// Provides:
///   - TurboQuantSpace: SpaceInterface<float> for HNSW with TurboQuant codes
///   - compressIndex:   convert L2-built HNSW to compressed TQ representation
///   - saveRawVectors:  save original float vectors for re-ranking
///   - loadRawVectors:  load specific vectors by ID for re-ranking
///
/// Usage:
///   1. Build HNSW with L2Space (exact distances)
///   2. saveRawVectors (before compress, for re-ranking)
///   3. compressIndex (encodes floats to TQ codes in-place)
///   4. hnsw.saveIndex (saves compressed index)
///   5. space.setSearchMode(hnsw), then searchKnn(&pq, K)
///   6. Re-rank: loadRawVectors for shortlist, compute exact L2, pick top-K
///
/// Opt-in header: not included by hnswlib.h. Include directly when needed.

#include "hnswlib.h"
#include "turbo_quant.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <fstream>
#include <memory>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace hnswlib {
namespace turboquant {

// ===========================================================================
// TurboQuantSpace — SpaceInterface adapter for HNSW
// ===========================================================================

// ===========================================================================
// Lloyd-Max quantizer tables for Gaussian N(0,1)
// Optimal minimum-MSE scalar quantizer (Max, 1960).
// Boundaries = decision thresholds; centroids = reconstruction levels.
// Tables are symmetric around 0.
// ===========================================================================

// Hardcoded reference values (Max, 1960):
// LM3_CENTROIDS ≈ {-2.1519, -1.3440, -0.7560, -0.2451, 0.2451,
// 0.7560, 1.3440, 2.1519} LM4_CENTROIDS ≈ {-3.0867, -2.0995, -1.6180, -1.2562,
// -0.9423, -0.6568, -0.3881, -0.1284, ...}

struct LloydMaxTable {
    std::vector<float> boundaries;
    std::vector<float> centroids;
};

inline LloydMaxTable computeLloydMax(int bits, int maxIter = 1000,
                                     double tol = 1e-12) {
    const int levels = 1 << bits;
    const int half = levels / 2;

    // φ(x) — standard normal PDF
    auto phi = [](double x) -> double { return std::exp(-0.5 * x * x) / std::sqrt(2.0 * M_PI); };
    // Φ(x) — standard normal CDF
    auto Phi = [](double x) -> double { return 0.5 * std::erfc(-x * M_SQRT1_2); };
    // E[x | a < x < b] for N(0,1)
    auto conditionalMean = [&](double a, double b) -> double {
        double denom = Phi(b) - Phi(a);
        if (denom < 1e-15)
            return 0.5 * (a + b);
        return (phi(a) - phi(b)) / denom;
    };

    // Initialize: positive centroids uniformly in (0, 3.5)
    std::vector<double> pos_c(half);
    for (int i = 0; i < half; ++i) {
        pos_c[i] = (i + 0.5) * 3.5 / half;
    }

    for (int iter = 0; iter < maxIter; ++iter) {
        // Boundaries between positive centroids
        std::vector<double> pos_b(half - 1);
        for (int i = 0; i < half - 1; ++i) {
            pos_b[i] = 0.5 * (pos_c[i] + pos_c[i + 1]);
        }

        // Update centroids: E[x | boundary_k < x < boundary_{k+1}]
        std::vector<double> new_c(half);
        double maxDelta = 0.0;
        for (int i = 0; i < half; ++i) {
            double lo = (i == 0) ? 0.0 : pos_b[i - 1];
            double hi = (i == half - 1) ? 1e10 : pos_b[i];
            new_c[i] = conditionalMean(lo, hi);
            maxDelta = std::max(maxDelta, std::abs(new_c[i] - pos_c[i]));
        }
        pos_c = new_c;
        if (maxDelta < tol)
            break;
    }

    // Build full symmetric tables
    LloydMaxTable table;
    table.centroids.resize(levels);
    table.boundaries.resize(levels - 1);

    for (int i = 0; i < half; ++i) {
        table.centroids[half + i] = static_cast<float>(pos_c[i]);
        table.centroids[half - 1 - i] = -static_cast<float>(pos_c[i]);
    }
    // Boundary 0 is always 0.0 (symmetry axis)
    table.boundaries[half - 1] = 0.0f;
    for (int i = 0; i < half - 1; ++i) {
        double b = 0.5 * (pos_c[i] + pos_c[i + 1]);
        table.boundaries[half + i] = static_cast<float>(b);
        table.boundaries[half - 2 - i] = -static_cast<float>(b);
    }

    return table;
}

// ===========================================================================
// TurboQuantPreparedQuery — pre-computed query state for search
//
// Created once per query via prepareQuery(), reused across all candidate
// distance computations. Eliminates 2 RHT calls per distance evaluation.
// Lives on the caller's stack — no shared mutable state, fully thread-safe.
// ===========================================================================

struct TurboQuantPreparedQuery {
  std::vector<float> q_rot; ///< normalized + RHT-rotated query (dim floats)
  std::vector<float> s_q;   ///< RHT(q_rot, qjl_signs) for QJL correction
  float q_norm_sq;           ///< ||query||^2
  float q_norm;              ///< ||query||
  const float *centroids;    ///< pointer to centroid table (not owned)
};

// Forward declarations
class TurboQuantSpace;

// Distance function forward declarations (defined after TurboQuantSpace)
static float distSearchScalar(const void *q, const void *code_buf, const void *qty_ptr);
static float distBuildScalar(const void *pVect1, const void *pVect2, const void *param_ptr);
static float distSearchScalarB4(const void *q, const void *code_buf, const void *qty_ptr);
static float distBuildScalarB4(const void *pVect1, const void *pVect2, const void *param_ptr);

#if defined(USE_NEON)
static float distSearchNEON(const void *q, const void *code_buf, const void *qty_ptr);
static float distBuildNEON(const void *pVect1, const void *pVect2, const void *param_ptr);
static float distSearchNEONB4(const void *q, const void *code_buf, const void *qty_ptr);
static float distBuildNEONB4(const void *pVect1, const void *pVect2, const void *param_ptr);
#endif

#if defined(USE_SSE)
static float distSearchSSE(const void *q, const void *code_buf, const void *qty_ptr);
static float distBuildSSE(const void *pVect1, const void *pVect2, const void *param_ptr);
#endif

#if defined(USE_AVX)
static float distSearchAVX(const void *q, const void *code_buf, const void *qty_ptr);
static float distBuildAVX(const void *pVect1, const void *pVect2, const void *param_ptr);
#endif

class TurboQuantSpace : public SpaceInterface<float> {
    LloydMaxTable lm_table_;
    float *boundaries_;
    uint16_t num_boundaries_;
    float *centroids_;

    std::vector<float> rotation_signs_;
    std::vector<float> qjl_signs_precomp_;

    const uint64_t rot_seed_;
    const uint64_t qjl_seed_;
    const size_t dim_;
    const int bits_per_coord_;
    const bool packed_nibbles_;   // true when bits_per_coord_ <= 4
    const size_t packed_bytes_;   // bytes of packed region
    const size_t data_size_;      // packed + meta
    const int num_levels_;
    const float scale_;
    DISTFUNC<float> fstdistfunc_;
    DISTFUNC<float> fstdistfunc_build_;
    DISTFUNC<float> fstdistfunc_search_;

public:
    TurboQuantSpace(size_t dim,
                    int bits_per_coord = 4,
                    uint64_t rot_seed = 42,
                    uint64_t qjl_seed = 137)
        : rot_seed_(rot_seed)
        , qjl_seed_(qjl_seed)
        , dim_(dim)
        , bits_per_coord_(bits_per_coord)
        , packed_nibbles_(bits_per_coord <= 4)
        , packed_bytes_(TurboQuantCode::packedBytes(dim, bits_per_coord))
        , data_size_(TurboQuantCode::codeSizeBytes(dim, bits_per_coord))
        , num_levels_(1 << (bits_per_coord - 1))
        , scale_(std::sqrt(static_cast<float>(M_PI) / 2.0f) / std::sqrt(static_cast<float>(dim)))
    {
        assert(dim >= 4 && "TurboQuantSpace: dim must be at least 4");
        assert((dim & (dim - 1)) == 0 && "TurboQuantSpace: dim must be a power of 2");
        assert(bits_per_coord >= 2 &&
               "TurboQuantSpace: need at least 2 bits (1 MSE + 1 QJL)");
        assert(bits_per_coord <= 9 &&
               "TurboQuantSpace: max 9 bits (8-bit MSE + 1 QJL, uint8_t limit)");

        // Runtime SIMD dispatch
        // Default to build (symmetric code×code) distance.
        // Call setSearchMode(hnsw) after build to switch to asymmetric search.
        //
        // Packed-nibble layout (b<=4) currently uses scalar paths on all ISAs;
        // SIMD LUT-based variants are a follow-up optimization.
        if (packed_nibbles_) {
#if defined(USE_NEON)
            fstdistfunc_ = distBuildNEONB4;
            fstdistfunc_build_ = distBuildNEONB4;
            fstdistfunc_search_ = distSearchNEONB4;
#else
            fstdistfunc_ = distBuildScalarB4;
            fstdistfunc_build_ = distBuildScalarB4;
            fstdistfunc_search_ = distSearchScalarB4;
#endif
        } else {
#if defined(USE_AVX)
            fstdistfunc_ = distBuildAVX;
            fstdistfunc_build_ = distBuildAVX;
            fstdistfunc_search_ = distSearchAVX;
#elif defined(USE_SSE)
            fstdistfunc_ = distBuildSSE;
            fstdistfunc_build_ = distBuildSSE;
            fstdistfunc_search_ = distSearchSSE;
#elif defined(USE_NEON)
            fstdistfunc_ = distBuildNEON;
            fstdistfunc_build_ = distBuildNEON;
            fstdistfunc_search_ = distSearchNEON;
#else
            fstdistfunc_ = distBuildScalar;
            fstdistfunc_build_ = distBuildScalar;
            fstdistfunc_search_ = distSearchScalar;
#endif
        }

        lm_table_ = computeLloydMax(bits_per_coord - 1);
        boundaries_ = lm_table_.boundaries.data();
        num_boundaries_ = static_cast<uint16_t>(lm_table_.boundaries.size());
        centroids_ = lm_table_.centroids.data();
        rotation_signs_ = generateSigns(dim_, rot_seed_);
        qjl_signs_precomp_ = generateSigns(dim_, qjl_seed_);
    }

    // SpaceInterface
    size_t get_data_size() override { return data_size_; }
    DISTFUNC<float> get_dist_func() override { return fstdistfunc_; }
    void *get_dist_func_param() override { return this; }

    // Search distance function (asymmetric: PreparedQuery × code)
    DISTFUNC<float> get_search_dist_func() const { return fstdistfunc_search_; }

    // Accessors
    size_t dim() const { return dim_; }
    size_t codeSizeBytes() const { return data_size_; }
    size_t packedBytes() const { return packed_bytes_; }
    int bitsPerCoord() const { return bits_per_coord_; }
    bool packedNibbles() const { return packed_nibbles_; }
    int numLevels() const { return num_levels_; }
    float scale() const { return scale_; }
    const float *centroids() const { return centroids_; }
    uint64_t rotSeed() const { return rot_seed_; }
    uint64_t qjlSeed() const { return qjl_seed_; }

    // -- Encoding -------------------------------------------------------------

    /// Encode a raw float vector directly into an HNSW buffer slot.
    /// Layout (b>=5):  [packed: dim bytes]     [meta: 12 B]
    /// Layout (b<=4):  [packed: (dim+1)/2 B]  [meta: 12 B]
    ///   packed unit = (sq_idx << 1) | qjl_bit
    ///   b<=4: two units per byte, low nibble first (coord 2i, 2i+1)
    void encodeVector(const float *raw, void *out_buf) const {
        TurboQuantCode code(out_buf, dim_, bits_per_coord_);

        // Step 1: norm
        float norm_sq = 0.0f;
        for (size_t i = 0; i < dim_; ++i)
            norm_sq += raw[i] * raw[i];
        float norm = std::sqrt(norm_sq);

        // Step 2: normalize + RHT
        std::vector<float> rotated(dim_);
        float inv_norm = (norm > 1e-10f) ? (1.0f / norm) : 0.0f;
        for (size_t i = 0; i < dim_; ++i)
            rotated[i] = raw[i] * inv_norm;
        randomizedHadamard(rotated.data(), rotation_signs_.data(), dim_);

        // Step 3: sigma
        float var = 0.0f;
        for (size_t i = 0; i < dim_; ++i)
            var += rotated[i] * rotated[i];
        float sigma = std::sqrt(var / static_cast<float>(dim_));
        if (sigma < 1e-10f)
            sigma = 1e-10f;
        float inv_sigma = 1.0f / sigma;

        // Step 4: quantize + residual (single pass)
        std::vector<float> residual(dim_);
        std::vector<uint8_t> sq_idx(dim_);
        for (size_t i = 0; i < dim_; ++i) {
            float normalized = rotated[i] * inv_sigma;
            sq_idx[i] = quantize(normalized);
            residual[i] = rotated[i] - centroids_[sq_idx[i]] * sigma;
        }

        // Step 5: gamma
        float gamma_sq = 0.0f;
        for (size_t i = 0; i < dim_; ++i)
            gamma_sq += residual[i] * residual[i];
        float gamma = std::sqrt(gamma_sq);

        // Step 6: QJL projection -> pack units into output buffer.
        //   unit[i] = (sq_idx[i] << 1) | (residual[i] >= 0)
        randomizedHadamard(residual.data(), qjl_signs_precomp_.data(), dim_);
        uint8_t *out = code.sq_packed_;
        if (packed_nibbles_) {
            // Two nibbles per byte; sq_idx is in 0..7 (3 bits) so nibble fits 4 bits.
            std::memset(out, 0, packed_bytes_);
            for (size_t i = 0; i < dim_; ++i) {
                uint8_t unit = static_cast<uint8_t>(
                    (sq_idx[i] << 1) | ((residual[i] >= 0.0f) ? 1u : 0u));
                if ((i & 1) == 0)
                    out[i >> 1] = unit;             // low nibble
                else
                    out[i >> 1] |= unit << 4;       // high nibble
            }
        } else {
            for (size_t i = 0; i < dim_; ++i) {
                out[i] = static_cast<uint8_t>(
                    (sq_idx[i] << 1) | ((residual[i] >= 0.0f) ? 1u : 0u));
            }
        }

        // Meta
        code.setNorm(norm);
        code.setGamma(gamma);
        code.setSigma(sigma);
    }

    inline uint8_t quantize(const float val) const {
        uint8_t idx = 0;
        for (uint16_t i = 0; i < num_boundaries_; ++i) {
            idx += (val > boundaries_[i]); // branchless accumulation
        }
        return idx;
    }

    // -- Mode switching -------------------------------------------------------
    //
    // Two-phase workflow:
    //   1. Build: use setBuildMode(hnsw) — symmetric distance (code × code).
    //   2. Search: call setSearchMode(hnsw) once after build completes.
    //      Then each thread does:
    //        auto pq = space.prepareQuery(raw_query);
    //        auto result = hnsw.searchKnn(&pq, K);
    //      No shared mutable state — fully thread-safe.

    /// Switch HNSW to search mode: asymmetric distance (PreparedQuery* × code).
    void setSearchMode(HierarchicalNSW<float> &hnsw) {
        hnsw.fstdistfunc_ = fstdistfunc_search_;
        hnsw.dist_func_param_ = this;
    }

    /// Switch HNSW to build mode: symmetric distance (code × code).
    void setBuildMode(HierarchicalNSW<float> &hnsw) {
        hnsw.fstdistfunc_ = fstdistfunc_build_;
        hnsw.dist_func_param_ = this;
    }

    // -- Query preparation ----------------------------------------------------

    /// Prepare query: amortizes 2 RHT calls across all distance computations.
    /// Result lives on the caller's stack — no shared state.
    TurboQuantPreparedQuery prepareQuery(const float *raw_query) const {
        TurboQuantPreparedQuery pq;
        const size_t d = dim_;

        // Compute query norm
        pq.q_norm_sq = 0.0f;
        for (size_t i = 0; i < d; ++i)
            pq.q_norm_sq += raw_query[i] * raw_query[i];
        pq.q_norm = std::sqrt(pq.q_norm_sq);
        float q_inv = (pq.q_norm > 1e-10f) ? (1.0f / pq.q_norm) : 0.0f;

        // Normalize and apply PolarQuant rotation
        pq.q_rot.resize(d);
        for (size_t i = 0; i < d; ++i)
            pq.q_rot[i] = raw_query[i] * q_inv;
        randomizedHadamard(pq.q_rot.data(), rotation_signs_.data(), d);

        // Store centroids pointer for direct compute in distSearch
        pq.centroids = centroids_;

        // QJL projection of query for correction term
        pq.s_q = pq.q_rot;
        randomizedHadamard(pq.s_q.data(), qjl_signs_precomp_.data(), d);

        return pq;
    }
};

// ===========================================================================
// Distance function implementations (scalar + SIMD)
//
// HNSW buffer layout (interleaved):
//   [packed: dim bytes] [meta: 3 floats (norm, gamma, sigma)]
//   packed[i] = (sq_idx << 1) | qjl_bit
//     sq_idx:  centroid index for Lloyd-Max SQ (upper 7 or 3 bits)
//     qjl_bit: QJL correction sign (bit 0): 1 = positive, 0 = negative
//
// distSearch: PreparedQuery × code (asymmetric, used for search)
// distBuild:  code × code (symmetric, SQ-only, used for graph construction)
//
// Defined after TurboQuantSpace so they can access its members.
// Each variant has the same signature; the constructor selects one at runtime.
// ===========================================================================

// ---------------------------------------------------------------------------
// Scalar fallback
// ---------------------------------------------------------------------------

static float distSearchScalar(const void *q, const void *code_buf,
                              const void *qty_ptr) {
  const auto *space = static_cast<const TurboQuantSpace *>(qty_ptr);
  const auto *pq = static_cast<const TurboQuantPreparedQuery *>(q);
  const size_t dim = space->dim();
  const uint8_t *packed = reinterpret_cast<const uint8_t *>(code_buf);
  const float *meta = reinterpret_cast<const float *>(
      static_cast<const char *>(code_buf) + dim);
  const float x_norm = meta[0];
  const float gamma = meta[1];
  const float sigma = meta[2];

  // Fused loop: extract sq_idx and qjl_bit from interleaved byte
  // byte[i] = (sq_idx << 1) | qjl_bit; qjl_bit=1 → +1, qjl_bit=0 → -1
  float ip_mse = 0.0f;
  float dot_qjl = 0.0f;
  const float *centroids = pq->centroids;
  const float *q_rot = pq->q_rot.data();
  const float *s_q = pq->s_q.data();
  for (size_t i = 0; i < dim; ++i) {
    uint8_t byte = packed[i];
    ip_mse += q_rot[i] * centroids[byte >> 1];
    float sign = (byte & 1) ? 1.0f : -1.0f;
    dot_qjl += s_q[i] * sign;
  }
  ip_mse *= sigma;
  const float correction = space->scale() * gamma * dot_qjl;

  // L2 = ||q||² + ||x||² - 2·IP
  const float ip = (ip_mse + correction) * x_norm * pq->q_norm;
  return std::max(0.0f, pq->q_norm_sq + x_norm * x_norm - 2.0f * ip);
}

static float distBuildScalar(const void *pVect1, const void *pVect2,
                             const void *param_ptr) {
  const auto *space = static_cast<const TurboQuantSpace *>(param_ptr);
  const size_t dim = space->dim();
  const float *centroids = space->centroids();

  const char *buf_a = static_cast<const char *>(pVect1);
  const char *buf_b = static_cast<const char *>(pVect2);

  const uint8_t *packed_a = reinterpret_cast<const uint8_t *>(buf_a);
  const float *meta_a = reinterpret_cast<const float *>(buf_a + dim);
  const float norm_a = meta_a[0];
  const float sigma_a = meta_a[2];

  const uint8_t *packed_b = reinterpret_cast<const uint8_t *>(buf_b);
  const float *meta_b = reinterpret_cast<const float *>(buf_b + dim);
  const float norm_b = meta_b[0];
  const float sigma_b = meta_b[2];

  float ip_rot = 0.0f;
  for (size_t i = 0; i < dim; ++i)
    ip_rot += (centroids[packed_a[i] >> 1] * sigma_a) *
              (centroids[packed_b[i] >> 1] * sigma_b);

  const float ip = ip_rot * norm_a * norm_b;
  return std::max(0.0f, norm_a * norm_a + norm_b * norm_b - 2.0f * ip);
}

// ---------------------------------------------------------------------------
// Packed-nibble (b<=4) scalar variants
//   byte[i] holds two 4-bit units: low=coord 2i, high=coord 2i+1
//   unit = (sq_idx << 1) | qjl_bit;  sq_idx in 0..7
// ---------------------------------------------------------------------------

static float distSearchScalarB4(const void *q, const void *code_buf,
                                const void *qty_ptr) {
  const auto *space = static_cast<const TurboQuantSpace *>(qty_ptr);
  const auto *pq = static_cast<const TurboQuantPreparedQuery *>(q);
  const size_t dim = space->dim();
  const size_t packed_bytes = space->packedBytes();
  const uint8_t *packed = reinterpret_cast<const uint8_t *>(code_buf);
  const float *meta = reinterpret_cast<const float *>(
      static_cast<const char *>(code_buf) + packed_bytes);
  const float x_norm = meta[0];
  const float gamma = meta[1];
  const float sigma = meta[2];

  const float *centroids = pq->centroids;
  const float *q_rot = pq->q_rot.data();
  const float *s_q = pq->s_q.data();

  float ip_mse = 0.0f;
  float dot_qjl = 0.0f;
  // dim is a power of two and >=4, so it's always even
  for (size_t i = 0, b = 0; i < dim; i += 2, ++b) {
    uint8_t byte = packed[b];
    uint8_t lo = byte & 0x0F;
    uint8_t hi = byte >> 4;
    ip_mse += q_rot[i]     * centroids[lo >> 1];
    ip_mse += q_rot[i + 1] * centroids[hi >> 1];
    dot_qjl += s_q[i]     * ((lo & 1) ? 1.0f : -1.0f);
    dot_qjl += s_q[i + 1] * ((hi & 1) ? 1.0f : -1.0f);
  }
  ip_mse *= sigma;
  const float correction = space->scale() * gamma * dot_qjl;
  const float ip = (ip_mse + correction) * x_norm * pq->q_norm;
  return std::max(0.0f, pq->q_norm_sq + x_norm * x_norm - 2.0f * ip);
}

static float distBuildScalarB4(const void *pVect1, const void *pVect2,
                               const void *param_ptr) {
  const auto *space = static_cast<const TurboQuantSpace *>(param_ptr);
  const size_t dim = space->dim();
  const size_t packed_bytes = space->packedBytes();
  const float *centroids = space->centroids();

  const char *buf_a = static_cast<const char *>(pVect1);
  const char *buf_b = static_cast<const char *>(pVect2);

  const uint8_t *packed_a = reinterpret_cast<const uint8_t *>(buf_a);
  const float *meta_a = reinterpret_cast<const float *>(buf_a + packed_bytes);
  const float norm_a = meta_a[0];
  const float sigma_a = meta_a[2];

  const uint8_t *packed_b = reinterpret_cast<const uint8_t *>(buf_b);
  const float *meta_b = reinterpret_cast<const float *>(buf_b + packed_bytes);
  const float norm_b = meta_b[0];
  const float sigma_b = meta_b[2];

  float ip_rot = 0.0f;
  for (size_t i = 0, bi = 0; i < dim; i += 2, ++bi) {
    uint8_t ba = packed_a[bi];
    uint8_t bb = packed_b[bi];
    uint8_t lo_a = (ba & 0x0F) >> 1;
    uint8_t hi_a = (ba >> 4)  >> 1;
    uint8_t lo_b = (bb & 0x0F) >> 1;
    uint8_t hi_b = (bb >> 4)  >> 1;
    ip_rot += (centroids[lo_a] * sigma_a) * (centroids[lo_b] * sigma_b);
    ip_rot += (centroids[hi_a] * sigma_a) * (centroids[hi_b] * sigma_b);
  }

  const float ip = ip_rot * norm_a * norm_b;
  return std::max(0.0f, norm_a * norm_a + norm_b * norm_b - 2.0f * ip);
}

// ---------------------------------------------------------------------------
// NEON implementation (ARM, 4 floats per iteration)
// ---------------------------------------------------------------------------
#if defined(USE_NEON)

static float distSearchNEON(const void *q, const void *code_buf,
                            const void *qty_ptr) {
  const auto *space = static_cast<const TurboQuantSpace *>(qty_ptr);
  const auto *pq = static_cast<const TurboQuantPreparedQuery *>(q);
  const size_t dim = space->dim();
  const uint8_t *packed = reinterpret_cast<const uint8_t *>(code_buf);
  const float *meta = reinterpret_cast<const float *>(
      static_cast<const char *>(code_buf) + dim);
  const float x_norm = meta[0];
  const float gamma = meta[1];
  const float sigma = meta[2];

  const float *centroids = pq->centroids;
  const float *q_rot = pq->q_rot.data();
  const float *s_q = pq->s_q.data();

  // Sign-flip mask: XOR with this flips the sign bit of a float
  const uint32x4_t sign_bit = vdupq_n_u32(0x80000000u);
  // Mask for extracting qjl_bit (bit 0)
  const uint8x8_t one_u8 = vdup_n_u8(1);

  float32x4_t sum_ip0 = vdupq_n_f32(0.0f);
  float32x4_t sum_ip1 = vdupq_n_f32(0.0f);
  float32x4_t sum_qjl0 = vdupq_n_f32(0.0f);
  float32x4_t sum_qjl1 = vdupq_n_f32(0.0f);

  size_t i = 0;
  for (; i + 8 <= dim; i += 8) {
    // Extract sq_idx = byte >> 1, gather centroids
    float c0[4], c1[4];
    c0[0] = centroids[packed[i] >> 1];
    c0[1] = centroids[packed[i + 1] >> 1];
    c0[2] = centroids[packed[i + 2] >> 1];
    c0[3] = centroids[packed[i + 3] >> 1];
    c1[0] = centroids[packed[i + 4] >> 1];
    c1[1] = centroids[packed[i + 5] >> 1];
    c1[2] = centroids[packed[i + 6] >> 1];
    c1[3] = centroids[packed[i + 7] >> 1];
    sum_ip0 = vmlaq_f32(sum_ip0, vld1q_f32(q_rot + i), vld1q_f32(c0));
    sum_ip1 = vmlaq_f32(sum_ip1, vld1q_f32(q_rot + i + 4), vld1q_f32(c1));

    // QJL: extract bit 0 → neg_mask → conditionally flip sign of s_q
    //   bit=1 → positive (+1, keep s_q), bit=0 → negative (-1, flip s_q)
    uint8x8_t bytes = vld1_u8(packed + i);
    uint8x8_t bits = vand_u8(bytes, one_u8); // 0 or 1
    int8x8_t signs = vsub_s8(vreinterpret_s8_u8(vadd_u8(bits, bits)),
                             vdup_n_s8(1)); // 2*bit-1 → ±1
    // Widen ±1 to int32 lanes, build neg_mask, XOR to flip sign of s_q
    int16x8_t s16 = vmovl_s8(signs);
    uint32x4_t neg0 = vcltq_s32(vmovl_s16(vget_low_s16(s16)), vdupq_n_s32(0));
    uint32x4_t neg1 = vcltq_s32(vmovl_s16(vget_high_s16(s16)), vdupq_n_s32(0));

    sum_qjl0 =
        vaddq_f32(sum_qjl0, vreinterpretq_f32_u32(veorq_u32(
                                vreinterpretq_u32_f32(vld1q_f32(s_q + i)),
                                vandq_u32(neg0, sign_bit))));
    sum_qjl1 =
        vaddq_f32(sum_qjl1, vreinterpretq_f32_u32(veorq_u32(
                                vreinterpretq_u32_f32(vld1q_f32(s_q + i + 4)),
                                vandq_u32(neg1, sign_bit))));
  }

  for (; i + 4 <= dim; i += 4) {
    float cv[4];
    cv[0] = centroids[packed[i] >> 1];
    cv[1] = centroids[packed[i + 1] >> 1];
    cv[2] = centroids[packed[i + 2] >> 1];
    cv[3] = centroids[packed[i + 3] >> 1];
    sum_ip0 = vmlaq_f32(sum_ip0, vld1q_f32(q_rot + i), vld1q_f32(cv));

    // Scalar QJL for remainder
    float sq_arr[4];
    for (int j = 0; j < 4; ++j)
      sq_arr[j] = s_q[i + j] * ((packed[i + j] & 1) ? 1.0f : -1.0f);
    sum_qjl0 = vaddq_f32(sum_qjl0, vld1q_f32(sq_arr));
  }

  float ip_mse = vaddvq_f32(vaddq_f32(sum_ip0, sum_ip1)) * sigma;
  float dot_qjl = vaddvq_f32(vaddq_f32(sum_qjl0, sum_qjl1));

  const float correction = space->scale() * gamma * dot_qjl;
  const float ip = (ip_mse + correction) * x_norm * pq->q_norm;
  return std::max(0.0f, pq->q_norm_sq + x_norm * x_norm - 2.0f * ip);
}

static float distBuildNEON(const void *pVect1, const void *pVect2,
                           const void *param_ptr) {
  const auto *space = static_cast<const TurboQuantSpace *>(param_ptr);
  const size_t dim = space->dim();
  const float *centroids = space->centroids();

  const char *buf_a = static_cast<const char *>(pVect1);
  const char *buf_b = static_cast<const char *>(pVect2);

  const uint8_t *packed_a = reinterpret_cast<const uint8_t *>(buf_a);
  const float *meta_a = reinterpret_cast<const float *>(buf_a + dim);
  const float norm_a = meta_a[0];
  const float sigma_a = meta_a[2];

  const uint8_t *packed_b = reinterpret_cast<const uint8_t *>(buf_b);
  const float *meta_b = reinterpret_cast<const float *>(buf_b + dim);
  const float norm_b = meta_b[0];
  const float sigma_b = meta_b[2];

  const float32x4_t v_sigma_a = vdupq_n_f32(sigma_a);
  const float32x4_t v_sigma_b = vdupq_n_f32(sigma_b);
  float32x4_t sum = vdupq_n_f32(0.0f);

  size_t i = 0;
  for (; i + 4 <= dim; i += 4) {
    float ca[4], cb[4];
    ca[0] = centroids[packed_a[i] >> 1];
    cb[0] = centroids[packed_b[i] >> 1];
    ca[1] = centroids[packed_a[i + 1] >> 1];
    cb[1] = centroids[packed_b[i + 1] >> 1];
    ca[2] = centroids[packed_a[i + 2] >> 1];
    cb[2] = centroids[packed_b[i + 2] >> 1];
    ca[3] = centroids[packed_a[i + 3] >> 1];
    cb[3] = centroids[packed_b[i + 3] >> 1];

    float32x4_t va = vmulq_f32(vld1q_f32(ca), v_sigma_a);
    float32x4_t vb = vmulq_f32(vld1q_f32(cb), v_sigma_b);
    sum = vmlaq_f32(sum, va, vb);
  }

  float ip_rot = vaddvq_f32(sum);

  for (; i < dim; ++i)
    ip_rot += (centroids[packed_a[i] >> 1] * sigma_a) *
              (centroids[packed_b[i] >> 1] * sigma_b);

  const float ip = ip_rot * norm_a * norm_b;
  return std::max(0.0f, norm_a * norm_a + norm_b * norm_b - 2.0f * ip);
}

// ---------------------------------------------------------------------------
// NEON b=4 (packed nibble) — tbl-based centroid gather
//
// LUT: centroids[] has 8 float32 entries = 32 bytes = uint8x16x2_t.
// Per iteration we process 8 coords (4 packed bytes):
//   - load 4 bytes, split into low/high nibbles, zip-interleave to 8 units
//   - sq_idx = unit >> 1  (0..7)
//   - qjl_bit = unit & 1
//   - centroid gather via vqtbl2q_u8 with byte-level indices
//     (each float needs 4 consecutive bytes: sq*4, sq*4+1, sq*4+2, sq*4+3)
// ---------------------------------------------------------------------------

static inline uint8x16x2_t loadCentroidLUTb4(const float *centroids) {
  // centroids points to 8 floats (32 bytes).
  uint8x16x2_t lut;
  lut.val[0] = vld1q_u8(reinterpret_cast<const uint8_t *>(centroids));
  lut.val[1] = vld1q_u8(reinterpret_cast<const uint8_t *>(centroids) + 16);
  return lut;
}

// Build a 16-byte index vector that, when passed to vqtbl2q_u8 with the
// centroid LUT, yields 4 consecutive float32 values corresponding to the
// sq_idx values in the low 4 lanes of `sq4` (uint8x8_t).
static inline uint8x16_t buildByteIdx4(uint8x8_t sq4) {
  // {0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3}
  static const uint8_t dup_tbl[16] = {
      0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
  // {0,1,2,3, 0,1,2,3, 0,1,2,3, 0,1,2,3}
  static const uint8_t off_tbl[16] = {
      0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};

  uint8x16_t dup_idx = vld1q_u8(dup_tbl);
  uint8x16_t off = vld1q_u8(off_tbl);

  // Put sq4 (8 bytes) into the low half of a 16-byte vector.
  uint8x16_t sq16 = vcombine_u8(sq4, vdup_n_u8(0));
  // Duplicate each of the first 4 bytes 4 times.
  uint8x16_t sq_dup = vqtbl1q_u8(sq16, dup_idx);
  // byte_idx = sq_dup * 4 + off
  return vaddq_u8(vshlq_n_u8(sq_dup, 2), off);
}

static float distSearchNEONB4(const void *q, const void *code_buf,
                              const void *qty_ptr) {
  const auto *space = static_cast<const TurboQuantSpace *>(qty_ptr);
  const auto *pq = static_cast<const TurboQuantPreparedQuery *>(q);
  const size_t dim = space->dim();
  const size_t packed_bytes = space->packedBytes();
  const uint8_t *packed = reinterpret_cast<const uint8_t *>(code_buf);
  const float *meta = reinterpret_cast<const float *>(
      static_cast<const char *>(code_buf) + packed_bytes);
  const float x_norm = meta[0];
  const float gamma = meta[1];
  const float sigma = meta[2];

  const float *centroids = pq->centroids;
  const float *q_rot = pq->q_rot.data();
  const float *s_q = pq->s_q.data();

  const uint8x16x2_t lut = loadCentroidLUTb4(centroids);
  const uint32x4_t sign_bit = vdupq_n_u32(0x80000000u);
  const uint8x8_t mask_lo = vdup_n_u8(0x0F);
  const uint8x8_t mask_1 = vdup_n_u8(0x01);

  float32x4_t sum_ip0 = vdupq_n_f32(0.0f);
  float32x4_t sum_ip1 = vdupq_n_f32(0.0f);
  float32x4_t sum_qjl0 = vdupq_n_f32(0.0f);
  float32x4_t sum_qjl1 = vdupq_n_f32(0.0f);

  // 8 coords per iteration = 4 packed bytes
  size_t i = 0;
  for (; i + 8 <= dim; i += 8) {
    // Load 4 packed bytes (remaining lanes don't matter).
    uint8x8_t bytes = vld1_u8(packed + (i >> 1));
    uint8x8_t lo = vand_u8(bytes, mask_lo);
    uint8x8_t hi = vshr_n_u8(bytes, 4);
    // Interleave lo/hi so lane order matches coord order (lo[0],hi[0],lo[1],hi[1],...)
    uint8x8_t units = vzip1_u8(lo, hi);          // 8 units
    uint8x8_t sq = vshr_n_u8(units, 1);          // sq_idx in 0..7
    uint8x8_t qjl = vand_u8(units, mask_1);      // qjl_bit in {0,1}

    // Split sq into two 4-lane halves and gather 4 centroids per tbl call.
    uint8x8_t sq_lo4 = sq;                       // lanes 0..3 used for first gather
    uint8x8_t sq_hi4 = vext_u8(sq, sq, 4);       // lanes 4..7 -> low 4

    uint8x16_t idx0 = buildByteIdx4(sq_lo4);
    uint8x16_t idx1 = buildByteIdx4(sq_hi4);
    float32x4_t c0 = vreinterpretq_f32_u8(vqtbl2q_u8(lut, idx0));
    float32x4_t c1 = vreinterpretq_f32_u8(vqtbl2q_u8(lut, idx1));

    sum_ip0 = vmlaq_f32(sum_ip0, vld1q_f32(q_rot + i), c0);
    sum_ip1 = vmlaq_f32(sum_ip1, vld1q_f32(q_rot + i + 4), c1);

    // QJL sign fold: bit=1 -> +s_q, bit=0 -> -s_q
    int8x8_t signs = vsub_s8(vreinterpret_s8_u8(vadd_u8(qjl, qjl)),
                             vdup_n_s8(1)); // 2*bit - 1
    int16x8_t s16 = vmovl_s8(signs);
    uint32x4_t neg0 = vcltq_s32(vmovl_s16(vget_low_s16(s16)), vdupq_n_s32(0));
    uint32x4_t neg1 = vcltq_s32(vmovl_s16(vget_high_s16(s16)), vdupq_n_s32(0));
    sum_qjl0 =
        vaddq_f32(sum_qjl0, vreinterpretq_f32_u32(veorq_u32(
                                vreinterpretq_u32_f32(vld1q_f32(s_q + i)),
                                vandq_u32(neg0, sign_bit))));
    sum_qjl1 =
        vaddq_f32(sum_qjl1, vreinterpretq_f32_u32(veorq_u32(
                                vreinterpretq_u32_f32(vld1q_f32(s_q + i + 4)),
                                vandq_u32(neg1, sign_bit))));
  }

  float ip_mse = vaddvq_f32(vaddq_f32(sum_ip0, sum_ip1));
  float dot_qjl = vaddvq_f32(vaddq_f32(sum_qjl0, sum_qjl1));

  // Scalar remainder (dim is power-of-two, but keep for completeness).
  for (; i < dim; ++i) {
    uint8_t byte = packed[i >> 1];
    uint8_t unit = (i & 1) ? (byte >> 4) : (byte & 0x0F);
    ip_mse += q_rot[i] * centroids[unit >> 1];
    dot_qjl += s_q[i] * ((unit & 1) ? 1.0f : -1.0f);
  }

  ip_mse *= sigma;
  const float correction = space->scale() * gamma * dot_qjl;
  const float ip = (ip_mse + correction) * x_norm * pq->q_norm;
  return std::max(0.0f, pq->q_norm_sq + x_norm * x_norm - 2.0f * ip);
}

static float distBuildNEONB4(const void *pVect1, const void *pVect2,
                             const void *param_ptr) {
  const auto *space = static_cast<const TurboQuantSpace *>(param_ptr);
  const size_t dim = space->dim();
  const size_t packed_bytes = space->packedBytes();
  const float *centroids = space->centroids();

  const char *buf_a = static_cast<const char *>(pVect1);
  const char *buf_b = static_cast<const char *>(pVect2);

  const uint8_t *packed_a = reinterpret_cast<const uint8_t *>(buf_a);
  const float *meta_a = reinterpret_cast<const float *>(buf_a + packed_bytes);
  const float norm_a = meta_a[0];
  const float sigma_a = meta_a[2];

  const uint8_t *packed_b = reinterpret_cast<const uint8_t *>(buf_b);
  const float *meta_b = reinterpret_cast<const float *>(buf_b + packed_bytes);
  const float norm_b = meta_b[0];
  const float sigma_b = meta_b[2];

  const uint8x16x2_t lut = loadCentroidLUTb4(centroids);
  const float32x4_t v_sigma_a = vdupq_n_f32(sigma_a);
  const float32x4_t v_sigma_b = vdupq_n_f32(sigma_b);
  const uint8x8_t mask_lo = vdup_n_u8(0x0F);

  float32x4_t sum = vdupq_n_f32(0.0f);

  size_t i = 0;
  for (; i + 8 <= dim; i += 8) {
    uint8x8_t ba = vld1_u8(packed_a + (i >> 1));
    uint8x8_t bb = vld1_u8(packed_b + (i >> 1));

    uint8x8_t lo_a = vand_u8(ba, mask_lo);
    uint8x8_t hi_a = vshr_n_u8(ba, 4);
    uint8x8_t lo_b = vand_u8(bb, mask_lo);
    uint8x8_t hi_b = vshr_n_u8(bb, 4);

    uint8x8_t units_a = vzip1_u8(lo_a, hi_a);
    uint8x8_t units_b = vzip1_u8(lo_b, hi_b);
    uint8x8_t sq_a = vshr_n_u8(units_a, 1);
    uint8x8_t sq_b = vshr_n_u8(units_b, 1);

    uint8x8_t sq_a_hi = vext_u8(sq_a, sq_a, 4);
    uint8x8_t sq_b_hi = vext_u8(sq_b, sq_b, 4);

    float32x4_t ca0 = vreinterpretq_f32_u8(vqtbl2q_u8(lut, buildByteIdx4(sq_a)));
    float32x4_t ca1 = vreinterpretq_f32_u8(vqtbl2q_u8(lut, buildByteIdx4(sq_a_hi)));
    float32x4_t cb0 = vreinterpretq_f32_u8(vqtbl2q_u8(lut, buildByteIdx4(sq_b)));
    float32x4_t cb1 = vreinterpretq_f32_u8(vqtbl2q_u8(lut, buildByteIdx4(sq_b_hi)));

    float32x4_t va0 = vmulq_f32(ca0, v_sigma_a);
    float32x4_t va1 = vmulq_f32(ca1, v_sigma_a);
    float32x4_t vb0 = vmulq_f32(cb0, v_sigma_b);
    float32x4_t vb1 = vmulq_f32(cb1, v_sigma_b);

    sum = vmlaq_f32(sum, va0, vb0);
    sum = vmlaq_f32(sum, va1, vb1);
  }

  float ip_rot = vaddvq_f32(sum);
  for (; i < dim; ++i) {
    uint8_t ba = packed_a[i >> 1];
    uint8_t bb = packed_b[i >> 1];
    uint8_t ua = (i & 1) ? (ba >> 4) : (ba & 0x0F);
    uint8_t ub = (i & 1) ? (bb >> 4) : (bb & 0x0F);
    ip_rot += (centroids[ua >> 1] * sigma_a) * (centroids[ub >> 1] * sigma_b);
  }

  const float ip = ip_rot * norm_a * norm_b;
  return std::max(0.0f, norm_a * norm_a + norm_b * norm_b - 2.0f * ip);
}

#endif // USE_NEON

// ---------------------------------------------------------------------------
// SSE implementation (4 floats per iteration)
// ---------------------------------------------------------------------------
#if defined(USE_SSE)

static float distSearchSSE(const void *q, const void *code_buf,
                           const void *qty_ptr) {
  const auto *space = static_cast<const TurboQuantSpace *>(qty_ptr);
  const auto *pq = static_cast<const TurboQuantPreparedQuery *>(q);
  const size_t dim = space->dim();
  const uint8_t *packed = reinterpret_cast<const uint8_t *>(code_buf);
  const float *meta = reinterpret_cast<const float *>(
      static_cast<const char *>(code_buf) + dim);
  const float x_norm = meta[0];
  const float gamma = meta[1];
  const float sigma = meta[2];

  const float *centroids = pq->centroids;
  const float *q_rot = pq->q_rot.data();
  const float *s_q = pq->s_q.data();

  const __m128i sign_bit = _mm_set1_epi32(static_cast<int>(0x80000000u));

  __m128 sum_ip = _mm_setzero_ps();
  __m128 sum_qjl = _mm_setzero_ps();

  size_t i = 0;
  for (; i + 4 <= dim; i += 4) {
    // Centroid gather from interleaved: sq_idx = byte >> 1
    float cv[4];
    cv[0] = centroids[packed[i] >> 1];
    cv[1] = centroids[packed[i + 1] >> 1];
    cv[2] = centroids[packed[i + 2] >> 1];
    cv[3] = centroids[packed[i + 3] >> 1];
    sum_ip = _mm_add_ps(sum_ip,
                        _mm_mul_ps(_mm_loadu_ps(q_rot + i), _mm_loadu_ps(cv)));

    // QJL: extract bit 0 → neg_mask → conditionally flip sign of s_q
    //   bit=1 → positive (+1, keep s_q), bit=0 → negative (-1, flip s_q)
    __m128i bytes =
        _mm_cvtsi32_si128(*reinterpret_cast<const int32_t *>(packed + i));
    bytes = _mm_unpacklo_epi8(bytes, bytes);  // 8→16 bit
    bytes = _mm_unpacklo_epi16(bytes, bytes); // 16→32 bit (one byte per lane)
    __m128i bit0 = _mm_and_si128(bytes, _mm_set1_epi32(1));
    __m128i neg_mask = _mm_cmpeq_epi32(bit0, _mm_setzero_si128());

    __m128 vsq = _mm_loadu_ps(s_q + i);
    __m128i vsq_i =
        _mm_xor_si128(_mm_castps_si128(vsq), _mm_and_si128(neg_mask, sign_bit));
    sum_qjl = _mm_add_ps(sum_qjl, _mm_castsi128_ps(vsq_i));
  }

  float PORTABLE_ALIGN32 tmp[4];

  _mm_store_ps(tmp, sum_ip);
  float ip_mse = (tmp[0] + tmp[1] + tmp[2] + tmp[3]) * sigma;

  _mm_store_ps(tmp, sum_qjl);
  float dot_qjl = tmp[0] + tmp[1] + tmp[2] + tmp[3];

  for (; i < dim; ++i) {
    ip_mse += q_rot[i] * centroids[packed[i] >> 1] * sigma;
    dot_qjl += s_q[i] * ((packed[i] & 1) ? 1.0f : -1.0f);
  }

  const float correction = space->scale() * gamma * dot_qjl;
  const float ip = (ip_mse + correction) * x_norm * pq->q_norm;
  return std::max(0.0f, pq->q_norm_sq + x_norm * x_norm - 2.0f * ip);
}

static float distBuildSSE(const void *pVect1, const void *pVect2,
                          const void *param_ptr) {
  const auto *space = static_cast<const TurboQuantSpace *>(param_ptr);
  const size_t dim = space->dim();
  const float *centroids = space->centroids();

  const char *buf_a = static_cast<const char *>(pVect1);
  const char *buf_b = static_cast<const char *>(pVect2);

  const uint8_t *packed_a = reinterpret_cast<const uint8_t *>(buf_a);
  const float *meta_a = reinterpret_cast<const float *>(buf_a + dim);
  const float norm_a = meta_a[0];
  const float sigma_a = meta_a[2];

  const uint8_t *packed_b = reinterpret_cast<const uint8_t *>(buf_b);
  const float *meta_b = reinterpret_cast<const float *>(buf_b + dim);
  const float norm_b = meta_b[0];
  const float sigma_b = meta_b[2];

  const __m128 v_sigma_a = _mm_set1_ps(sigma_a);
  const __m128 v_sigma_b = _mm_set1_ps(sigma_b);
  __m128 sum = _mm_setzero_ps();

  size_t i = 0;
  for (; i + 4 <= dim; i += 4) {
    float ca[4], cb[4];
    ca[0] = centroids[packed_a[i] >> 1];
    cb[0] = centroids[packed_b[i] >> 1];
    ca[1] = centroids[packed_a[i + 1] >> 1];
    cb[1] = centroids[packed_b[i + 1] >> 1];
    ca[2] = centroids[packed_a[i + 2] >> 1];
    cb[2] = centroids[packed_b[i + 2] >> 1];
    ca[3] = centroids[packed_a[i + 3] >> 1];
    cb[3] = centroids[packed_b[i + 3] >> 1];

    __m128 va = _mm_mul_ps(_mm_loadu_ps(ca), v_sigma_a);
    __m128 vb = _mm_mul_ps(_mm_loadu_ps(cb), v_sigma_b);
    sum = _mm_add_ps(sum, _mm_mul_ps(va, vb));
  }

  float PORTABLE_ALIGN32 tmp[4];
  _mm_store_ps(tmp, sum);
  float ip_rot = tmp[0] + tmp[1] + tmp[2] + tmp[3];

  for (; i < dim; ++i)
    ip_rot += (centroids[packed_a[i] >> 1] * sigma_a) *
              (centroids[packed_b[i] >> 1] * sigma_b);

  const float ip = ip_rot * norm_a * norm_b;
  return std::max(0.0f, norm_a * norm_a + norm_b * norm_b - 2.0f * ip);
}

#endif // USE_SSE

// ---------------------------------------------------------------------------
// AVX2 implementation (8 floats per iteration, uses gather)
// ---------------------------------------------------------------------------
#if defined(USE_AVX)

static float distSearchAVX(const void *q, const void *code_buf,
                           const void *qty_ptr) {
  const auto *space = static_cast<const TurboQuantSpace *>(qty_ptr);
  const auto *pq = static_cast<const TurboQuantPreparedQuery *>(q);
  const size_t dim = space->dim();
  const uint8_t *packed = reinterpret_cast<const uint8_t *>(code_buf);
  const float *meta = reinterpret_cast<const float *>(
      static_cast<const char *>(code_buf) + dim);
  const float x_norm = meta[0];
  const float gamma = meta[1];
  const float sigma = meta[2];

  const float *centroids = pq->centroids;
  const float *q_rot = pq->q_rot.data();
  const float *s_q = pq->s_q.data();

  const __m256i sign_bit = _mm256_set1_epi32(static_cast<int>(0x80000000u));

  __m256 sum_ip = _mm256_setzero_ps();
  __m256 sum_qjl = _mm256_setzero_ps();

  size_t i = 0;
  for (; i + 8 <= dim; i += 8) {
    // Centroid gather: sq_idx = byte >> 1
#ifdef __AVX2__
    __m256i idx = _mm256_set_epi32(packed[i + 7] >> 1, packed[i + 6] >> 1,
                                   packed[i + 5] >> 1, packed[i + 4] >> 1,
                                   packed[i + 3] >> 1, packed[i + 2] >> 1,
                                   packed[i + 1] >> 1, packed[i] >> 1);
    __m256 vc = _mm256_i32gather_ps(centroids, idx, 4);
#else
    __m256 vc = _mm256_set_ps(
        centroids[packed[i + 7] >> 1], centroids[packed[i + 6] >> 1],
        centroids[packed[i + 5] >> 1], centroids[packed[i + 4] >> 1],
        centroids[packed[i + 3] >> 1], centroids[packed[i + 2] >> 1],
        centroids[packed[i + 1] >> 1], centroids[packed[i] >> 1]);
#endif
    sum_ip =
        _mm256_add_ps(sum_ip, _mm256_mul_ps(_mm256_loadu_ps(q_rot + i), vc));

    // QJL: extract bit 0 → neg_mask → conditionally flip sign of s_q
    //   bit=1 → positive (+1, keep s_q), bit=0 → negative (-1, flip s_q)
    __m256 vsq = _mm256_loadu_ps(s_q + i);

#ifdef __AVX2__
    __m128i bytes8 =
        _mm_loadl_epi64(reinterpret_cast<const __m128i *>(packed + i));
    __m256i bytes32 = _mm256_cvtepu8_epi32(bytes8);
    __m256i bit0 = _mm256_and_si256(bytes32, _mm256_set1_epi32(1));
    __m256i neg_mask = _mm256_cmpeq_epi32(bit0, _mm256_setzero_si256());
#else
    __m256i neg_mask = _mm256_set_epi32(
        (packed[i + 7] & 1) ? 0 : -1, (packed[i + 6] & 1) ? 0 : -1,
        (packed[i + 5] & 1) ? 0 : -1, (packed[i + 4] & 1) ? 0 : -1,
        (packed[i + 3] & 1) ? 0 : -1, (packed[i + 2] & 1) ? 0 : -1,
        (packed[i + 1] & 1) ? 0 : -1, (packed[i] & 1) ? 0 : -1);
#endif

    __m256i vsq_i = _mm256_xor_si256(_mm256_castps_si256(vsq),
                                     _mm256_and_si256(neg_mask, sign_bit));
    sum_qjl = _mm256_add_ps(sum_qjl, _mm256_castsi256_ps(vsq_i));
  }

  // Horizontal sum
  __m128 hi_ip = _mm256_extractf128_ps(sum_ip, 1);
  __m128 lo_ip = _mm256_castps256_ps128(sum_ip);
  __m128 s_ip = _mm_add_ps(lo_ip, hi_ip);
  s_ip = _mm_add_ps(s_ip, _mm_movehl_ps(s_ip, s_ip));
  s_ip = _mm_add_ss(s_ip, _mm_shuffle_ps(s_ip, s_ip, 1));
  float ip_mse = _mm_cvtss_f32(s_ip) * sigma;

  __m128 hi_qjl = _mm256_extractf128_ps(sum_qjl, 1);
  __m128 lo_qjl = _mm256_castps256_ps128(sum_qjl);
  __m128 s_qjl = _mm_add_ps(lo_qjl, hi_qjl);
  s_qjl = _mm_add_ps(s_qjl, _mm_movehl_ps(s_qjl, s_qjl));
  s_qjl = _mm_add_ss(s_qjl, _mm_shuffle_ps(s_qjl, s_qjl, 1));
  float dot_qjl = _mm_cvtss_f32(s_qjl);

  for (; i < dim; ++i) {
    ip_mse += q_rot[i] * centroids[packed[i] >> 1] * sigma;
    dot_qjl += s_q[i] * ((packed[i] & 1) ? 1.0f : -1.0f);
  }

  const float correction = space->scale() * gamma * dot_qjl;
  const float ip = (ip_mse + correction) * x_norm * pq->q_norm;
  return std::max(0.0f, pq->q_norm_sq + x_norm * x_norm - 2.0f * ip);
}

static float distBuildAVX(const void *pVect1, const void *pVect2,
                          const void *param_ptr) {
  const auto *space = static_cast<const TurboQuantSpace *>(param_ptr);
  const size_t dim = space->dim();
  const float *centroids = space->centroids();

  const char *buf_a = static_cast<const char *>(pVect1);
  const char *buf_b = static_cast<const char *>(pVect2);

  const uint8_t *packed_a = reinterpret_cast<const uint8_t *>(buf_a);
  const float *meta_a = reinterpret_cast<const float *>(buf_a + dim);
  const float norm_a = meta_a[0];
  const float sigma_a = meta_a[2];

  const uint8_t *packed_b = reinterpret_cast<const uint8_t *>(buf_b);
  const float *meta_b = reinterpret_cast<const float *>(buf_b + dim);
  const float norm_b = meta_b[0];
  const float sigma_b = meta_b[2];

  const __m256 v_sigma_a = _mm256_set1_ps(sigma_a);
  const __m256 v_sigma_b = _mm256_set1_ps(sigma_b);
  __m256 sum = _mm256_setzero_ps();

  size_t i = 0;
  for (; i + 8 <= dim; i += 8) {
#ifdef __AVX2__
    __m256i idx_a = _mm256_set_epi32(packed_a[i + 7] >> 1, packed_a[i + 6] >> 1,
                                     packed_a[i + 5] >> 1, packed_a[i + 4] >> 1,
                                     packed_a[i + 3] >> 1, packed_a[i + 2] >> 1,
                                     packed_a[i + 1] >> 1, packed_a[i] >> 1);
    __m256 va =
        _mm256_mul_ps(_mm256_i32gather_ps(centroids, idx_a, 4), v_sigma_a);

    __m256i idx_b = _mm256_set_epi32(packed_b[i + 7] >> 1, packed_b[i + 6] >> 1,
                                     packed_b[i + 5] >> 1, packed_b[i + 4] >> 1,
                                     packed_b[i + 3] >> 1, packed_b[i + 2] >> 1,
                                     packed_b[i + 1] >> 1, packed_b[i] >> 1);
    __m256 vb =
        _mm256_mul_ps(_mm256_i32gather_ps(centroids, idx_b, 4), v_sigma_b);
#else
    __m256 va = _mm256_mul_ps(
        _mm256_set_ps(
            centroids[packed_a[i + 7] >> 1], centroids[packed_a[i + 6] >> 1],
            centroids[packed_a[i + 5] >> 1], centroids[packed_a[i + 4] >> 1],
            centroids[packed_a[i + 3] >> 1], centroids[packed_a[i + 2] >> 1],
            centroids[packed_a[i + 1] >> 1], centroids[packed_a[i] >> 1]),
        v_sigma_a);
    __m256 vb = _mm256_mul_ps(
        _mm256_set_ps(
            centroids[packed_b[i + 7] >> 1], centroids[packed_b[i + 6] >> 1],
            centroids[packed_b[i + 5] >> 1], centroids[packed_b[i + 4] >> 1],
            centroids[packed_b[i + 3] >> 1], centroids[packed_b[i + 2] >> 1],
            centroids[packed_b[i + 1] >> 1], centroids[packed_b[i] >> 1]),
        v_sigma_b);
#endif
    sum = _mm256_add_ps(sum, _mm256_mul_ps(va, vb));
  }

  __m128 hi = _mm256_extractf128_ps(sum, 1);
  __m128 lo = _mm256_castps256_ps128(sum);
  __m128 s = _mm_add_ps(lo, hi);
  s = _mm_add_ps(s, _mm_movehl_ps(s, s));
  s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
  float ip_rot = _mm_cvtss_f32(s);

  for (; i < dim; ++i)
    ip_rot += (centroids[packed_a[i] >> 1] * sigma_a) *
              (centroids[packed_b[i] >> 1] * sigma_b);

  const float ip = ip_rot * norm_a * norm_b;
  return std::max(0.0f, norm_a * norm_a + norm_b * norm_b - 2.0f * ip);
}

#endif // USE_AVX

// ===========================================================================
// Compress/Save/Load utilities
// ===========================================================================

// ---------------------------------------------------------------------------
// Raw vector file format (.tqrv) — version 2
//
// Header (32 bytes, fixed):
//   [magic: u32 = "TQRV"] [version: u32 = 2]
//   [num_vectors: u64] [dim: u64]
//   [dtype: u32] [reserved: u32]
//
// Data (starting at offset 32):
//   N * dim * element_size bytes, contiguous by internal ID.
//
// dtype: 0 = float32 (4 bytes), 1 = float16 (2 bytes).
// Vector i is at offset: 32 + i * dim * element_size.
// ---------------------------------------------------------------------------

static const uint32_t TQRV_MAGIC = 0x54515256; // "TQRV"
static const uint32_t TQRV_VERSION = 2;

enum RawVectorDtype : uint32_t {
  DTYPE_FLOAT32 = 0,
  DTYPE_FLOAT16 = 1,
};

// ===========================================================================
// Section 0: IEEE 754 float16 ↔ float32 conversion
//
// Portable bit-manipulation, no hardware fp16 dependency.
// Used for compact raw-vector storage (.tqrv dtype=1).
// ===========================================================================

inline uint16_t float_to_fp16(float val) {
    uint32_t bits;
    std::memcpy(&bits, &val, 4);
    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t frac = bits & 0x7FFFFF;

    if (exp <= 0) {
        // Underflow → zero (denormals omitted for simplicity)
        return static_cast<uint16_t>(sign);
    }
    if (exp >= 31) {
        // Overflow → infinity, preserve NaN
        return static_cast<uint16_t>(sign | 0x7C00 | ((frac != 0) ? 0x0200 : 0));
    }
    return static_cast<uint16_t>(sign | (exp << 10) | (frac >> 13));
}

inline float fp16_to_float(uint16_t h) {
    uint32_t sign = (static_cast<uint32_t>(h) & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t frac = h & 0x03FF;

    uint32_t bits;
    if (exp == 0) {
        // Zero or denormal → flush to signed zero
        bits = sign;
    } else if (exp == 31) {
        // Inf / NaN
        bits = sign | 0x7F800000 | (frac << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (frac << 13);
    }
    float result;
    std::memcpy(&result, &bits, 4);
    return result;
}

#if defined(USE_NEON)

inline uint16_t float_to_fp16_NEON(float val) {
    // ARM64 с +fp16 — native hardware
    __fp16 h = static_cast<__fp16>(val);
    uint16_t res;
    std::memcpy(&res, &h, sizeof(res));
    return res;
}

inline float fp16_to_float_NEON(uint16_t h) {
    // ARM64 hardware
    __fp16 fh;
    std::memcpy(&fh, &h, sizeof(fh));
    return static_cast<float>(fh);
}

#endif

/// Save raw float32 vectors from an L2-built HNSW index to a flat file.
/// Must be called BEFORE compressIndex (while data slots still contain floats).
inline Status saveRawVectors(const std::string &path,
                             const HierarchicalNSW<float> &hnsw, size_t dim,
                             RawVectorDtype dtype = DTYPE_FLOAT32) {

  std::ofstream out(path, std::ios::binary);
  if (!out.good())
    return Status("saveRawVectors: cannot open output file");

  size_t n = hnsw.cur_element_count.load();

  // Header (32 bytes)
  uint64_t n64 = static_cast<uint64_t>(n);
  uint64_t dim64 = static_cast<uint64_t>(dim);
  uint32_t dtype32 = static_cast<uint32_t>(dtype);
  uint32_t reserved = 0;
  out.write(reinterpret_cast<const char *>(&TQRV_MAGIC), 4);
  out.write(reinterpret_cast<const char *>(&TQRV_VERSION), 4);
  out.write(reinterpret_cast<const char *>(&n64), 8);
  out.write(reinterpret_cast<const char *>(&dim64), 8);
  out.write(reinterpret_cast<const char *>(&dtype32), 4);
  out.write(reinterpret_cast<const char *>(&reserved), 4);

  const size_t vec_bytes = dim * sizeof(float);

  if (dtype == DTYPE_FLOAT32) {
    for (size_t i = 0; i < n; ++i) {
      const char *data = hnsw.getDataByInternalId(static_cast<tableint>(i));
      out.write(data, vec_bytes);
    }
  } else {
    // float16: convert each vector on the fly
    std::vector<uint16_t> fp16_buf(dim);
    for (size_t i = 0; i < n; ++i) {
      const float *data = reinterpret_cast<const float *>(
          hnsw.getDataByInternalId(static_cast<tableint>(i)));
      for (size_t j = 0; j < dim; ++j)
        fp16_buf[j] = float_to_fp16(data[j]);
      out.write(reinterpret_cast<const char *>(fp16_buf.data()),
                dim * sizeof(uint16_t));
    }
  }

  if (!out.good())
    return Status("saveRawVectors: write error");

  out.close();
  return OkStatus();
}

/// Compress an L2-built HNSW index in-place: encode each float32 vector
/// to a TurboQuant code in its data slot, then switch the distance function.
///
/// Preconditions:
///   - hnsw was built with L2Space (data slots contain float[dim])
///   - saveRawVectors was called first if re-ranking is needed
///
/// After this call:
///   - Data slots contain TQ codes (zero-padded to original stride)
///   - hnsw uses tq_space for distance computation
///   - tq_space must outlive hnsw (it owns dist_func_param data)
///   - Do NOT call addPoint after compressIndex
inline Status compressIndex(HierarchicalNSW<float> &hnsw,
                            TurboQuantSpace &tq_space) {

  size_t n = hnsw.cur_element_count.load();
  size_t dim = tq_space.dim();
  size_t code_size = tq_space.codeSizeBytes();
  size_t data_size = hnsw.data_size_;

  if (code_size > data_size)
    return Status("compressIndex: TQ code size exceeds L2 data slot size");

  std::vector<float> float_buf(dim);
  std::vector<char> code_buf(code_size);

  for (size_t i = 0; i < n; ++i) {
    char *slot = hnsw.getDataByInternalId(static_cast<tableint>(i));

    // Read float data into temporary buffer
    std::memcpy(float_buf.data(), slot, dim * sizeof(float));

    // Encode to TQ code
    tq_space.encodeVector(float_buf.data(), code_buf.data());

    // Write TQ code back (zero-pad to original slot size)
    std::memset(slot, 0, data_size);
    std::memcpy(slot, code_buf.data(), code_size);
  }

  // Switch distance function to TQ (build mode — symmetric)
  tq_space.setBuildMode(hnsw);

  return OkStatus();
}

/// Load specific raw vectors by internal ID from a .tqrv file (v1 or v2).
/// Seeks to each requested vector — does not load the entire file.
/// Output is always float32 (fp16 data is converted on read).
inline Status loadRawVectors(const std::string &path, const size_t *ids,
                             size_t num_ids, size_t dim, float *out) {

  std::ifstream in(path, std::ios::binary);
  if (!in.good())
    return Status("loadRawVectors: cannot open file");

  uint32_t magic, version;
  uint64_t n64, dim64;
  in.read(reinterpret_cast<char *>(&magic), 4);
  in.read(reinterpret_cast<char *>(&version), 4);
  in.read(reinterpret_cast<char *>(&n64), 8);
  in.read(reinterpret_cast<char *>(&dim64), 8);

  if (magic != TQRV_MAGIC)
    return Status("loadRawVectors: invalid file magic");
  if (dim64 != static_cast<uint64_t>(dim))
    return Status("loadRawVectors: dimension mismatch");

  // Parse version-dependent fields
  uint64_t data_offset = 32;
  RawVectorDtype dtype = DTYPE_FLOAT32;
  if (version == 1) {
    uint64_t off_tmp;
    in.read(reinterpret_cast<char *>(&off_tmp), 8); // data_offset field
    data_offset = off_tmp;
  } else if (version == 2) {
    uint32_t dtype32, reserved;
    in.read(reinterpret_cast<char *>(&dtype32), 4);
    in.read(reinterpret_cast<char *>(&reserved), 4);
    dtype = static_cast<RawVectorDtype>(dtype32);
  } else {
    return Status("loadRawVectors: unsupported version");
  }

  const size_t elem_size = (dtype == DTYPE_FLOAT16) ? 2 : 4;
  const size_t vec_bytes = dim * elem_size;
  std::vector<uint16_t> fp16_buf;
  if (dtype == DTYPE_FLOAT16)
    fp16_buf.resize(dim);

  for (size_t i = 0; i < num_ids; ++i) {
    if (ids[i] >= n64)
      return Status("loadRawVectors: ID out of range");

    std::streamoff offset =
        static_cast<std::streamoff>(data_offset + ids[i] * vec_bytes);
    in.seekg(offset, std::ios::beg);

    if (dtype == DTYPE_FLOAT32) {
      in.read(reinterpret_cast<char *>(out + i * dim), vec_bytes);
    } else {
      in.read(reinterpret_cast<char *>(fp16_buf.data()), vec_bytes);
      for (size_t j = 0; j < dim; ++j)
        out[i * dim + j] = fp16_to_float(fp16_buf[j]);
    }

    if (!in.good())
      return Status("loadRawVectors: read error");
  }

  return OkStatus();
}

// ===========================================================================
// MappedRawVectors — memory-mapped access to .tqrv files
//
// Maps the entire file read-only. get() returns a pointer into the mapped
// region (zero-copy for float32) or converts fp16→fp32 into a caller buffer.
// POSIX (Linux/macOS): mmap + madvise(MADV_RANDOM).
// Windows: falls back to reading the entire file into malloc'd memory.
// ===========================================================================

class MappedRawVectors {
  void *mapped_;     ///< mmap base (POSIX) or malloc'd block (Windows)
  size_t file_size_; ///< total mapped/allocated size
  const char *data_; ///< pointer to first vector byte
  uint64_t num_vectors_;
  uint64_t dim_;
  RawVectorDtype dtype_;
  size_t elem_size_; ///< bytes per element: 4 (fp32) or 2 (fp16)
  size_t vec_bytes_; ///< dim_ * elem_size_
  bool is_mmap_;     ///< true = mmap, false = malloc fallback

public:
  MappedRawVectors()
      : mapped_(nullptr), file_size_(0), data_(nullptr), num_vectors_(0),
        dim_(0), dtype_(DTYPE_FLOAT32), elem_size_(4), vec_bytes_(0),
        is_mmap_(false) {}

  ~MappedRawVectors() { close(); }

  // Non-copyable, movable
  MappedRawVectors(const MappedRawVectors &) = delete;
  MappedRawVectors &operator=(const MappedRawVectors &) = delete;
  MappedRawVectors(MappedRawVectors &&o) noexcept { moveFrom(o); }
  MappedRawVectors &operator=(MappedRawVectors &&o) noexcept {
    if (this != &o) {
      close();
      moveFrom(o);
    }
    return *this;
  }

  /// Open and map a .tqrv file (v1 or v2).
  Status open(const std::string &path) {
    if (mapped_)
      return Status("MappedRawVectors: already open");

#ifndef _WIN32
    // --- POSIX path: mmap ---
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0)
      return Status("MappedRawVectors: cannot open file");

    struct stat st;
    if (fstat(fd, &st) != 0) {
      ::close(fd);
      return Status("MappedRawVectors: fstat failed");
    }
    file_size_ = static_cast<size_t>(st.st_size);

    mapped_ = mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (mapped_ == MAP_FAILED) {
      mapped_ = nullptr;
      return Status("MappedRawVectors: mmap failed");
    }

    madvise(mapped_, file_size_, MADV_RANDOM);
    is_mmap_ = true;
#else
    // --- Windows fallback: read into memory ---
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in.good())
      return Status("MappedRawVectors: cannot open file");
    file_size_ = static_cast<size_t>(in.tellg());
    in.seekg(0);

    mapped_ = std::malloc(file_size_);
    if (!mapped_)
      return Status("MappedRawVectors: malloc failed");
    in.read(static_cast<char *>(mapped_), file_size_);
    if (!in.good()) {
      std::free(mapped_);
      mapped_ = nullptr;
      return Status("MappedRawVectors: read failed");
    }
    is_mmap_ = false;
#endif

    return parseHeader();
  }

  void close() {
    if (!mapped_)
      return;
#ifndef _WIN32
    if (is_mmap_)
      munmap(mapped_, file_size_);
    else
      std::free(mapped_);
#else
    std::free(mapped_);
#endif
    mapped_ = nullptr;
    data_ = nullptr;
  }

  // -- Accessors ----------------------------------------------------------

  bool is_open() const { return mapped_ != nullptr; }
  uint64_t num_vectors() const { return num_vectors_; }
  uint64_t dim() const { return dim_; }
  RawVectorDtype dtype() const { return dtype_; }

  /// Zero-copy access for float32 files.  Returns nullptr for float16.
  const float *get_float32(uint64_t label) const {
    assert(label < num_vectors_ && "get_float32: label out of range");
    if (dtype_ != DTYPE_FLOAT32)
      return nullptr;
    return reinterpret_cast<const float *>(data_ + label * vec_bytes_);
  }

  /// Universal access: writes dim floats to buf.
  /// For float32 — memcpy.  For float16 — converts to float32.
  void get(uint64_t label, float *buf) const {
    assert(label < num_vectors_ && "get: label out of range");
    const char *src = data_ + label * vec_bytes_;
    if (dtype_ == DTYPE_FLOAT32) {
      std::memcpy(buf, src, vec_bytes_);
    } else {
      const uint16_t *fp16 = reinterpret_cast<const uint16_t *>(src);
      for (uint64_t i = 0; i < dim_; ++i)
        buf[i] = fp16_to_float(fp16[i]);
    }
  }

private:
  Status parseHeader() {
    if (file_size_ < 32) {
      close();
      return Status("MappedRawVectors: file too small");
    }

    const char *hdr = static_cast<const char *>(mapped_);
    uint32_t magic, version;
    std::memcpy(&magic, hdr, 4);
    std::memcpy(&version, hdr + 4, 4);

    if (magic != TQRV_MAGIC) {
      close();
      return Status("MappedRawVectors: invalid magic");
    }

    std::memcpy(&num_vectors_, hdr + 8, 8);
    std::memcpy(&dim_, hdr + 16, 8);

    uint64_t data_offset = 32;
    if (version == 1) {
      dtype_ = DTYPE_FLOAT32;
      std::memcpy(&data_offset, hdr + 24, 8);
    } else if (version == 2) {
      uint32_t dtype32;
      std::memcpy(&dtype32, hdr + 24, 4);
      dtype_ = static_cast<RawVectorDtype>(dtype32);
    } else {
      close();
      return Status("MappedRawVectors: unsupported version");
    }

    elem_size_ = (dtype_ == DTYPE_FLOAT16) ? 2 : 4;
    vec_bytes_ = dim_ * elem_size_;
    data_ = static_cast<const char *>(mapped_) + data_offset;

    size_t expected = data_offset + num_vectors_ * vec_bytes_;
    if (file_size_ < expected) {
      close();
      return Status("MappedRawVectors: file truncated");
    }

    return OkStatus();
  }

  void moveFrom(MappedRawVectors &o) {
    mapped_ = o.mapped_;
    o.mapped_ = nullptr;
    file_size_ = o.file_size_;
    o.file_size_ = 0;
    data_ = o.data_;
    o.data_ = nullptr;
    num_vectors_ = o.num_vectors_;
    o.num_vectors_ = 0;
    dim_ = o.dim_;
    o.dim_ = 0;
    dtype_ = o.dtype_;
    elem_size_ = o.elem_size_;
    vec_bytes_ = o.vec_bytes_;
    is_mmap_ = o.is_mmap_;
  }
};

// ===========================================================================
// TurboQuantIndex — high-level facade for TQ-compressed HNSW
//
// Owns TurboQuantSpace + HierarchicalNSW + optional MappedRawVectors.
// Provides build/save/load/search/searchRerank with minimal boilerplate.
//
// Thread safety: search() and searchRerank() are fully thread-safe after
// build() or load() completes. Each call creates a stack-local
// TurboQuantPreparedQuery — no shared mutable state.
//
// Usage:
//   TurboQuantIndex idx(128, 8);           // dim=128, q8
//   idx.build(data, N, 16, 200);           // M=16, ef_construction=200
//   idx.save("index.bin", "raw.tqrv");
//
//   TurboQuantIndex idx2(128, 8);
//   idx2.load("index.bin", "raw.tqrv");    // raw_path optional
//   idx2.setEf(64);
//   auto r1 = idx2.search(query, 10);
//   auto r2 = idx2.searchRerank(query, 10, 100);
// ===========================================================================

class TurboQuantIndex {
  size_t dim_;
  int bits_per_coord_;
  uint64_t rot_seed_;
  uint64_t qjl_seed_;

  std::unique_ptr<TurboQuantSpace> space_;
  std::unique_ptr<HierarchicalNSW<float>> hnsw_;
  MappedRawVectors raw_vectors_;

  // In-memory raw vectors (used by buildFromL2). Row-major, n * dim_ floats.
  // If non-empty, takes precedence over raw_vectors_ in searchRerank.
  std::vector<float> inmem_raw_;

  // Temporary L2 space used during buildFromL2. Kept as a member so the
  // HNSW graph (which holds a raw pointer to its space) stays valid between
  // initL2Build() and finalizeFromL2().
  std::unique_ptr<L2Space> l2_space_tmp_;

public:
  TurboQuantIndex(size_t dim, int bits_per_coord = 8, uint64_t rot_seed = 42,
                  uint64_t qjl_seed = 137)
      : dim_(dim), bits_per_coord_(bits_per_coord), rot_seed_(rot_seed),
        qjl_seed_(qjl_seed) {}

  // -- Build ----------------------------------------------------------------

  /// Initialize space and HNSW graph for building. Call addPoint() to populate.
  void initBuild(size_t max_elements, size_t M = 16,
                 size_t ef_construction = 200) {
    space_.reset(
        new TurboQuantSpace(dim_, bits_per_coord_, rot_seed_, qjl_seed_));
    hnsw_.reset(new HierarchicalNSW<float>(space_.get(), max_elements, M,
                                           ef_construction));
  }

  /// Encode a float vector and add it to the index. Thread-safe (addPoint uses
  /// mutexes). Caller must provide a thread-local buffer of size
  /// codeSizeBytes().
  void addPoint(const float *vec, labeltype label, char *encode_buf) {
    space_->encodeVector(vec, encode_buf);
    hnsw_->addPoint(encode_buf, label);
  }

  /// Build a TQ-compressed HNSW index from raw float vectors (single-threaded).
  /// data[i] points to a float[dim] vector for label i.
  Status build(const float *const *data, size_t n, size_t M = 16,
               size_t ef_construction = 200) {
    initBuild(n, M, ef_construction);

    std::vector<char> buf(space_->codeSizeBytes());
    for (size_t i = 0; i < n; ++i)
      addPoint(data[i], i, buf.data());

    return OkStatus();
  }

  /// Build from contiguous array: data points to n*dim floats, row-major.
  Status build(const float *data, size_t n, size_t M = 16,
               size_t ef_construction = 200) {
    std::vector<const float *> ptrs(n);
    for (size_t i = 0; i < n; ++i)
      ptrs[i] = data + i * dim_;
    return build(ptrs.data(), n, M, ef_construction);
  }

  // -- L2-graph build path (low-level, for parallel insertion) --------------
  //
  // Use these three methods when you need parallel graph construction on
  // raw float32 vectors (e.g. from Python bindings):
  //
  //   idx.initL2Build(n, M, ef_construction);
  //   // parallel loop:
  //   idx.addL2Point(vec_i, i);
  //   idx.finalizeFromL2(raw_data_ptr, n, /*keep_raw=*/true);
  //
  // After finalizeFromL2, search() and searchRerank() are thread-safe.

  /// Allocate an HNSW graph backed by L2Space for raw float32 insertion.
  void initL2Build(size_t max_elements, size_t M = 16,
                   size_t ef_construction = 200) {
    l2_space_tmp_.reset(new L2Space(dim_));
    // Drop any previous TQ space — it will be recreated in finalizeFromL2.
    space_.reset();
    hnsw_.reset(new HierarchicalNSW<float>(l2_space_tmp_.get(), max_elements,
                                           M, ef_construction));
  }

  /// Insert a raw float32 vector into the L2 graph. Thread-safe.
  void addL2Point(const float *vec, labeltype label) {
    hnsw_->addPoint(vec, label);
  }

  /// Replace stored raw data with TQ codes, switch to TQ asymmetric search,
  /// and optionally stash raw data in memory for rerank.
  /// raw_data must point to the same float32 vectors that were inserted
  /// (row-major, n * dim), or nullptr if keep_raw is false.
  Status finalizeFromL2(const float *raw_data, size_t n, bool keep_raw) {
    if (!hnsw_ || !l2_space_tmp_)
      return Status("finalizeFromL2: initL2Build was not called");

    space_.reset(
        new TurboQuantSpace(dim_, bits_per_coord_, rot_seed_, qjl_seed_));

    auto st = compressIndex(*hnsw_, *space_);
    if (!st.ok())
      return st;

    // HNSW now points into space_ via the swapped dist func. We can drop
    // the temporary L2Space — no live references remain.
    l2_space_tmp_.reset();

    space_->setSearchMode(*hnsw_);

    if (keep_raw && raw_data != nullptr) {
      inmem_raw_.assign(raw_data, raw_data + n * dim_);
    } else {
      inmem_raw_.clear();
      inmem_raw_.shrink_to_fit();
    }

    return OkStatus();
  }

  /// Build an HNSW graph using L2 on raw float32, then replace stored data
  /// with TQ codes and switch to TQ asymmetric search. Keeps a copy of raw
  /// data in memory so searchRerank() works without a .tqrv file.
  ///
  /// This is the "l2graph_tq_rerank" mode: accurate graph topology from L2,
  /// fast candidate retrieval via TQ, and exact L2 rerank. On SIFT1M it
  /// reaches recall within ~0.02 of pure L2 while using TQ-sized data slots.
  ///
  /// Parameters:
  ///   data            — n * dim float32, row-major
  ///   n               — number of vectors
  ///   M, ef_construction — HNSW graph parameters
  ///   keep_raw        — if true, stores raw data in-memory for rerank
  ///   add_point_fn    — optional hook to parallelize insertion; if nullptr,
  ///                     a serial loop is used
  ///
  /// Thread safety: after this returns, search() and searchRerank() are
  /// fully thread-safe.
  Status buildFromL2(const float *data, size_t n, size_t M = 16,
                     size_t ef_construction = 200, bool keep_raw = true) {
    if (data == nullptr || n == 0)
      return Status("buildFromL2: empty data");

    initL2Build(n, M, ef_construction);
    for (size_t i = 0; i < n; ++i)
      addL2Point(data + i * dim_, i);
    return finalizeFromL2(data, n, keep_raw);
  }

  // -- Save / Load ----------------------------------------------------------

  /// Save index and optionally raw vectors for re-ranking.
  /// raw_data points to the original float vectors (n * dim, row-major).
  /// If raw_path is empty, raw vectors are not saved.
  Status save(const std::string &index_path, const std::string &raw_path = "",
              const float *raw_data = nullptr, size_t n = 0,
              RawVectorDtype dtype = DTYPE_FLOAT32) const {
    if (!hnsw_)
      return Status("TurboQuantIndex::save: no index built");

    hnsw_->saveIndex(index_path);

    if (!raw_path.empty() && raw_data != nullptr && n > 0) {
      // Write raw vectors directly (hnsw data slots are already compressed)
      std::ofstream out(raw_path, std::ios::binary);
      if (!out.good())
        return Status("TurboQuantIndex::save: cannot open raw file");

      uint64_t n64 = static_cast<uint64_t>(n);
      uint64_t dim64 = static_cast<uint64_t>(dim_);
      uint32_t dtype32 = static_cast<uint32_t>(dtype);
      uint32_t reserved = 0;
      out.write(reinterpret_cast<const char *>(&TQRV_MAGIC), 4);
      out.write(reinterpret_cast<const char *>(&TQRV_VERSION), 4);
      out.write(reinterpret_cast<const char *>(&n64), 8);
      out.write(reinterpret_cast<const char *>(&dim64), 8);
      out.write(reinterpret_cast<const char *>(&dtype32), 4);
      out.write(reinterpret_cast<const char *>(&reserved), 4);

      if (dtype == DTYPE_FLOAT32) {
        out.write(reinterpret_cast<const char *>(raw_data),
                  n * dim_ * sizeof(float));
      } else {
        std::vector<uint16_t> fp16_buf(dim_);
        for (size_t i = 0; i < n; ++i) {
          const float *vec = raw_data + i * dim_;
          for (size_t j = 0; j < dim_; ++j)
            fp16_buf[j] = float_to_fp16(vec[j]);
          out.write(reinterpret_cast<const char *>(fp16_buf.data()),
                    dim_ * sizeof(uint16_t));
        }
      }

      if (!out.good())
        return Status("TurboQuantIndex::save: write error");
    }

    return OkStatus();
  }

  /// Load index from disk. raw_path is optional (enables searchRerank).
  Status load(const std::string &index_path, const std::string &raw_path = "") {
    space_.reset(
        new TurboQuantSpace(dim_, bits_per_coord_, rot_seed_, qjl_seed_));
    hnsw_.reset(new HierarchicalNSW<float>(space_.get(), index_path));
    space_->setSearchMode(*hnsw_);

    if (!raw_path.empty()) {
      auto st = raw_vectors_.open(raw_path);
      if (!st.ok())
        return st;
    }

    return OkStatus();
  }

  // -- Search ---------------------------------------------------------------

  void setEf(size_t ef) {
    if (hnsw_)
      hnsw_->setEf(ef);
  }

  size_t getEf() const { return hnsw_ ? hnsw_->ef_ : 0; }

  /// TQ-only search. Thread-safe.
  /// Returns priority queue of (distance, label) pairs, worst first.
  std::priority_queue<std::pair<float, labeltype>> search(const float *query,
                                                          size_t k) const {
    ensureSearchMode();
    auto pq = space_->prepareQuery(query);
    return hnsw_->searchKnn(&pq, k);
  }

  /// TQ search + exact L2 re-ranking from mmap'd raw vectors. Thread-safe.
  /// Retrieves rerank_ef candidates via TQ, re-ranks by exact L2, returns
  /// top-k. rerank_ef controls the number of TQ candidates to re-rank. Set hnsw
  /// ef >= rerank_ef before calling (via setEf). Default rerank_ef = 0 means
  /// use current ef (i.e. re-rank all candidates that searchKnn returns).
  std::vector<std::pair<float, labeltype>>
  searchRerank(const float *query, size_t k, size_t rerank_ef = 0) const {
    const bool use_inmem = !inmem_raw_.empty();
    if (!use_inmem && !raw_vectors_.is_open())
      return {};

    if (rerank_ef == 0)
      rerank_ef = hnsw_->ef_;

    // Step 1: TQ search for broad candidate set
    // searchKnn uses max(ef_, rerank_ef) internally, so ef_ >= rerank_ef
    // is recommended for best performance (avoids redundant graph traversal).
    ensureSearchMode();
    auto pq = space_->prepareQuery(query);
    auto tq_result = hnsw_->searchKnn(&pq, rerank_ef);

    // Step 2: exact L2 re-rank
    std::vector<std::pair<float, labeltype>> shortlist;
    shortlist.reserve(tq_result.size());

    // Thread-local buffer for fp16 conversion
    std::vector<float> vec_buf(dim_);

    while (!tq_result.empty()) {
      labeltype id = tq_result.top().second;
      tq_result.pop();

      const float *raw_vec;
      if (use_inmem) {
        raw_vec = inmem_raw_.data() + static_cast<size_t>(id) * dim_;
      } else {
        raw_vec = raw_vectors_.get_float32(id);
        if (raw_vec == nullptr) {
          raw_vectors_.get(id, vec_buf.data());
          raw_vec = vec_buf.data();
        }
      }

      // Exact L2 distance
      float dist = 0.0f;
      for (size_t i = 0; i < dim_; ++i) {
        float diff = query[i] - raw_vec[i];
        dist += diff * diff;
      }
      shortlist.push_back({dist, id});
    }

    // Step 3: select top-k
    if (shortlist.size() > k) {
      std::partial_sort(shortlist.begin(), shortlist.begin() + k,
                        shortlist.end());
      shortlist.resize(k);
    } else {
      std::sort(shortlist.begin(), shortlist.end());
    }

    return shortlist;
  }

  // -- Accessors ------------------------------------------------------------

  bool hasRawVectors() const {
    return raw_vectors_.is_open() || !inmem_raw_.empty();
  }
  size_t dim() const { return dim_; }
  size_t numElements() const {
    return hnsw_ ? hnsw_->cur_element_count.load() : 0;
  }
  size_t codeSizeBytes() const { return space_ ? space_->codeSizeBytes() : 0; }

  const TurboQuantSpace *space() const { return space_.get(); }
  const HierarchicalNSW<float> *hnsw() const { return hnsw_.get(); }

private:
  void ensureSearchMode() const {
    // After build() the HNSW still has build dist func.
    // Lazily switch on first search. Safe: setSearchMode just writes
    // two pointers, and concurrent reads of the same value are fine.
    if (hnsw_->fstdistfunc_ != space_->get_search_dist_func()) {
      const_cast<TurboQuantSpace *>(space_.get())
          ->setSearchMode(*const_cast<HierarchicalNSW<float> *>(hnsw_.get()));
    }
  }
};

} // namespace turboquant
} // namespace hnswlib