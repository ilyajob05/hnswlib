#pragma once
/// turbo_quant.h — TurboQuant (ICLR 2026, arXiv:2504.19874), Algorithm 2
///
/// Two-stage near-optimal online vector quantization (TurboQuant_prod):
///   Stage 1 (PolarQuant MSE): (b-1)-bit Lloyd-Max scalar quantization
///   Stage 2 (QJL):            1-bit sign sketch of the residual via RHT
///
/// Reconstruction:
///   x̃ = x̃_mse + (√(π/2)/√d) · γ · Sᵀ · qjl   where S = RHT matrix
///
/// Constraints: header-only, zero dependencies, C++11, power-of-2 dimensions.
/// Phase: prototype (V1 Gather approach, 1 byte per SQ index).
///        Tight bit packing deferred to Phase 2.

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <cstring>
#include <vector>
#include <cassert>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace hnswlib {
namespace turboquant {

// ===========================================================================
// Section 1: PRNG — SplitMix64
// Deterministic 64-bit PRNG for random sign flips in RHT. Not cryptographic.
// ===========================================================================

class SplitMix64 {
    uint64_t state_;
public:
    explicit SplitMix64(uint64_t seed) : state_(seed) {}

    uint64_t next() {
        uint64_t z = (state_ += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }

    float randomSign() {
        return (next() & 1) ? 1.0f : -1.0f;
    }

    /// Applies random sign flip to count elements.
    /// One RNG call per 64 elements.
    void applyRandomSigns(float* __restrict__ data, size_t count) {
        size_t i = 0;
        for (; i + 64 <= count; i += 64) {
            uint64_t bits = next();
            for (size_t j = 0; j < 64; ++j) {
                float sign = (bits & (1ULL << j)) ? 1.0f : -1.0f;
                data[i + j] *= sign;
            }
        }
        if (i < count) {
            uint64_t bits = next();
            for (size_t j = 0; i + j < count; ++j) {
                float sign = (bits & (1ULL << j)) ? 1.0f : -1.0f;
                data[i + j] *= sign;
            }
        }
    }
};

// ===========================================================================
// Section 2: Walsh-Hadamard Transform (WHT)
// In-place iterative butterfly, O(d log d). Energy-preserving (orthogonal).
//
// Input:  data — float array of length d (d must be a positive power of 2)
// Output: data — WHT-transformed, normalized by 1/√d
// ===========================================================================

inline void whtInplace(float* data, size_t d) {
    assert(d > 0 && (d & (d - 1)) == 0);

    for (size_t step = 1; step < d; step <<= 1) {
        const size_t jump = step << 1;
        for (size_t i = 0; i < d; i += jump) {
            float* __restrict__ low  = &data[i];
            float* __restrict__ high = &data[i + step];
            for (size_t j = 0; j < step; ++j) {
                float a = low[j];
                float b = high[j];
                low[j]  = a + b;
                high[j] = a - b;
            }
        }
    }

    float norm = 1.0f / std::sqrt(static_cast<float>(d));
    for (size_t i = 0; i < d; ++i) {
        data[i] *= norm;
    }
}

// ===========================================================================
// Section 3: Randomized Hadamard Transform (RHT)
// Applies random diagonal sign matrix D (determined by seed) then WHT.
// Orthogonal transform preserving inner products and norms.
// Used as both PolarQuant rotation and QJL projection matrix S.
//
// Input:  data — float[d], seed — deterministic RNG seed
// Output: data — RHT-transformed in-place
// ===========================================================================

inline void randomizedHadamard(float* data, size_t d, uint64_t seed) {
    assert(d > 0 && (d & (d - 1)) == 0
           && "randomizedHadamard: d must be a positive power of 2");

    SplitMix64 rng(seed);
    for (size_t i = 0; i < d; ++i) {
        data[i] *= rng.randomSign();
    }

    whtInplace(data, d);
}

// ===========================================================================
// Section 4: Lloyd-Max quantizer tables for Gaussian N(0,1)
// Optimal minimum-MSE scalar quantizer (Max, 1960).
// Boundaries = decision thresholds; centroids = reconstruction levels.
// Tables are symmetric around 0.
// ===========================================================================

namespace detail {

// 3-bit Lloyd-Max (8 levels) — MSE stage when total budget b=4
constexpr float LM3_BOUNDARIES[7] = {
    -1.748f, -1.050f, -0.5006f,
     0.0f,
     0.5006f,  1.050f,  1.748f
};

constexpr float LM3_CENTROIDS[8] = {
    -2.1519f, -1.3440f, -0.7560f, -0.2451f,
     0.2451f,  0.7560f,  1.3440f,  2.1519f
};

// 4-bit Lloyd-Max (16 levels) — MSE stage when total budget b=5
constexpr float LM4_BOUNDARIES[15] = {
    -2.401f, -1.844f, -1.437f, -1.099f,
    -0.7961f, -0.5097f, -0.2318f,
     0.0f,
     0.2318f,  0.5097f,  0.7961f,
     1.099f,   1.437f,   1.844f,   2.401f
};

constexpr float LM4_CENTROIDS[16] = {
    -3.0867f, -2.0995f, -1.6180f, -1.2562f,
    -0.9423f, -0.6568f, -0.3881f, -0.1284f,
     0.1284f,  0.3881f,  0.6568f,  0.9423f,
     1.2562f,  1.6180f,  2.0995f,  3.0867f
};

}  // namespace detail

// ===========================================================================
// Section 5: Scalar Quantization
// ===========================================================================

/// Maps a scalar value to a Lloyd-Max bin index via linear scan.
/// Input:  val — normalized value (zero-mean, unit-variance expected)
///         boundaries — sorted decision thresholds, length num_boundaries
/// Output: bin index in [0, num_boundaries]
inline uint8_t sqQuantize(float val, const float* boundaries,
                          int num_boundaries) {
    assert(num_boundaries > 0 && "sqQuantize: empty boundary table");

    uint8_t idx = 0;
    for (int i = 0; i < num_boundaries; ++i) {
        if (val > boundaries[i]) {
            idx = static_cast<uint8_t>(i + 1);
        }
    }
    return idx;
}

/// Maps a bin index back to the corresponding Lloyd-Max centroid value.
/// Caller must ensure idx < num_centroids (2^mse_bits).
inline float sqDequantize(uint8_t idx, const float* centroids) {
    return centroids[idx];
}

// ===========================================================================
// Section 6: TurboQuantCode — encoded representation of a single vector
//
// Algorithm 2 output. Stores the compressed representation produced by
// encode(), consumed by asymmetricInnerProduct() and asymmetricL2().
// ===========================================================================

class TurboQuantCode {
public:
    /// SQ indices, one uint8_t per coordinate [dim elements].
    /// Each value in [0, 2^mse_bits - 1].
    /// Note: 1 byte per index is wasteful for 3-bit; tight packing is Phase 2.
    std::vector<uint8_t> sq_packed_;

    /// QJL sign bits, 1 bit per coordinate, packed into uint64_t words.
    /// Bit i of word [i/64] is 1 iff projected residual component i >= 0.
    std::vector<uint64_t> qjl_signs_;

    float gamma_;   ///< ‖residual‖₂ in rotated space
    float norm_;    ///< ‖x‖₂ of the original vector
    float sigma_;   ///< Std dev of rotated coordinates (Lloyd-Max rescaling)
};

// ===========================================================================
// Section 7: TurboQuantEncoder — Algorithm 2 (TurboQuant_prod)
//
// Bit budget per coordinate: b total
//   Stage 1 (MSE): (b-1) bits → 2^(b-1) level Lloyd-Max
//   Stage 2 (QJL): 1 bit → sign(S · residual)
//
// Supported configurations:
//   b=4: 3-bit SQ (8 levels) + 1-bit QJL
//   b=5: 4-bit SQ (16 levels) + 1-bit QJL
// ===========================================================================

class TurboQuantEncoder {
    size_t dim_;
    int total_bits_;
    int mse_bits_;
    uint64_t rotation_seed_;
    uint64_t qjl_seed_;

    const float* boundaries_;
    int num_boundaries_;
    const float* centroids_;

 public:
    TurboQuantEncoder(size_t d, int bits_per_coord = 4,
                      uint64_t rot_seed = 42, uint64_t q_seed = 137)
        : dim_(d)
        , total_bits_(bits_per_coord)
        , mse_bits_(bits_per_coord - 1)
        , rotation_seed_(rot_seed)
        , qjl_seed_(q_seed)
        , boundaries_(nullptr)
        , num_boundaries_(0)
        , centroids_(nullptr)
    {
        assert(d >= 4 && "TurboQuantEncoder: dim must be at least 4");
        assert((d & (d - 1)) == 0
               && "TurboQuantEncoder: dim must be a power of 2");
        assert(bits_per_coord >= 2
               && "TurboQuantEncoder: need at least 2 bits (1 MSE + 1 QJL)");

        if (mse_bits_ == 3) {
            boundaries_ = detail::LM3_BOUNDARIES;
            num_boundaries_ = 7;
            centroids_ = detail::LM3_CENTROIDS;
        } else if (mse_bits_ == 4) {
            boundaries_ = detail::LM4_BOUNDARIES;
            num_boundaries_ = 15;
            centroids_ = detail::LM4_CENTROIDS;
        } else {
            assert(false && "TurboQuantEncoder: only b=4 (3-bit) and b=5 (4-bit) are implemented");
        }
    }

    // -- Accessors ----------------------------------------------------------

    size_t dim() const { return dim_; }
    int totalBits() const { return total_bits_; }
    int mseBits() const { return mse_bits_; }
    uint64_t rotationSeed() const { return rotation_seed_; }
    const float* centroids() const { return centroids_; }

    // -----------------------------------------------------------------------
    // QUANTprod(x) — Algorithm 2: Encode
    //
    // Input:  raw — float[dim], the original vector
    // Output: TurboQuantCode with SQ indices, QJL signs, gamma, norm, sigma
    // -----------------------------------------------------------------------
    TurboQuantCode encode(const float* raw) const {
        assert(raw != nullptr && "encode: null input pointer");

        TurboQuantCode code;

        // Step 1: Compute and store original norm
        float norm_sq = 0.0f;
        for (size_t i = 0; i < dim_; ++i) norm_sq += raw[i] * raw[i];
        code.norm_ = std::sqrt(norm_sq);

        // Step 2: Normalize to unit sphere, then apply PolarQuant rotation (RHT)
        std::vector<float> rotated(dim_);
        float inv_norm = (code.norm_ > 1e-10f) ? (1.0f / code.norm_) : 0.0f;
        for (size_t i = 0; i < dim_; ++i) rotated[i] = raw[i] * inv_norm;
        randomizedHadamard(rotated.data(), dim_, rotation_seed_);

        // Step 3: Compute sigma (std dev of rotated coords) for Lloyd-Max rescaling
        float var = 0.0f;
        for (size_t i = 0; i < dim_; ++i) var += rotated[i] * rotated[i];
        float sigma = std::sqrt(var / static_cast<float>(dim_));
        if (sigma < 1e-10f) sigma = 1e-10f;
        code.sigma_ = sigma;
        float inv_sigma = 1.0f / sigma;

        // Step 4: SQ quantize with (b-1) bits and compute residual
        code.sq_packed_.resize(dim_, 0);
        std::vector<float> residual(dim_);
        for (size_t i = 0; i < dim_; ++i) {
            float normalized = rotated[i] * inv_sigma;
            uint8_t idx = sqQuantize(normalized, boundaries_, num_boundaries_);
            float reconstructed = sqDequantize(idx, centroids_) * sigma;
            residual[i] = rotated[i] - reconstructed;
            code.sq_packed_[i] = idx;
        }

        // Step 5: γ = ‖residual‖₂
        float res_norm_sq = 0.0f;
        for (size_t i = 0; i < dim_; ++i) res_norm_sq += residual[i] * residual[i];
        code.gamma_ = std::sqrt(res_norm_sq);

        // Step 6: QJL sign sketch — qjl_signs = sign(S · residual)
        //         S implemented as RHT with qjl_seed (orthogonal, O(d log d))
        std::vector<float> projected(residual);
        randomizedHadamard(projected.data(), dim_, qjl_seed_);

        size_t num_words = (dim_ + 63) / 64;
        code.qjl_signs_.resize(num_words, 0);
        for (size_t i = 0; i < dim_; ++i) {
            if (projected[i] >= 0.0f) {
                code.qjl_signs_[i / 64] |= (1ULL << (i % 64));
            }
        }

        return code;
    }

    // -----------------------------------------------------------------------
    // Asymmetric inner product: ⟨y, x̃⟩
    //
    // x̃ = x̃_mse + x̃_qjl                     (Algorithm 2 reconstruction)
    // x̃_qjl = (√(π/2) / √d) · γ · Sᵀ · qjl   [S = RHT, orthogonal]
    //
    // In rotated space (rotation is orthogonal → IP preserved):
    //   ⟨ŷ_rot, x̃_rot⟩ = ⟨ŷ_rot, x̃_mse_rot⟩ + (√(π/2)/√d)·γ·⟨S·ŷ_rot, qjl⟩
    //
    // Proof of unbiasedness (Theorem 2):
    //   E[⟨y, x̃⟩] = ⟨y, x̃_mse⟩ + E[(√(π/2)/√d)·γ·⟨S·y, qjl⟩]
    //              = ⟨y, x̃_mse⟩ + ⟨y, r⟩        [by QJL Lemma 4]
    //              = ⟨y, x̃_mse + r⟩ = ⟨y, x⟩
    //
    // Input:  raw_query — float[dim], uncompressed query vector
    //         code      — TurboQuantCode from encode()
    // Output: estimated ⟨raw_query, original_vector⟩
    // -----------------------------------------------------------------------
    float asymmetricInnerProduct(const float* raw_query,
                                 const TurboQuantCode& code) const {
        assert(raw_query != nullptr
               && "asymmetricInnerProduct: null query pointer");
        assert(code.sq_packed_.size() == dim_
               && "asymmetricInnerProduct: code dimension mismatch");

        // Term 0: Normalize and rotate query
        float q_norm_sq = 0.0f;
        for (size_t i = 0; i < dim_; ++i) q_norm_sq += raw_query[i] * raw_query[i];
        float q_norm = std::sqrt(q_norm_sq);
        float q_inv = (q_norm > 1e-10f) ? (1.0f / q_norm) : 0.0f;

        std::vector<float> q_rot(dim_);
        for (size_t i = 0; i < dim_; ++i) q_rot[i] = raw_query[i] * q_inv;
        randomizedHadamard(q_rot.data(), dim_, rotation_seed_);

        // Term 1: ⟨q_rot, x̃_mse_rot⟩  (gather: dequant + dot product)
        float ip_mse = 0.0f;
        for (size_t i = 0; i < dim_; ++i) {
            float centroid_val = sqDequantize(code.sq_packed_[i], centroids_)
                                 * code.sigma_;
            ip_mse += q_rot[i] * centroid_val;
        }

        // Term 2: QJL correction — (√(π/2)/√d) · γ · ⟨S·q_rot, qjl⟩
        std::vector<float> s_q(q_rot);
        randomizedHadamard(s_q.data(), dim_, qjl_seed_);

        // ⟨S·q_rot, qjl⟩: bit=1 → qjl=+1, bit=0 → qjl=-1
        float dot_qjl = 0.0f;
        for (size_t i = 0; i < dim_; ++i) {
            bool positive = (code.qjl_signs_[i / 64] >> (i % 64)) & 1ULL;
            float sign = positive ? 1.0f : -1.0f;
            dot_qjl += s_q[i] * sign;
        }

        // Scale: √(π/2)/√d for orthogonal RHT (not √(π/2)/d as in Gaussian S)
        // See HANDOFF_TURBOQUANT.md "Critical scaling note" for derivation.
        float scale = std::sqrt(static_cast<float>(M_PI) / 2.0f)
                    / std::sqrt(static_cast<float>(dim_));
        float correction = scale * code.gamma_ * dot_qjl;

        // Rescale: IP was computed on unit vectors, restore original norms
        return (ip_mse + correction) * code.norm_ * q_norm;
    }

    /// Asymmetric L2 distance via identity: ‖q-x‖² = ‖q‖² + ‖x‖² - 2⟨q,x⟩
    ///
    /// Input:  raw_query — float[dim], uncompressed query vector
    ///         code      — TurboQuantCode from encode()
    /// Output: estimated squared L2 distance (clamped to >= 0)
    float asymmetricL2(const float* raw_query,
                       const TurboQuantCode& code) const {
        float q_norm_sq = 0.0f;
        for (size_t i = 0; i < dim_; ++i) q_norm_sq += raw_query[i] * raw_query[i];
        float ip = asymmetricInnerProduct(raw_query, code);
        return std::max(0.0f, q_norm_sq + code.norm_ * code.norm_ - 2.0f * ip);
    }

    /// Memory per encoded vector (bytes), current layout (not tight-packed).
    /// Layout: [sq_packed: dim bytes | qjl_signs: ceil(dim/64)*8 | norm,gamma,sigma: 12]
    size_t codeSizeBytes() const {
        size_t sq_bytes = dim_;                        // 1 byte per SQ index
        size_t qjl_bytes = ((dim_ + 63) / 64) * 8;    // d bits packed
        size_t meta = sizeof(float) * 3;               // norm_, gamma_, sigma_
        return sq_bytes + qjl_bytes + meta;
    }

    /// Effective bits per coordinate (including metadata overhead).
    float effectiveBitsPerCoord() const {
        return static_cast<float>(codeSizeBytes() * 8)
             / static_cast<float>(dim_);
    }
};

}  // namespace turboquant
}  // namespace hnswlib
