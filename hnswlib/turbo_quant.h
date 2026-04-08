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
/// Phase: V2 interleaved layout — SQ index and QJL bit packed per coord.
///   HNSW buffer: byte[i] = (sq_idx << 1) | qjl_bit, 1 byte/coord.

#include "hnswlib.h"
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace hnswlib {
namespace turboquant {

class RndGen64 {
  uint64_t state_;

public:
  explicit RndGen64(uint64_t const seed) : state_(seed) {}

  uint64_t next() {
    uint64_t z = (state_ += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
  }

};

// Walsh-Hadamard Transform (WHT) — scalar fallback
inline void whtInplaceScalar(float *data, size_t const d) {
  for (size_t step = 1; step < d; step <<= 1) {
    const size_t jump = step << 1;
    for (size_t i = 0; i < d; i += jump) {
      float *__restrict__ low = &data[i];
      float *__restrict__ high = &data[i + step];
      for (size_t j = 0; j < step; ++j) {
        float a = low[j];
        float b = high[j];
        low[j] = a + b;
        high[j] = a - b;
      }
    }
  }
}

#if defined(USE_AVX)
inline void whtInplaceAVX(float *data, size_t const d) {
  // Small steps (< 8): scalar butterfly
  for (size_t step = 1; step < 8 && step < d; step <<= 1) {
    const size_t jump = step << 1;
    for (size_t i = 0; i < d; i += jump) {
      float *__restrict__ low = &data[i];
      float *__restrict__ high = &data[i + step];
      for (size_t j = 0; j < step; ++j) {
        float a = low[j];
        float b = high[j];
        low[j] = a + b;
        high[j] = a - b;
      }
    }
  }
  // Large steps (>= 8): AVX butterfly, 8 floats at a time
  for (size_t step = 8; step < d; step <<= 1) {
    const size_t jump = step << 1;
    for (size_t i = 0; i < d; i += jump) {
      float *__restrict__ low = &data[i];
      float *__restrict__ high = &data[i + step];
      for (size_t j = 0; j < step; j += 8) {
        __m256 a = _mm256_loadu_ps(low + j);
        __m256 b = _mm256_loadu_ps(high + j);
        _mm256_storeu_ps(low + j, _mm256_add_ps(a, b));
        _mm256_storeu_ps(high + j, _mm256_sub_ps(a, b));
      }
    }
  }
}
#elif defined(USE_NEON)
inline void whtInplaceNEON_fp16(float16_t *__restrict__ data, size_t d) {
  assert((d & (d - 1)) == 0 && "d must be power of 2");

  // === 1. Скалярные маленькие стадии (step < 8) — overhead NEON слишком велик
  // ===
  for (size_t step = 1; step < 8 && step < d; step <<= 1) {
    const size_t jump = step << 1;
    for (size_t i = 0; i < d; i += jump) {
      float16_t *__restrict__ low = data + i;
      float16_t *__restrict__ high = data + i + step;
      for (size_t j = 0; j < step; ++j) {
        float16_t a = low[j];
        float16_t b = high[j];
        low[j] = a + b;
        high[j] = a - b;
      }
    }
  }

  // Large steps (>= 4): NEON butterfly, 4 floats at a time
  for (size_t step = 8; step < d; step <<= 1) {
    const size_t jump = step << 1;

    for (size_t i = 0; i < d; i += jump) {
      float16_t *__restrict__ low = data + i;
      float16_t *__restrict__ high = data + i + step;

      // Unroll ×4 → 32 элемента FP16 (4 × float16x8_t) за итерацию
      for (size_t j = 0; j < step; j += 32) {
        // Butterfly 1
        float16x8_t a0 = vld1q_f16(low + j + 0);
        float16x8_t b0 = vld1q_f16(high + j + 0);
        vst1q_f16(low + j + 0, vaddq_f16(a0, b0));
        vst1q_f16(high + j + 0, vsubq_f16(a0, b0));

        // Butterfly 2
        float16x8_t a1 = vld1q_f16(low + j + 8);
        float16x8_t b1 = vld1q_f16(high + j + 8);
        vst1q_f16(low + j + 8, vaddq_f16(a1, b1));
        vst1q_f16(high + j + 8, vsubq_f16(a1, b1));

        // Butterfly 3
        float16x8_t a2 = vld1q_f16(low + j + 16);
        float16x8_t b2 = vld1q_f16(high + j + 16);
        vst1q_f16(low + j + 16, vaddq_f16(a2, b2));
        vst1q_f16(high + j + 16, vsubq_f16(a2, b2));

        // Butterfly 4
        float16x8_t a3 = vld1q_f16(low + j + 24);
        float16x8_t b3 = vld1q_f16(high + j + 24);
        vst1q_f16(low + j + 24, vaddq_f16(a3, b3));
        vst1q_f16(high + j + 24, vsubq_f16(a3, b3));
      }
    }
  }
}
#elif defined(USE_SSE)
inline void whtInplaceSSE(float *data, size_t const d) {
  // Small steps (< 4): scalar butterfly
  for (size_t step = 1; step < 4 && step < d; step <<= 1) {
    const size_t jump = step << 1;
    for (size_t i = 0; i < d; i += jump) {
      float *__restrict__ low = &data[i];
      float *__restrict__ high = &data[i + step];
      for (size_t j = 0; j < step; ++j) {
        float a = low[j];
        float b = high[j];
        low[j] = a + b;
        high[j] = a - b;
      }
    }
  }
  // Large steps (>= 4): SSE butterfly, 4 floats at a time
  for (size_t step = 4; step < d; step <<= 1) {
    const size_t jump = step << 1;
    for (size_t i = 0; i < d; i += jump) {
      float *__restrict__ low = &data[i];
      float *__restrict__ high = &data[i + step];
      for (size_t j = 0; j < step; j += 4) {
        __m128 a = _mm_loadu_ps(low + j);
        __m128 b = _mm_loadu_ps(high + j);
        _mm_storeu_ps(low + j, _mm_add_ps(a, b));
        _mm_storeu_ps(high + j, _mm_sub_ps(a, b));
      }
    }
  }
}
#endif

// Walsh-Hadamard Transform (WHT)
inline void whtInplace(float16_t *data, size_t const d) {
  assert(d > 0 && (d & (d - 1)) == 0 &&
         "whtInplace: d must be a positive power of 2");

#if defined(USE_AVX)
  whtInplaceAVX(data, d);
#elif defined(USE_NEON)
  whtInplaceNEON_fp16(data, d);
#elif defined(USE_SSE)
  whtInplaceSSE(data, d);
#else
  whtInplaceScalar(data, d);
#endif

  // Normalize
  float32_t norm = 1.0f / std::sqrt(static_cast<float>(d));
#if defined(USE_AVX)
  {
    __m256 vnorm = _mm256_set1_ps(norm);
    for (size_t i = 0; i < d; i += 8)
      _mm256_storeu_ps(data + i,
                       _mm256_mul_ps(_mm256_loadu_ps(data + i), vnorm));
  }
#elif defined(USE_NEON)
  {
    float16x8_t vnorm = vdupq_n_f16(norm);
    for (size_t i = 0; i < d; i += 8)
      vst1q_f16(data + i, vmulq_f16(vld1q_f16(data + i), vnorm));
  }
#elif defined(USE_SSE)
  {
    __m128 vnorm = _mm_set1_ps(norm);
    for (size_t i = 0; i < d; i += 4)
      _mm_storeu_ps(data + i, _mm_mul_ps(_mm_loadu_ps(data + i), vnorm));
  }
#else
  for (size_t i = 0; i < d; ++i)
    data[i] *= norm;
#endif
}

inline std::vector<float16_t> generateSigns(size_t const d, uint64_t const seed) {
  std::vector<float16_t> signs(d);
  RndGen64 rng(seed);
  size_t i = 0;
  // Main loop: fixed 64-iteration inner loop for autovectorization
  for (; i + 64 <= d; i += 64) {
    uint64_t bits = rng.next();
    for (size_t j = 0; j < 64; ++j) {
      signs[i + j] = 1.0f - 2.0f * static_cast<float16_t>((bits >> j) & 1);
    }
  }
  // Remainder (< 64 elements)
  if (i < d) {
    uint64_t bits = rng.next();
    for (size_t j = 0; i + j < d; ++j) {
      signs[i + j] = 1.0f - 2.0f * static_cast<float16_t>((bits >> j) & 1);
    }
  }
  return signs;
}

// Randomized Walsh-Hadamard Transform. Elementwise multiply + WHT.
inline void randomizedHadamard(float16_t *data,
                               const float16_t *const __restrict__ signs,
                               size_t const d) {
  assert(d > 0 && (d & (d - 1)) == 0 &&
         "randomizedHadamard: d must be a positive power of 2");

  // Elementwise multiply: data[i] *= signs[i]
  size_t i = 0;
#if defined(USE_AVX)
  for (; i + 8 <= d; i += 8) {
    __m256 vd = _mm256_loadu_ps(data + i);
    __m256 vs = _mm256_loadu_ps(signs + i);
    _mm256_storeu_ps(data + i, _mm256_mul_ps(vd, vs));
  }
#elif defined(USE_NEON)
  for (; i + 8 <= d; i += 8) {
    float16x8_t vd = vld1q_f16(data + i);
    float16x8_t vs = vld1q_f16(signs + i);
    vst1q_f16(data + i, vmulq_f16(vd, vs));
  }
#elif defined(USE_SSE)
  for (; i + 4 <= d; i += 4) {
    __m128 vd = _mm_loadu_ps(data + i);
    __m128 vs = _mm_loadu_ps(signs + i);
    _mm_storeu_ps(data + i, _mm_mul_ps(vd, vs));
  }
#else
  for (; i + 64 <= d; i += 64) {
    for (size_t j = 0; j < 64; ++j)
      data[i + j] *= signs[i + j];
  }
#endif
  for (; i < d; ++i)
    data[i] *= signs[i];

  whtInplace(data, d);
}

// ===========================================================================
// Lloyd-Max quantizer tables for Gaussian N(0,1)
// Optimal minimum-MSE scalar quantizer (Max, 1960).
// Boundaries = decision thresholds; centroids = reconstruction levels.
// Tables are symmetric around 0.
// ===========================================================================

namespace detail {

// Hardcoded reference values (Max, 1960):
// LM3_CENTROIDS ≈ {-2.1519, -1.3440, -0.7560, -0.2451, 0.2451,
// 0.7560, 1.3440, 2.1519} LM4_CENTROIDS ≈ {-3.0867, -2.0995, -1.6180, -1.2562,
// -0.9423, -0.6568, -0.3881, -0.1284, ...}

struct LloydMaxTable {
  std::vector<float16_t> boundaries;
  std::vector<float16_t> centroids;
};

inline LloydMaxTable computeLloydMax(int bits, int maxIter = 1000,
                                     double tol = 1e-12) {
  const int levels = 1 << bits;
  const int half = levels / 2;

  // φ(x) — standard normal PDF
  auto phi = [](double x) -> double {
    return std::exp(-0.5 * x * x) / std::sqrt(2.0 * M_PI);
  };
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

} // namespace detail

// ===========================================================================
// TurboQuantCode — encoded representation of a single vector
//
// Algorithm 2 output. Stores the compressed representation produced by
// encode(), consumed by asymmetricInnerProduct() and asymmetricL2().
// ===========================================================================

class TurboQuantCode {
private:
  const float16_t *boundaries_;
  int num_boundaries_;
  const float16_t *centroids_;

public:
  std::vector<uint8_t> sq_packed_;
  std::vector<uint64_t> qjl_signs_;
  float16_t gamma_;
  float16_t norm_;
  float16_t sigma_;

  TurboQuantCode()
      : boundaries_(nullptr), num_boundaries_(0), centroids_(nullptr),
        gamma_(0), norm_(0), sigma_(0) {}

  TurboQuantCode(const float16_t *boundaries, int num_boundaries,
                 const float16_t *centroids)
      : boundaries_(boundaries), num_boundaries_(num_boundaries),
        centroids_(centroids), gamma_(0), norm_(0), sigma_(0) {}

  // -- Quantization parameters (read-only after construction) ---------------

  const float16_t *boundaries() const { return boundaries_; }
  int numBoundaries() const { return num_boundaries_; }
  const float16_t *centroids() const { return centroids_; }

  // -- Scalar quantization methods -----------------------------------------

  /// Maps a scalar value to a Lloyd-Max bin index via linear scan.
  /// Input:  val — normalized value (zero-mean, unit-variance expected)
  /// Output: bin index in [0, num_boundaries]
  uint8_t quantize(const float16_t val) const {
    assert(num_boundaries_ > 0 && "quantize: empty boundary table");
    uint8_t idx = 0;
    for (int i = 0; i < num_boundaries_; ++i) {
      if (val > boundaries_[i]) {
        idx = static_cast<uint8_t>(i + 1);
      }
    }
    return idx;
  }

  /// Batch quantization: quantize count values into sq_packed_.
  void quantizeBatch(const float16_t *__restrict__ vals, const size_t count) {
    assert(num_boundaries_ > 0 && "quantizeBatch: empty boundary table");
    sq_packed_.resize(count);
    for (size_t k = 0; k < count; ++k) {
      uint8_t idx = 0;
      for (int i = 0; i < num_boundaries_; ++i) {
        idx += (vals[k] > boundaries_[i]);
      }
      sq_packed_[k] = idx;
    }
  }

  /// Maps a bin index back to the corresponding Lloyd-Max centroid value.
  /// Caller must ensure idx < num_centroids (2^mse_bits).
  float16_t dequantize(const uint8_t idx) const { return centroids_[idx]; }

  /// Dequantize the i-th coordinate from sq_packed_.
  float16_t dequantize(const size_t i) const { return centroids_[sq_packed_[i]]; }

  /// Batch dequantization: reconstruct all sq_packed_ values scaled by sigma_.
  /// Output: out[i] = centroids_[sq_packed_[i]] * sigma_
  void dequantizeBatch(float16_t *__restrict__ out, const size_t count) const {
    for (size_t i = 0; i < count; ++i) {
      out[i] = centroids_[sq_packed_[i]] * sigma_;
    }
  }

  // -- Size helpers --------------------------------------------------------

  /// Serialized code size in bytes for a given dimension.
  static size_t codeSizeBytes(size_t dim) {
    return dim + ((dim + 63) / 64) * sizeof(uint64_t) + sizeof(float16_t) * 3;
  }

  // -- Serialization -------------------------------------------------------

  /// Serialize to a flat byte buffer.
  /// Caller must provide buf of at least codeSizeBytes(dim) bytes.
  /// Layout: [sq_packed (dim)] [qjl_signs (ceil(dim/64)*8)] [norm] [gamma]
  /// [sigma]
  void serializeTo(char *buf, size_t dim) const {
    std::memcpy(buf, sq_packed_.data(), dim);
    buf += dim;
    size_t qjl_bytes = ((dim + 63) / 64) * sizeof(uint64_t);
    std::memcpy(buf, qjl_signs_.data(), qjl_bytes);
    buf += qjl_bytes;
    std::memcpy(buf, &norm_, sizeof(float16_t));
    std::memcpy(buf + sizeof(float16_t), &gamma_, sizeof(float16_t));
    std::memcpy(buf + 2 * sizeof(float16_t), &sigma_, sizeof(float16_t));
  }

  /// Deserialize from a flat byte buffer.
  /// boundaries/centroids must point to the same static tables used for
  /// encoding.
  static TurboQuantCode deserializeFrom(const char *buf, size_t dim,
                                        const float16_t *boundaries,
                                        int num_boundaries,
                                        const float16_t *centroids) {
    TurboQuantCode code(boundaries, num_boundaries, centroids);
    code.sq_packed_.resize(dim);
    std::memcpy(code.sq_packed_.data(), buf, dim);
    buf += dim;
    size_t num_words = (dim + 63) / 64;
    code.qjl_signs_.resize(num_words);
    std::memcpy(code.qjl_signs_.data(), buf, num_words * sizeof(uint64_t));
    buf += num_words * sizeof(uint64_t);
    std::memcpy(&code.norm_, buf, sizeof(float16_t));
    std::memcpy(&code.gamma_, buf + sizeof(float16_t), sizeof(float16_t));
    std::memcpy(&code.sigma_, buf + 2 * sizeof(float16_t), sizeof(float16_t));
    return code;
  }
};

// ===========================================================================
// Section 7: TurboQuantEncoder — Algorithm 2 (TurboQuant_prod)
//
// Bit budget per coordinate: b total
//   Stage 1 (MSE): (b-1) bits → 2^(b-1) level Lloyd-Max
//   Stage 2 (QJL): 1 bit → sign(S · residual)
//
// Supported configurations (b = bits_per_coord, MSE = b-1 bits, QJL = 1 bit):
//   b=2..9: (b-1)-bit SQ (2^(b-1) levels) + 1-bit QJL
//   e.g. b=4: 3-bit SQ (8 levels), b=8: 7-bit SQ (128 levels)
// ===========================================================================

class TurboQuantEncoder {
  const size_t dim_;
  const int total_bits_;
  const int mse_bits_;
  const uint64_t rotation_seed_;
  const uint64_t qjl_seed_;

  detail::LloydMaxTable lm_table_;
  const float16_t *boundaries_;
  int num_boundaries_;
  const float16_t *centroids_;

  /// Precomputed sign vectors for RHT — fixed for a given seed, shared by all
  /// vectors.
  std::vector<float16_t> rotation_signs_;
  std::vector<float16_t> qjl_signs_precomp_;

public:
  TurboQuantEncoder(size_t d, int bits_per_coord = 4, uint64_t rot_seed = 42,
                    uint64_t q_seed = 137)
      : dim_(d), total_bits_(bits_per_coord), mse_bits_(bits_per_coord - 1),
        rotation_seed_(rot_seed), qjl_seed_(q_seed), boundaries_(nullptr),
        num_boundaries_(0), centroids_(nullptr) {
    assert(d >= 4 && "TurboQuantEncoder: dim must be at least 4");
    assert((d & (d - 1)) == 0 && "TurboQuantEncoder: dim must be a power of 2");
    assert(bits_per_coord >= 2 &&
           "TurboQuantEncoder: need at least 2 bits (1 MSE + 1 QJL)");
    assert(bits_per_coord <= 9 &&
           "TurboQuantEncoder: max 9 bits (8-bit MSE + 1 QJL, uint8_t limit)");

    lm_table_ = detail::computeLloydMax(mse_bits_);
    boundaries_ = lm_table_.boundaries.data();
    num_boundaries_ = static_cast<int>(lm_table_.boundaries.size());
    centroids_ = lm_table_.centroids.data();

    rotation_signs_ = generateSigns(dim_, rotation_seed_);
    qjl_signs_precomp_ = generateSigns(dim_, qjl_seed_);
  }

  // -- Accessors ----------------------------------------------------------

  size_t dim() const { return dim_; }
  int totalBits() const { return total_bits_; }
  int mseBits() const { return mse_bits_; }
  const float16_t *rotationSigns() const { return rotation_signs_.data(); }
  const float16_t *qjlSigns() const { return qjl_signs_precomp_.data(); }
  const float16_t *centroids() const { return centroids_; }

  /// Create an empty TurboQuantCode pre-configured with this encoder's
  /// quantization tables (boundaries/centroids).
  TurboQuantCode createCode() const {
    return TurboQuantCode(boundaries_, num_boundaries_, centroids_);
  }

  // -----------------------------------------------------------------------
  // QUANTprod(x) — Algorithm 2: Encode
  //
  // Input:  raw — float16_t[dim], the original vector
  // Output: TurboQuantCode with SQ indices, QJL signs, gamma, norm, sigma
  // -----------------------------------------------------------------------
  TurboQuantCode encode(const float16_t *raw) const {
    assert(raw != nullptr && "encode: null input pointer");

    TurboQuantCode code = createCode();

    // Step 1: Compute and store original norm
    float16_t norm_sq = 0.0f;
    for (size_t i = 0; i < dim_; ++i)
      norm_sq += raw[i] * raw[i];
    code.norm_ = std::sqrt(norm_sq);

    // Step 2: Normalize to unit sphere, then apply PolarQuant rotation (RHT)
    std::vector<float16_t> rotated(dim_);
    float16_t inv_norm = (code.norm_ > 1e-10f) ? (1.0f / code.norm_) : 0.0f;
    for (size_t i = 0; i < dim_; ++i)
      rotated[i] = raw[i] * inv_norm;
    randomizedHadamard(rotated.data(), rotation_signs_.data(), dim_);

    // Step 3: Compute sigma (std dev of rotated coords) for Lloyd-Max rescaling
    float16_t var = 0.0f;
    for (size_t i = 0; i < dim_; ++i)
      var += rotated[i] * rotated[i];
    float16_t sigma = std::sqrt(var / static_cast<float16_t>(dim_));
    if (sigma < 1e-10f)
      sigma = 1e-10f;
    code.sigma_ = sigma;
    float16_t inv_sigma = 1.0f / sigma;

    // Step 4: SQ quantize with (b-1) bits and compute residual
    std::vector<float16_t> normalized(dim_);
    for (size_t i = 0; i < dim_; ++i)
      normalized[i] = rotated[i] * inv_sigma;
    code.quantizeBatch(normalized.data(), dim_);

    std::vector<float16_t> reconstructed(dim_);
    code.dequantizeBatch(reconstructed.data(), dim_);

    std::vector<float16_t> residual(dim_);
    for (size_t i = 0; i < dim_; ++i) {
      residual[i] = rotated[i] - reconstructed[i];
    }

    // Step 5: γ = ‖residual‖₂
    float16_t res_norm_sq = 0.0f;
    for (size_t i = 0; i < dim_; ++i)
      res_norm_sq += residual[i] * residual[i];
    code.gamma_ = std::sqrt(res_norm_sq);

    // Step 6: QJL sign sketch — qjl_signs = sign(S · residual)
    //         S implemented as RHT with qjl_seed (orthogonal, O(d log d))
    std::vector<float16_t> projected(residual);
    randomizedHadamard(projected.data(), qjl_signs_precomp_.data(), dim_);

    size_t num_words = (dim_ + 63) / 64;
    code.qjl_signs_.resize(num_words, 0);
    for (size_t i = 0; i < dim_; ++i) {
      if (projected[i] >= 0.0f) {
        code.qjl_signs_[i / 64] |= (1ULL << (i % 64));
      }
    }

    return code;
  }

  /// Encode a float16_t vector into the HNSW interleaved buffer format.
  /// Layout: [packed: dim bytes] [meta: 12 bytes (norm, gamma, sigma)]
  /// packed[i] = (sq_idx << 1) | qjl_bit
  /// Total size: dim + 12 bytes. Caller must provide out_buf of this size.
  void encodeToHNSWBuffer(const float16_t *raw, char *out_buf) const {
    TurboQuantCode code = encode(raw);
    const size_t d = dim_;

    uint8_t *packed = reinterpret_cast<uint8_t *>(out_buf);
    for (size_t i = 0; i < d; ++i) {
      uint8_t qjl_bit = (code.qjl_signs_[i / 64] >> (i % 64)) & 1ULL;
      packed[i] = (code.sq_packed_[i] << 1) | qjl_bit;
    }

    // meta: norm, gamma, sigma (immediately after interleaved data)
    float16_t *meta = reinterpret_cast<float16_t *>(out_buf + d);
    meta[0] = code.norm_;
    meta[1] = code.gamma_;
    meta[2] = code.sigma_;
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
  // Input:  raw_query — float16_t[dim], uncompressed query vector
  //         code      — TurboQuantCode from encode()
  // Output: estimated ⟨raw_query, original_vector⟩
  // -----------------------------------------------------------------------
  float16_t asymmetricInnerProduct(const float16_t *raw_query,
                               const TurboQuantCode &code) const {
    assert(raw_query != nullptr &&
           "asymmetricInnerProduct: null query pointer");
    assert(code.sq_packed_.size() == dim_ &&
           "asymmetricInnerProduct: code dimension mismatch");

    // Term 0: Normalize and rotate query
    float16_t q_norm_sq = 0.0f;
    for (size_t i = 0; i < dim_; ++i)
      q_norm_sq += raw_query[i] * raw_query[i];
    float16_t q_norm = std::sqrt(q_norm_sq);
    float16_t q_inv = (q_norm > 1e-10f) ? (1.0f / q_norm) : 0.0f;

    std::vector<float16_t> q_rot(dim_);
    for (size_t i = 0; i < dim_; ++i)
      q_rot[i] = raw_query[i] * q_inv;
    randomizedHadamard(q_rot.data(), rotation_signs_.data(), dim_);

    // Term 1: ⟨q_rot, x̃_mse_rot⟩  (gather: dequant + dot product)
    std::vector<float16_t> mse_recon(dim_);
    code.dequantizeBatch(mse_recon.data(), dim_);
    float16_t ip_mse = 0.0f;
    for (size_t i = 0; i < dim_; ++i) {
      ip_mse += q_rot[i] * mse_recon[i];
    }

    // Term 2: QJL correction — (√(π/2)/√d) · γ · ⟨S·q_rot, qjl⟩
    std::vector<float16_t> s_q(q_rot);
    randomizedHadamard(s_q.data(), qjl_signs_precomp_.data(), dim_);

    // ⟨S·q_rot, qjl⟩: bit=1 → qjl=+1, bit=0 → qjl=-1
    float16_t dot_qjl = 0.0f;
    for (size_t i = 0; i < dim_; ++i) {
      bool positive = (code.qjl_signs_[i / 64] >> (i % 64)) & 1ULL;
      float16_t sign = positive ? 1.0f : -1.0f;
      dot_qjl += s_q[i] * sign;
    }

    // Scale: √(π/2)/√d for orthogonal RHT (not √(π/2)/d as in Gaussian S)
    // See HANDOFF_TURBOQUANT.md "Critical scaling note" for derivation.
    float16_t scale = std::sqrt(static_cast<float16_t>(M_PI) / 2.0f) /
                  std::sqrt(static_cast<float16_t>(dim_));
    float16_t correction = scale * code.gamma_ * dot_qjl;

    // Rescale: IP was computed on unit vectors, restore original norms
    return (ip_mse + correction) * code.norm_ * q_norm;
  }

  /// Asymmetric L2 distance via identity: ‖q-x‖² = ‖q‖² + ‖x‖² - 2⟨q,x⟩
  ///
  /// Input:  raw_query — float16_t[dim], uncompressed query vector
  ///         code      — TurboQuantCode from encode()
  /// Output: estimated squared L2 distance (clamped to >= 0)
  float16_t asymmetricL2(const float16_t *raw_query, const TurboQuantCode &code) const {
    float16_t q_norm_sq = 0.0f;
    for (size_t i = 0; i < dim_; ++i)
      q_norm_sq += raw_query[i] * raw_query[i];
    float16_t ip = asymmetricInnerProduct(raw_query, code);
    return std::max(0.0f, q_norm_sq + code.norm_ * code.norm_ - 2.0f * ip);
  }

  /// Memory per encoded vector (bytes), interleaved layout.
  /// Layout: [interleaved: dim bytes (sq_idx<<1|qjl_bit)] [meta: 12 bytes]
  size_t codeSizeBytes() const { return dim_ + sizeof(float16_t) * 3; }

  /// Effective bits per coordinate (including metadata overhead).
  float16_t effectiveBitsPerCoord() const {
    return static_cast<float16_t>(codeSizeBytes() * 8) / static_cast<float16_t>(dim_);
  }
};

// ===========================================================================
// Section 8: TurboQuantPreparedQuery — pre-computed query state for search
//
// Created once per query, reused across all candidate comparisons.
// Eliminates 2 RHT calls + 3 heap allocations per distance computation.
// ===========================================================================

struct TurboQuantPreparedQuery {
  std::vector<float16_t> q_rot; ///< normalized + RHT-rotated query (dim floats)
  std::vector<float16_t> s_q;   ///< RHT(q_rot, qjl_signs) for QJL correction
  float q_norm_sq;          ///< ||query||^2
  float q_norm;             ///< ||query||

  const float *centroids; ///< pointer to encoder centroids (NOT owned)
};

} // namespace turboquant
} // namespace hnswlib
