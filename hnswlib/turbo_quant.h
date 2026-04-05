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

  void applyRandomSigns(float *__restrict__ data, size_t count) {
    size_t i = 0;
    // Main loop: fixed 64-iteration inner loop for autovectorization
    for (; i + 64 <= count; i += 64) {
      uint64_t bits = next();
      for (size_t j = 0; j < 64; ++j) {
        float sign = 1.0f - 2.0f * static_cast<float>((bits >> j) & 1);
        data[i + j] *= sign;
      }
    }
    // Remainder (< 64 elements)
    if (i < count) {
      uint64_t bits = next();
      for (size_t j = 0; i + j < count; ++j) {
        float sign = 1.0f - 2.0f * static_cast<float>((bits >> j) & 1);
        data[i + j] *= sign;
      }
    }
  }
};

// Walsh-Hadamard Transform (WHT)
inline void whtInplace(float *data, size_t const d) {
  assert(d > 0 && (d & (d - 1)) == 0 &&
         "whtInplace: d must be a positive power of 2");

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

  float norm = 1.0f / std::sqrt(static_cast<float>(d));
  for (size_t i = 0; i < d; ++i) {
    data[i] *= norm;
  }
}

inline std::vector<float> generateSigns(size_t const d, uint64_t const seed) {
  std::vector<float> signs(d);
  RndGen64 rng(seed);
  size_t i = 0;
  // Main loop: fixed 64-iteration inner loop for autovectorization
  for (; i + 64 <= d; i += 64) {
    uint64_t bits = rng.next();
    for (size_t j = 0; j < 64; ++j) {
      signs[i + j] = 1.0f - 2.0f * static_cast<float>((bits >> j) & 1);
    }
  }
  // Remainder (< 64 elements)
  if (i < d) {
    uint64_t bits = rng.next();
    for (size_t j = 0; i + j < d; ++j) {
      signs[i + j] = 1.0f - 2.0f * static_cast<float>((bits >> j) & 1);
    }
  }
  return signs;
}

// Randomized Walsh-Hadamard Transform. Elementwise multiply + WHT.
inline void randomizedHadamard(float *data,
                               const float *const __restrict__ signs,
                               size_t const d) {
  assert(d > 0 && (d & (d - 1)) == 0 &&
         "randomizedHadamard: d must be a positive power of 2");

  size_t i = 0;
  for (; i + 64 <= d; i += 64) {
    for (size_t j = 0; j < 64; ++j) {
      data[i + j] *= signs[i + j];
    }
  }
  if (i < d) {
    for (size_t j = 0; i + j < d; ++j) {
      data[i + j] *= signs[i + j];
    }
  }
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
  std::vector<float> boundaries;
  std::vector<float> centroids;
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
  const float *boundaries_;
  int num_boundaries_;
  const float *centroids_;

public:
  std::vector<uint8_t> sq_packed_;
  std::vector<uint64_t> qjl_signs_;
  float gamma_;
  float norm_;
  float sigma_;

  TurboQuantCode()
      : boundaries_(nullptr), num_boundaries_(0), centroids_(nullptr),
        gamma_(0), norm_(0), sigma_(0) {}

  TurboQuantCode(const float *boundaries, int num_boundaries,
                 const float *centroids)
      : boundaries_(boundaries), num_boundaries_(num_boundaries),
        centroids_(centroids), gamma_(0), norm_(0), sigma_(0) {}

  // -- Quantization parameters (read-only after construction) ---------------

  const float *boundaries() const { return boundaries_; }
  int numBoundaries() const { return num_boundaries_; }
  const float *centroids() const { return centroids_; }

  // -- Scalar quantization methods -----------------------------------------

  /// Maps a scalar value to a Lloyd-Max bin index via linear scan.
  /// Input:  val — normalized value (zero-mean, unit-variance expected)
  /// Output: bin index in [0, num_boundaries]
  uint8_t quantize(const float val) const {
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
  void quantizeBatch(const float *__restrict__ vals, const size_t count) {
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
  float dequantize(const uint8_t idx) const { return centroids_[idx]; }

  /// Dequantize the i-th coordinate from sq_packed_.
  float dequantize(const size_t i) const { return centroids_[sq_packed_[i]]; }

  /// Batch dequantization: reconstruct all sq_packed_ values scaled by sigma_.
  /// Output: out[i] = centroids_[sq_packed_[i]] * sigma_
  void dequantizeBatch(float *__restrict__ out, const size_t count) const {
    for (size_t i = 0; i < count; ++i) {
      out[i] = centroids_[sq_packed_[i]] * sigma_;
    }
  }

  // -- Size helpers --------------------------------------------------------

  /// Serialized code size in bytes for a given dimension.
  static size_t codeSizeBytes(size_t dim) {
    return dim + ((dim + 63) / 64) * sizeof(uint64_t) + sizeof(float) * 3;
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
    std::memcpy(buf, &norm_, sizeof(float));
    std::memcpy(buf + sizeof(float), &gamma_, sizeof(float));
    std::memcpy(buf + 2 * sizeof(float), &sigma_, sizeof(float));
  }

  /// Deserialize from a flat byte buffer.
  /// boundaries/centroids must point to the same static tables used for
  /// encoding.
  static TurboQuantCode deserializeFrom(const char *buf, size_t dim,
                                        const float *boundaries,
                                        int num_boundaries,
                                        const float *centroids) {
    TurboQuantCode code(boundaries, num_boundaries, centroids);
    code.sq_packed_.resize(dim);
    std::memcpy(code.sq_packed_.data(), buf, dim);
    buf += dim;
    size_t num_words = (dim + 63) / 64;
    code.qjl_signs_.resize(num_words);
    std::memcpy(code.qjl_signs_.data(), buf, num_words * sizeof(uint64_t));
    buf += num_words * sizeof(uint64_t);
    std::memcpy(&code.norm_, buf, sizeof(float));
    std::memcpy(&code.gamma_, buf + sizeof(float), sizeof(float));
    std::memcpy(&code.sigma_, buf + 2 * sizeof(float), sizeof(float));
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
  const float *boundaries_;
  int num_boundaries_;
  const float *centroids_;

  /// Precomputed sign vectors for RHT — fixed for a given seed, shared by all
  /// vectors.
  std::vector<float> rotation_signs_;
  std::vector<float> qjl_signs_precomp_;

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
  const float *rotationSigns() const { return rotation_signs_.data(); }
  const float *qjlSigns() const { return qjl_signs_precomp_.data(); }
  const float *centroids() const { return centroids_; }

  /// Create an empty TurboQuantCode pre-configured with this encoder's
  /// quantization tables (boundaries/centroids).
  TurboQuantCode createCode() const {
    return TurboQuantCode(boundaries_, num_boundaries_, centroids_);
  }

  // -----------------------------------------------------------------------
  // QUANTprod(x) — Algorithm 2: Encode
  //
  // Input:  raw — float[dim], the original vector
  // Output: TurboQuantCode with SQ indices, QJL signs, gamma, norm, sigma
  // -----------------------------------------------------------------------
  TurboQuantCode encode(const float *raw) const {
    assert(raw != nullptr && "encode: null input pointer");

    TurboQuantCode code = createCode();

    // Step 1: Compute and store original norm
    float norm_sq = 0.0f;
    for (size_t i = 0; i < dim_; ++i)
      norm_sq += raw[i] * raw[i];
    code.norm_ = std::sqrt(norm_sq);

    // Step 2: Normalize to unit sphere, then apply PolarQuant rotation (RHT)
    std::vector<float> rotated(dim_);
    float inv_norm = (code.norm_ > 1e-10f) ? (1.0f / code.norm_) : 0.0f;
    for (size_t i = 0; i < dim_; ++i)
      rotated[i] = raw[i] * inv_norm;
    randomizedHadamard(rotated.data(), rotation_signs_.data(), dim_);

    // Step 3: Compute sigma (std dev of rotated coords) for Lloyd-Max rescaling
    float var = 0.0f;
    for (size_t i = 0; i < dim_; ++i)
      var += rotated[i] * rotated[i];
    float sigma = std::sqrt(var / static_cast<float>(dim_));
    if (sigma < 1e-10f)
      sigma = 1e-10f;
    code.sigma_ = sigma;
    float inv_sigma = 1.0f / sigma;

    // Step 4: SQ quantize with (b-1) bits and compute residual
    std::vector<float> normalized(dim_);
    for (size_t i = 0; i < dim_; ++i)
      normalized[i] = rotated[i] * inv_sigma;
    code.quantizeBatch(normalized.data(), dim_);

    std::vector<float> reconstructed(dim_);
    code.dequantizeBatch(reconstructed.data(), dim_);

    std::vector<float> residual(dim_);
    for (size_t i = 0; i < dim_; ++i) {
      residual[i] = rotated[i] - reconstructed[i];
    }

    // Step 5: γ = ‖residual‖₂
    float res_norm_sq = 0.0f;
    for (size_t i = 0; i < dim_; ++i)
      res_norm_sq += residual[i] * residual[i];
    code.gamma_ = std::sqrt(res_norm_sq);

    // Step 6: QJL sign sketch — qjl_signs = sign(S · residual)
    //         S implemented as RHT with qjl_seed (orthogonal, O(d log d))
    std::vector<float> projected(residual);
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

  void encodeToHNSWBuffer(const float *raw, char *out_buf) const {
    TurboQuantCode code = encode(raw); // ← вся логика уже здесь!
    const size_t d = dim_;

    // sq_packed (уже uint8_t)
    std::memcpy(out_buf, code.sq_packed_.data(), d);

    // QJL: распаковываем из битового формата в int8_t[+1/-1] (то, что нужно
    // HNSW)
    int8_t *qjl_out = reinterpret_cast<int8_t *>(out_buf + d);
    for (size_t i = 0; i < d; ++i) {
      bool positive = (code.qjl_signs_[i / 64] >> (i % 64)) & 1ULL;
      qjl_out[i] = positive ? 1 : -1;
    }

    // meta: norm, gamma, sigma
    float *meta = reinterpret_cast<float *>(out_buf + d + d);
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
  // Input:  raw_query — float[dim], uncompressed query vector
  //         code      — TurboQuantCode from encode()
  // Output: estimated ⟨raw_query, original_vector⟩
  // -----------------------------------------------------------------------
  float asymmetricInnerProduct(const float *raw_query,
                               const TurboQuantCode &code) const {
    assert(raw_query != nullptr &&
           "asymmetricInnerProduct: null query pointer");
    assert(code.sq_packed_.size() == dim_ &&
           "asymmetricInnerProduct: code dimension mismatch");

    // Term 0: Normalize and rotate query
    float q_norm_sq = 0.0f;
    for (size_t i = 0; i < dim_; ++i)
      q_norm_sq += raw_query[i] * raw_query[i];
    float q_norm = std::sqrt(q_norm_sq);
    float q_inv = (q_norm > 1e-10f) ? (1.0f / q_norm) : 0.0f;

    std::vector<float> q_rot(dim_);
    for (size_t i = 0; i < dim_; ++i)
      q_rot[i] = raw_query[i] * q_inv;
    randomizedHadamard(q_rot.data(), rotation_signs_.data(), dim_);

    // Term 1: ⟨q_rot, x̃_mse_rot⟩  (gather: dequant + dot product)
    std::vector<float> mse_recon(dim_);
    code.dequantizeBatch(mse_recon.data(), dim_);
    float ip_mse = 0.0f;
    for (size_t i = 0; i < dim_; ++i) {
      ip_mse += q_rot[i] * mse_recon[i];
    }

    // Term 2: QJL correction — (√(π/2)/√d) · γ · ⟨S·q_rot, qjl⟩
    std::vector<float> s_q(q_rot);
    randomizedHadamard(s_q.data(), qjl_signs_precomp_.data(), dim_);

    // ⟨S·q_rot, qjl⟩: bit=1 → qjl=+1, bit=0 → qjl=-1
    float dot_qjl = 0.0f;
    for (size_t i = 0; i < dim_; ++i) {
      bool positive = (code.qjl_signs_[i / 64] >> (i % 64)) & 1ULL;
      float sign = positive ? 1.0f : -1.0f;
      dot_qjl += s_q[i] * sign;
    }

    // Scale: √(π/2)/√d for orthogonal RHT (not √(π/2)/d as in Gaussian S)
    // See HANDOFF_TURBOQUANT.md "Critical scaling note" for derivation.
    float scale = std::sqrt(static_cast<float>(M_PI) / 2.0f) /
                  std::sqrt(static_cast<float>(dim_));
    float correction = scale * code.gamma_ * dot_qjl;

    // Rescale: IP was computed on unit vectors, restore original norms
    return (ip_mse + correction) * code.norm_ * q_norm;
  }

  /// Asymmetric L2 distance via identity: ‖q-x‖² = ‖q‖² + ‖x‖² - 2⟨q,x⟩
  ///
  /// Input:  raw_query — float[dim], uncompressed query vector
  ///         code      — TurboQuantCode from encode()
  /// Output: estimated squared L2 distance (clamped to >= 0)
  float asymmetricL2(const float *raw_query, const TurboQuantCode &code) const {
    float q_norm_sq = 0.0f;
    for (size_t i = 0; i < dim_; ++i)
      q_norm_sq += raw_query[i] * raw_query[i];
    float ip = asymmetricInnerProduct(raw_query, code);
    return std::max(0.0f, q_norm_sq + code.norm_ * code.norm_ - 2.0f * ip);
  }

  /// Memory per encoded vector (bytes), current layout (not tight-packed).
  /// Layout: [sq_packed: dim bytes | qjl_signs: ceil(dim/64)*8 |
  /// norm,gamma,sigma: 12]
  size_t codeSizeBytes() const {
    size_t sq_bytes = dim_;                    // 1 byte per SQ index
    size_t qjl_bytes = ((dim_ + 63) / 64) * 8; // d bits packed
    size_t meta = sizeof(float) * 3;           // norm_, gamma_, sigma_
    return sq_bytes + qjl_bytes + meta;
  }

  /// Effective bits per coordinate (including metadata overhead).
  float effectiveBitsPerCoord() const {
    return static_cast<float>(codeSizeBytes() * 8) / static_cast<float>(dim_);
  }
};

// ===========================================================================
// Section 8: TurboQuantPreparedQuery — pre-computed query state for search
//
// Created once per query, reused across all candidate comparisons.
// Eliminates 2 RHT calls + 3 heap allocations per distance computation.
// ===========================================================================

struct TurboQuantPreparedQuery {
  std::vector<float> q_rot; ///< normalized + RHT-rotated query (dim floats)
  std::vector<float> s_q;   ///< RHT(q_rot, qjl_signs) for QJL correction
  float q_norm_sq;          ///< ||query||^2
  float q_norm;             ///< ||query||

  /// ADC lookup table: lut[i * num_levels + j] = q_rot[i] * centroids[j].
  /// Precomputed once per query; eliminates per-candidate multiply in
  /// distSearch.
  std::vector<float> lut;
  int num_levels; ///< number of SQ centroid levels (8 or 16)
};

} // namespace turboquant
} // namespace hnswlib
