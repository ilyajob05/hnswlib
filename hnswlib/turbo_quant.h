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

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "hnswlib.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace hnswlib {
namespace turboquant {

// ===========================================================================
// TurboQuantCode — encoded representation of a single vector
// ===========================================================================

class TurboQuantCode {
public:
    float gamma_;
    float norm_;
    float sigma_;
    uint8_t *sq_packed_;

    TurboQuantCode(const size_t dim)
        : gamma_(0)
        , norm_(0)
        , sigma_(0)
    {}

    inline void set(size_t i, uint8_t sq_idx, bool qjl_positive)
    {
        sq_packed_[i] = (sq_idx << 1) | static_cast<uint8_t>(qjl_positive);
    }

  inline uint8_t sqIndex(size_t i) const { return sq_packed_[i] >> 1; }

  inline bool qjlSign(size_t i) const { return sq_packed_[i] & 1; }

  inline uint8_t quantize(const float val) const
  {
      assert(num_boundaries_ > 0 && "quantize: empty boundary table");
      uint8_t idx = 0;
      for (int i = 0; i < num_boundaries_; ++i) {
          if (val > boundaries_[i]) {
              idx = static_cast<uint8_t>(i + 1);
          }
      }
      return idx;
  }

  void quantizeEmb(const float *__restrict__ vals, const size_t count)
  {
      assert(num_boundaries_ > 0 && "quantizeBatch: empty boundary table");
      sq_packed_.resize(count);
      for (size_t k = 0; k < count; ++k) {
          sq_packed_[k] = quantize(vals[k]);
      }
  }

  inline float dequantize_centroids(const uint8_t idx) const { return centroids_[idx]; }

  inline float dequantize_term(const size_t i) const {
    return dequantize_centroids(sq_packed_[i]);
  }

  void dequantizeBatch(float *__restrict__ out, const size_t count) const {
    for (size_t i = 0; i < count; ++i) {
      out[i] = dequantize_term(i) * sigma_;
    }
  }

  // -- Size helpers --------------------------------------------------------

  /// Serialized code size in bytes for a given dimension.
  /// Layout: [sq_packed: dim bytes] [qjl_signs: ceil(dim/64)*8 bytes] [norm, gamma, sigma: 3×float]
  static size_t codeSizeBytes(size_t dim) {
    return dim + ((dim + 63) / 64) * sizeof(uint64_t) + sizeof(float) * 3;
  }

  // -- Serialization -------------------------------------------------------

  /// Serialize to a flat byte buffer.
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

  const float *boundaries_;
  const int num_boundaries_;
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

  /// Encode a float vector into the HNSW interleaved buffer format.
  /// Layout: [packed: dim bytes] [meta: 12 bytes (norm, gamma, sigma)]
  /// packed[i] = (sq_idx << 1) | qjl_bit
  /// Total size: dim + 12 bytes. Caller must provide out_buf of this size.
  void encodeToHNSWBuffer(const float *raw, char *out_buf) const {
    TurboQuantCode code = encode(raw);
    const size_t d = dim_;

    uint8_t *packed = reinterpret_cast<uint8_t *>(out_buf);
    for (size_t i = 0; i < d; ++i) {
      uint8_t qjl_bit = (code.qjl_signs_[i / 64] >> (i % 64)) & 1ULL;
      packed[i] = (code.sq_packed_[i] << 1) | qjl_bit;
    }

    // meta: norm, gamma, sigma (immediately after interleaved data)
    float *meta = reinterpret_cast<float *>(out_buf + d);
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
  float asymmetricL2(const float *raw_query,
                         const TurboQuantCode &code) const {
    float q_norm_sq = 0.0f;
    for (size_t i = 0; i < dim_; ++i)
      q_norm_sq += raw_query[i] * raw_query[i];
    float ip = asymmetricInnerProduct(raw_query, code);
    return std::max(0.0f, q_norm_sq + code.norm_ * code.norm_ - 2.0f * ip);
  }

  /// Memory per encoded vector (bytes), interleaved layout.
  /// Layout: [interleaved: dim bytes (sq_idx<<1|qjl_bit)] [meta: 12 bytes]
  size_t codeSizeBytes() const { return dim_ + sizeof(float) * 3; }

  /// Effective bits per coordinate (including metadata overhead).
  float effectiveBitsPerCoord() const {
    return static_cast<float>(codeSizeBytes() * 8) /
           static_cast<float>(dim_);
  }
};

} // namespace turboquant
} // namespace hnswlib
