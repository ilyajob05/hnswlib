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
///   5. Search with beginSearch/searchKnn/endSearch
///   6. Re-rank: loadRawVectors for shortlist, compute exact L2, pick top-K
///
/// Opt-in header: not included by hnswlib.h. Include directly when needed.

#include "hnswlib.h"
#include "turbo_quant.h"

#include <algorithm>
#include <cstring>
#include <fstream>

namespace hnswlib {
namespace turboquant {

// ===========================================================================
// TurboQuantSpace — SpaceInterface adapter for HNSW
//
// HNSW buffer layout (defined by encodeToHNSWBuffer):
//   [sq_packed: dim bytes (uint8_t)] [qjl_signs: dim bytes (int8_t ±1)] [meta:
//   3 floats] meta[0] = norm, meta[1] = gamma, meta[2] = sigma Total: dim + dim
//   + 12 bytes
//
// Distance modes:
//   - Asymmetric (search): float query × TQ code. Set via beginSearch().
//   - Symmetric (build):   TQ code × TQ code. Used when prepared_query_ is
//   null.
//
// dist_func_param = this pointer. Static distance functions cast back to
// const TurboQuantSpace* and access encoder/fields directly — no data
// duplication.
//
// Thread safety: beginSearch/endSearch modify prepared_query_.
//   For concurrent search, use one TurboQuantSpace per thread.
// ===========================================================================

class TurboQuantSpace : public SpaceInterface<float> {
  TurboQuantEncoder encoder_;
  size_t dim_;
  size_t code_size_; ///< HNSW buffer size: dim + dim + 12
  int num_levels_;   ///< 2^mse_bits SQ centroid levels
  float scale_;      ///< √(π/2) / √d for QJL correction

  /// Non-null during search → asymmetric mode. Null → symmetric.
  const TurboQuantPreparedQuery *prepared_query_;

  // -- Static distance functions -------------------------------------------

  /// Asymmetric distance: prepared float query × compressed code.
  /// Uses precomputed LUT for SQ part, avoids per-element multiply.
  static float distSearch(const TurboQuantPreparedQuery *pq,
                          const char *code_buf, const TurboQuantSpace *space) {
    const size_t dim = space->dim_;
    const uint8_t *sq_packed = reinterpret_cast<const uint8_t *>(code_buf);
    const int8_t *qjl_signs = reinterpret_cast<const int8_t *>(code_buf + dim);
    const float *meta = reinterpret_cast<const float *>(code_buf + dim + dim);
    const float x_norm = meta[0];
    const float gamma = meta[1];
    const float sigma = meta[2];

    // Term 1: SQ inner product via ADC lookup table
    float ip_mse = 0.0f;
    const int num_levels = pq->num_levels;
    const float *lut = pq->lut.data();
    for (size_t i = 0; i < dim; ++i)
      ip_mse += lut[i * num_levels + sq_packed[i]];
    ip_mse *= sigma;

    // Term 2: QJL correction
    float dot_qjl = 0.0f;
    for (size_t i = 0; i < dim; ++i)
      dot_qjl += pq->s_q[i] * qjl_signs[i];
    const float correction = space->scale_ * gamma * dot_qjl;

    // L2 = ||q||² + ||x||² - 2·IP
    const float ip = (ip_mse + correction) * x_norm * pq->q_norm;
    return std::max(0.0f, pq->q_norm_sq + x_norm * x_norm - 2.0f * ip);
  }

  /// Symmetric distance: both sides are compressed codes.
  /// Used during graph traversal when both nodes are compressed.
  static float distSymmetric(const char *buf_a, const char *buf_b,
                             const TurboQuantSpace *space) {
    const size_t dim = space->dim_;
    const float *centroids = space->encoder_.centroids();

    const uint8_t *sq_a = reinterpret_cast<const uint8_t *>(buf_a);
    const float *meta_a = reinterpret_cast<const float *>(buf_a + dim + dim);
    const float norm_a = meta_a[0];
    const float sigma_a = meta_a[2];

    const uint8_t *sq_b = reinterpret_cast<const uint8_t *>(buf_b);
    const float *meta_b = reinterpret_cast<const float *>(buf_b + dim + dim);
    const float norm_b = meta_b[0];
    const float sigma_b = meta_b[2];

    // Approximate IP in rotated space using centroids lookup
    float ip_rot = 0.0f;
    for (size_t i = 0; i < dim; ++i)
      ip_rot += (centroids[sq_a[i]] * sigma_a) * (centroids[sq_b[i]] * sigma_b);

    const float ip = ip_rot * norm_a * norm_b;
    return std::max(0.0f, norm_a * norm_a + norm_b * norm_b - 2.0f * ip);
  }

  /// Dispatch: asymmetric if prepared_query_ is set, else symmetric.
  static float turboQuantL2(const void *pVect1, const void *pVect2,
                            const void *param_ptr) {
    const auto *space = static_cast<const TurboQuantSpace *>(param_ptr);
    const char *buf2 = static_cast<const char *>(pVect2);
    if (space->prepared_query_ != nullptr)
      return distSearch(space->prepared_query_, buf2, space);
    else
      return distSymmetric(static_cast<const char *>(pVect1), buf2, space);
  }

public:
  TurboQuantSpace(size_t dim, int bits_per_coord = 4, uint64_t rot_seed = 42,
                  uint64_t qjl_seed = 137)
      : encoder_(dim, bits_per_coord, rot_seed, qjl_seed), dim_(dim),
        code_size_(dim + dim + sizeof(float) * 3) // HNSW buffer layout
        ,
        num_levels_(1 << (bits_per_coord - 1)),
        scale_(std::sqrtf(static_cast<float>(M_PI) / 2.0f) /
               std::sqrtf(static_cast<float>(dim))),
        prepared_query_(nullptr) {
    assert(dim >= 4 && "TurboQuantSpace: dim must be at least 4");
    assert((dim & (dim - 1)) == 0 &&
           "TurboQuantSpace: dim must be a power of 2");
  }

  // -- SpaceInterface -------------------------------------------------------

  size_t get_data_size() override { return code_size_; }
  DISTFUNC<float> get_dist_func() override { return &turboQuantL2; }
  void *get_dist_func_param() override { return this; }

  // -- Accessors ------------------------------------------------------------

  size_t dim() const { return dim_; }
  size_t codeSizeBytes() const { return code_size_; }
  int numLevels() const { return num_levels_; }
  float scale() const { return scale_; }
  const TurboQuantEncoder &encoder() const { return encoder_; }

  // -- Encoding -------------------------------------------------------------

  void encodeVector(const float *raw, char *out_buf) const {
    encoder_.encodeToHNSWBuffer(raw, out_buf);
  }

  // -- Search state management ----------------------------------------------

  /// Prepare query: amortizes 2 RHT calls across all distance computations.
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
    randomizedHadamard(pq.q_rot.data(), encoder_.rotationSigns(), d);

    // ADC lookup table: lut[i * num_levels + j] = q_rot[i] * centroids[j]
    pq.num_levels = num_levels_;
    pq.lut.resize(d * num_levels_);
    const float *centroids = encoder_.centroids();
    for (size_t i = 0; i < d; ++i) {
      for (int j = 0; j < num_levels_; ++j) {
        pq.lut[i * num_levels_ + j] = pq.q_rot[i] * centroids[j];
      }
    }

    // QJL projection of query for correction term
    pq.s_q = pq.q_rot;
    randomizedHadamard(pq.s_q.data(), encoder_.qjlSigns(), d);

    return pq;
  }

  /// Set distance function to asymmetric mode for the given prepared query.
  /// Must be paired with endSearch(). Not thread-safe.
  void beginSearch(const TurboQuantPreparedQuery &pq) { prepared_query_ = &pq; }

  /// Reset distance function to symmetric mode.
  void endSearch() { prepared_query_ = nullptr; }
};

// ===========================================================================
// Compress/Save/Load utilities
// ===========================================================================

/// Raw vectors file magic and header size.
static const uint32_t TQRV_MAGIC = 0x54515256; // "TQRV"
static const uint32_t TQRV_VERSION = 1;
static const uint64_t TQRV_DATA_OFFSET = 32;

/// Save raw float vectors from an L2-built HNSW index to a flat file.
/// Must be called BEFORE compressIndex (while data slots still contain floats).
///
/// File format (.tqrv):
///   [magic: u32] [version: u32] [num_vectors: u64] [dim: u64]
///   [data_offset: u64 = 32] [pad: 4 bytes]
///   [vectors: N * dim * sizeof(float), contiguous by internal ID]
///
/// Vector i is at file offset: data_offset + i * dim * sizeof(float).
inline Status saveRawVectors(const std::string &path,
                             const HierarchicalNSW<float> &hnsw, size_t dim) {

  std::ofstream out(path, std::ios::binary);
  if (!out.good())
    return Status("saveRawVectors: cannot open output file");

  size_t n = hnsw.cur_element_count.load();

  // Header (32 bytes)
  out.write(reinterpret_cast<const char *>(&TQRV_MAGIC), 4);
  out.write(reinterpret_cast<const char *>(&TQRV_VERSION), 4);
  uint64_t n64 = static_cast<uint64_t>(n);
  uint64_t dim64 = static_cast<uint64_t>(dim);
  out.write(reinterpret_cast<const char *>(&n64), 8);
  out.write(reinterpret_cast<const char *>(&dim64), 8);
  out.write(reinterpret_cast<const char *>(&TQRV_DATA_OFFSET), 8);

  // Vectors (N * dim * 4 bytes)
  for (size_t i = 0; i < n; ++i) {
    const char *data = hnsw.getDataByInternalId(static_cast<tableint>(i));
    out.write(data, dim * sizeof(float));
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

  // Switch distance function to TQ
  hnsw.fstdistfunc_ = tq_space.get_dist_func();
  hnsw.dist_func_param_ = tq_space.get_dist_func_param();

  return OkStatus();
}

/// Load specific raw vectors by internal ID from a .tqrv file.
/// Does NOT load the entire file — seeks to each requested vector.
///
/// Parameters:
///   path     — path to .tqrv file (created by saveRawVectors)
///   ids      — array of internal IDs to load
///   num_ids  — number of IDs
///   dim      — vector dimension (must match file)
///   out      — pre-allocated buffer for num_ids * dim floats
inline Status loadRawVectors(const std::string &path, const size_t *ids,
                             size_t num_ids, size_t dim, float *out) {

  std::ifstream in(path, std::ios::binary);
  if (!in.good())
    return Status("loadRawVectors: cannot open file");

  // Read and validate header
  uint32_t magic, version;
  uint64_t n64, dim64, data_offset;
  in.read(reinterpret_cast<char *>(&magic), 4);
  in.read(reinterpret_cast<char *>(&version), 4);
  in.read(reinterpret_cast<char *>(&n64), 8);
  in.read(reinterpret_cast<char *>(&dim64), 8);
  in.read(reinterpret_cast<char *>(&data_offset), 8);

  if (magic != TQRV_MAGIC)
    return Status("loadRawVectors: invalid file magic");
  if (version != TQRV_VERSION)
    return Status("loadRawVectors: unsupported version");
  if (dim64 != static_cast<uint64_t>(dim))
    return Status("loadRawVectors: dimension mismatch");

  size_t vec_bytes = dim * sizeof(float);

  for (size_t i = 0; i < num_ids; ++i) {
    if (ids[i] >= n64)
      return Status("loadRawVectors: ID out of range");

    std::streamoff offset =
        static_cast<std::streamoff>(data_offset + ids[i] * vec_bytes);
    in.seekg(offset, std::ios::beg);
    in.read(reinterpret_cast<char *>(out + i * dim), vec_bytes);

    if (!in.good())
      return Status("loadRawVectors: read error");
  }

  return OkStatus();
}

} // namespace turboquant
} // namespace hnswlib