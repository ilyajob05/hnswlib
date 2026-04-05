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
#include <thread>

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
//
// HNSW buffer layout (defined by encodeToHNSWBuffer):
//   [sq_packed: dim bytes (uint8_t)] [qjl_signs: dim bytes (int8_t ±1)] [meta:
//   3 floats] meta[0] = norm, meta[1] = gamma, meta[2] = sigma Total: dim + dim
//   + 12 bytes
//
// Distance modes (two separate functions, explicitly switched):
//   - turboQuantL2Build  (symmetric):  TQ code × TQ code. Default for addPoint.
//   - turboQuantL2Search (asymmetric): PreparedQuery* × TQ code. For searchKnn.
//
// dist_func_param = this pointer. Static distance functions cast back to
// const TurboQuantSpace* and access encoder/fields directly.
//
// Thread safety:
//   Build: addPoint is thread-safe (HNSW internal locks).
//   Search: fully thread-safe after setSearchMode(). Each thread creates its
//   own TurboQuantPreparedQuery on the stack and passes &pq as query_data
//   to searchKnn. No shared mutable state.
// ===========================================================================

class TurboQuantSpace : public SpaceInterface<float> {
  TurboQuantEncoder encoder_;
  size_t dim_;
  size_t code_size_; ///< HNSW buffer size: dim + dim + 12
  int num_levels_;   ///< 2^mse_bits SQ centroid levels
  float scale_;      ///< √(π/2) / √d for QJL correction

  // -- Static distance functions -------------------------------------------

  /// Asymmetric distance: pVect1 = TurboQuantPreparedQuery*, pVect2 = TQ code.
  /// Uses precomputed LUT for SQ part, avoids per-element multiply.
  static float distSearchImpl(const TurboQuantPreparedQuery *pq,
                               const char *code_buf,
                               const TurboQuantSpace *space) {
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

  /// Build distance: both pVect1 and pVect2 are TQ codes (symmetric).
  static float turboQuantL2Build(const void *pVect1, const void *pVect2,
                                 const void *param_ptr) {
    const auto *space = static_cast<const TurboQuantSpace *>(param_ptr);
    const size_t dim = space->dim_;
    const float *centroids = space->encoder_.centroids();

    const char *buf_a = static_cast<const char *>(pVect1);
    const char *buf_b = static_cast<const char *>(pVect2);

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

  /// Search distance: pVect1 = TurboQuantPreparedQuery*, pVect2 = TQ code.
  static float turboQuantL2Search(const void *pVect1, const void *pVect2,
                                  const void *param_ptr) {
    const auto *space = static_cast<const TurboQuantSpace *>(param_ptr);
    const auto *pq = static_cast<const TurboQuantPreparedQuery *>(pVect1);
    return distSearchImpl(pq, static_cast<const char *>(pVect2), space);
  }

public:
  TurboQuantSpace(size_t dim, int bits_per_coord = 4, uint64_t rot_seed = 42,
                  uint64_t qjl_seed = 137)
      : encoder_(dim, bits_per_coord, rot_seed, qjl_seed), dim_(dim),
        code_size_(dim + dim + sizeof(float) * 3),
        num_levels_(1 << (bits_per_coord - 1)),
        scale_(std::sqrtf(static_cast<float>(M_PI) / 2.0f) /
               std::sqrtf(static_cast<float>(dim))) {
    assert(dim >= 4 && "TurboQuantSpace: dim must be at least 4");
    assert((dim & (dim - 1)) == 0 &&
           "TurboQuantSpace: dim must be a power of 2");
  }

  // -- SpaceInterface -------------------------------------------------------

  size_t get_data_size() override { return code_size_; }
  DISTFUNC<float> get_dist_func() override { return &turboQuantL2Build; }
  DISTFUNC<float> getSearchDistFunc() const { return &turboQuantL2Search; }
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

  // -- Mode switching -------------------------------------------------------
  //
  // Two-phase workflow:
  //   1. Build: hnsw uses turboQuantL2Build (default from get_dist_func).
  //   2. Search: call setSearchMode(hnsw) once after build completes.
  //      Then each thread does:
  //        auto pq = space.prepareQuery(raw_query);
  //        auto result = hnsw.searchKnn(&pq, K);
  //      No shared mutable state — fully thread-safe.

  /// Switch HNSW to search mode: asymmetric distance (PreparedQuery* × code).
  void setSearchMode(HierarchicalNSW<float> &hnsw) const {
    hnsw.fstdistfunc_ = &turboQuantL2Search;
    hnsw.dist_func_param_ = const_cast<TurboQuantSpace *>(this);
  }

  /// Switch HNSW back to build mode: symmetric distance (code × code).
  void setBuildMode(HierarchicalNSW<float> &hnsw) const {
    hnsw.fstdistfunc_ = &turboQuantL2Build;
    hnsw.dist_func_param_ = const_cast<TurboQuantSpace *>(this);
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
};

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

static const uint32_t TQRV_MAGIC   = 0x54515256;  // "TQRV"
static const uint32_t TQRV_VERSION = 2;

enum RawVectorDtype : uint32_t {
    DTYPE_FLOAT32 = 0,
    DTYPE_FLOAT16 = 1,
};

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
    in.read(reinterpret_cast<char *>(&off_tmp), 8);  // data_offset field
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
    void *mapped_;          ///< mmap base (POSIX) or malloc'd block (Windows)
    size_t file_size_;      ///< total mapped/allocated size
    const char *data_;      ///< pointer to first vector byte
    uint64_t num_vectors_;
    uint64_t dim_;
    RawVectorDtype dtype_;
    size_t elem_size_;      ///< bytes per element: 4 (fp32) or 2 (fp16)
    size_t vec_bytes_;      ///< dim_ * elem_size_
    bool is_mmap_;          ///< true = mmap, false = malloc fallback

 public:
    MappedRawVectors()
        : mapped_(nullptr), file_size_(0), data_(nullptr),
          num_vectors_(0), dim_(0), dtype_(DTYPE_FLOAT32),
          elem_size_(4), vec_bytes_(0), is_mmap_(false) {}

    ~MappedRawVectors() { close(); }

    // Non-copyable, movable
    MappedRawVectors(const MappedRawVectors &) = delete;
    MappedRawVectors &operator=(const MappedRawVectors &) = delete;
    MappedRawVectors(MappedRawVectors &&o) noexcept { moveFrom(o); }
    MappedRawVectors &operator=(MappedRawVectors &&o) noexcept {
        if (this != &o) { close(); moveFrom(o); }
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
        if (fstat(fd, &st) != 0) { ::close(fd); return Status("MappedRawVectors: fstat failed"); }
        file_size_ = static_cast<size_t>(st.st_size);

        mapped_ = mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd, 0);
        ::close(fd);
        if (mapped_ == MAP_FAILED) { mapped_ = nullptr; return Status("MappedRawVectors: mmap failed"); }

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
        if (!in.good()) { std::free(mapped_); mapped_ = nullptr; return Status("MappedRawVectors: read failed"); }
        is_mmap_ = false;
#endif

        return parseHeader();
    }

    void close() {
        if (!mapped_) return;
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
        if (dtype_ != DTYPE_FLOAT32) return nullptr;
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
        if (file_size_ < 32) { close(); return Status("MappedRawVectors: file too small"); }

        const char *hdr = static_cast<const char *>(mapped_);
        uint32_t magic, version;
        std::memcpy(&magic, hdr, 4);
        std::memcpy(&version, hdr + 4, 4);

        if (magic != TQRV_MAGIC) { close(); return Status("MappedRawVectors: invalid magic"); }

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
        if (file_size_ < expected) { close(); return Status("MappedRawVectors: file truncated"); }

        return OkStatus();
    }

    void moveFrom(MappedRawVectors &o) {
        mapped_ = o.mapped_;       o.mapped_ = nullptr;
        file_size_ = o.file_size_; o.file_size_ = 0;
        data_ = o.data_;           o.data_ = nullptr;
        num_vectors_ = o.num_vectors_; o.num_vectors_ = 0;
        dim_ = o.dim_;             o.dim_ = 0;
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

public:
  TurboQuantIndex(size_t dim, int bits_per_coord = 8,
                  uint64_t rot_seed = 42, uint64_t qjl_seed = 137)
      : dim_(dim), bits_per_coord_(bits_per_coord),
        rot_seed_(rot_seed), qjl_seed_(qjl_seed) {}

  // -- Build ----------------------------------------------------------------

  /// Initialize space and HNSW graph for building. Call addPoint() to populate.
  void initBuild(size_t max_elements, size_t M = 16, size_t ef_construction = 200) {
    space_.reset(new TurboQuantSpace(dim_, bits_per_coord_, rot_seed_, qjl_seed_));
    hnsw_.reset(new HierarchicalNSW<float>(space_.get(), max_elements, M, ef_construction));
  }

  /// Encode a float vector and add it to the index. Thread-safe (addPoint uses mutexes).
  /// Caller must provide a thread-local buffer of size codeSizeBytes().
  void addPoint(const float *vec, labeltype label, char *encode_buf) {
    space_->encodeVector(vec, encode_buf);
    hnsw_->addPoint(encode_buf, label);
  }

  /// Build a TQ-compressed HNSW index from raw float vectors (single-threaded).
  /// data[i] points to a float[dim] vector for label i.
  Status build(const float *const *data, size_t n,
               size_t M = 16, size_t ef_construction = 200) {
    initBuild(n, M, ef_construction);

    std::vector<char> buf(space_->codeSizeBytes());
    for (size_t i = 0; i < n; ++i)
      addPoint(data[i], i, buf.data());

    return OkStatus();
  }

  /// Build from contiguous array: data points to n*dim floats, row-major.
  Status build(const float *data, size_t n,
               size_t M = 16, size_t ef_construction = 200) {
    std::vector<const float *> ptrs(n);
    for (size_t i = 0; i < n; ++i)
      ptrs[i] = data + i * dim_;
    return build(ptrs.data(), n, M, ef_construction);
  }

  // -- Save / Load ----------------------------------------------------------

  /// Save index and optionally raw vectors for re-ranking.
  /// raw_data points to the original float vectors (n * dim, row-major).
  /// If raw_path is empty, raw vectors are not saved.
  Status save(const std::string &index_path,
              const std::string &raw_path = "",
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
  Status load(const std::string &index_path,
              const std::string &raw_path = "") {
    space_.reset(new TurboQuantSpace(dim_, bits_per_coord_, rot_seed_, qjl_seed_));
    hnsw_.reset(new HierarchicalNSW<float>(space_.get(), index_path));
    space_->setSearchMode(*hnsw_);

    if (!raw_path.empty()) {
      auto st = raw_vectors_.open(raw_path);
      if (!st.ok()) return st;
    }

    return OkStatus();
  }

  // -- Search ---------------------------------------------------------------

  void setEf(size_t ef) {
    if (hnsw_) hnsw_->setEf(ef);
  }

  size_t getEf() const {
    return hnsw_ ? hnsw_->ef_ : 0;
  }

  /// TQ-only search. Thread-safe.
  /// Returns priority queue of (distance, label) pairs, worst first.
  std::priority_queue<std::pair<float, labeltype>>
  search(const float *query, size_t k) const {
    ensureSearchMode();
    auto pq = space_->prepareQuery(query);
    return hnsw_->searchKnn(&pq, k);
  }

  /// TQ search + exact L2 re-ranking from mmap'd raw vectors. Thread-safe.
  /// Retrieves rerank_ef candidates via TQ, re-ranks by exact L2, returns top-k.
  /// rerank_ef controls the number of TQ candidates to re-rank.
  /// Set hnsw ef >= rerank_ef before calling (via setEf).
  /// Default rerank_ef = 0 means use current ef (i.e. re-rank all candidates
  /// that searchKnn returns).
  std::vector<std::pair<float, labeltype>>
  searchRerank(const float *query, size_t k, size_t rerank_ef = 0) const {
    if (!raw_vectors_.is_open())
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

      const float *raw_vec = raw_vectors_.get_float32(id);
      if (raw_vec == nullptr) {
        raw_vectors_.get(id, vec_buf.data());
        raw_vec = vec_buf.data();
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

  bool hasRawVectors() const { return raw_vectors_.is_open(); }
  size_t dim() const { return dim_; }
  size_t numElements() const { return hnsw_ ? hnsw_->cur_element_count.load() : 0; }
  size_t codeSizeBytes() const { return space_ ? space_->codeSizeBytes() : 0; }

  const TurboQuantSpace *space() const { return space_.get(); }
  const HierarchicalNSW<float> *hnsw() const { return hnsw_.get(); }

private:
  void ensureSearchMode() const {
    // After build() the HNSW still has build dist func.
    // Lazily switch on first search. Safe: setSearchMode just writes
    // two pointers, and concurrent reads of the same value are fine.
    if (hnsw_->fstdistfunc_ != space_->getSearchDistFunc()) {
      const_cast<TurboQuantSpace *>(space_.get())->setSearchMode(
          *const_cast<HierarchicalNSW<float> *>(hnsw_.get()));
    }
  }
};

} // namespace turboquant
} // namespace hnswlib