#pragma once
/// turbo_quant_space.h — TurboQuant SpaceInterface adapter + compress/save utilities
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

#include <fstream>
#include <cstring>
#include <algorithm>

namespace hnswlib {
namespace turboquant {

// ===========================================================================
// TurboQuantDistParam — shared state for distance function dispatch
// ===========================================================================

struct TurboQuantDistParam {
    size_t dim;
    const float* centroids;
    const TurboQuantPreparedQuery* prepared_query;  ///< non-null = search mode
};

// ===========================================================================
// TurboQuantSpace — SpaceInterface adapter for HNSW
//
// Distance modes:
//   - Asymmetric (search): float query × TQ code. Set via beginSearch().
//   - Symmetric (build):   TQ code × TQ code. Used when prepared_query is null.
//
// Thread safety: beginSearch/endSearch modify shared state.
//   For concurrent search, use one TurboQuantSpace per thread.
// ===========================================================================

class TurboQuantSpace : public SpaceInterface<float> {
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

    /// Asymmetric distance: prepared float query × compressed code.
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
        for (size_t i = 0; i < dim; ++i)
            ip_mse += pq->q_rot[i] * p->centroids[sq_packed[i]] * sigma;

        float dot_qjl = 0.0f;
        for (size_t i = 0; i < dim; ++i) {
            const bool positive = (qjl_signs[i / 64] >> (i % 64)) & 1ULL;
            dot_qjl += pq->s_q[i] * (positive ? 1.0f : -1.0f);
        }
        const float scale = std::sqrt(static_cast<float>(M_PI) / 2.0f)
                          / std::sqrt(static_cast<float>(dim));
        const float correction = scale * gamma * dot_qjl;

        const float ip = (ip_mse + correction) * x_norm * pq->q_norm;
        return std::max(0.0f, pq->q_norm_sq + x_norm * x_norm - 2.0f * ip);
    }

    /// Symmetric distance: both sides are compressed codes.
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
        for (size_t i = 0; i < dim; ++i)
            ip_rot += (p->centroids[sq_a[i]] * sigma_a)
                    * (p->centroids[sq_b[i]] * sigma_b);

        const float ip = ip_rot * norm_a * norm_b;
        return std::max(0.0f, norm_a * norm_a + norm_b * norm_b - 2.0f * ip);
    }

    /// Dispatch: asymmetric if prepared_query is set, else symmetric.
    static float turboQuantL2(const void* pVect1, const void* pVect2,
                              const void* param_ptr) {
        const auto* p = static_cast<const TurboQuantDistParam*>(param_ptr);
        const char* buf2 = static_cast<const char*>(pVect2);
        if (p->prepared_query != nullptr)
            return distSearch(p->prepared_query, buf2, p);
        else
            return distSymmetric(static_cast<const char*>(pVect1), buf2, p);
    }

    /// Encode a float vector to TQ code bytes.
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
            for (int b = 0; b < num_boundaries_; ++b)
                idx += (normalized[k] > boundaries_[b]);
            sq_out[k] = idx;
        }

        std::vector<float> residual(d);
        for (size_t i = 0; i < d; ++i)
            residual[i] = x_rot[i] - centroids_[sq_out[i]] * sigma;
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
            if (projected[i] >= 0.0f)
                qjl_out[i / 64] |= (1ULL << (i % 64));
        }

        size_t qjl_b = num_words * sizeof(uint64_t);
        float* meta = reinterpret_cast<float*>(out_buf + d + qjl_b);
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
        assert(dim >= 4 && "TurboQuantSpace: dim must be at least 4");
        assert((dim & (dim - 1)) == 0
               && "TurboQuantSpace: dim must be a power of 2");

        if (mse_bits_ == 3) {
            boundaries_ = detail::LM3.boundaries.data();
            num_boundaries_ = 7;
            centroids_ = detail::LM3.centroids.data();
        } else if (mse_bits_ == 4) {
            boundaries_ = detail::LM4.boundaries.data();
            num_boundaries_ = 15;
            centroids_ = detail::LM4.centroids.data();
        } else {
            assert(false && "TurboQuantSpace: only 4-bit and 5-bit modes");
        }
        rotation_signs_ = generateSigns(dim_, rot_seed);
        qjl_signs_ = generateSigns(dim_, qjl_seed);

        dist_param_.dim = dim_;
        dist_param_.centroids = centroids_;
        dist_param_.prepared_query = nullptr;
    }

    // -- SpaceInterface -------------------------------------------------------

    size_t get_data_size() override { return code_size_; }
    DISTFUNC<float> get_dist_func() override { return &turboQuantL2; }
    void* get_dist_func_param() override { return &dist_param_; }

    // -- Accessors ------------------------------------------------------------

    size_t dim() const { return dim_; }
    size_t codeSizeBytes() const { return code_size_; }

    // -- Encoding -------------------------------------------------------------

    /// Encode a float[dim] vector to a TQ code of codeSizeBytes() bytes.
    void encodeVector(const float* raw, char* out_buf) const {
        encodeVectorImpl(raw, out_buf);
    }

    // -- Search state management ----------------------------------------------

    /// Prepare query: amortizes 2 RHT calls across all distance computations.
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

    /// Set distance function to asymmetric mode for the given prepared query.
    /// Must be paired with endSearch(). Not thread-safe.
    void beginSearch(const TurboQuantPreparedQuery& pq) {
        dist_param_.prepared_query = &pq;
    }

    /// Reset distance function to symmetric mode.
    void endSearch() {
        dist_param_.prepared_query = nullptr;
    }
};


// ===========================================================================
// Compress/Save/Load utilities
// ===========================================================================

/// Raw vectors file magic and header size.
static const uint32_t TQRV_MAGIC = 0x54515256;  // "TQRV"
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
inline Status saveRawVectors(
    const std::string& path,
    const HierarchicalNSW<float>& hnsw,
    size_t dim) {

    std::ofstream out(path, std::ios::binary);
    if (!out.good())
        return Status("saveRawVectors: cannot open output file");

    size_t n = hnsw.cur_element_count.load();

    // Header (32 bytes)
    out.write(reinterpret_cast<const char*>(&TQRV_MAGIC), 4);
    out.write(reinterpret_cast<const char*>(&TQRV_VERSION), 4);
    uint64_t n64 = static_cast<uint64_t>(n);
    uint64_t dim64 = static_cast<uint64_t>(dim);
    out.write(reinterpret_cast<const char*>(&n64), 8);
    out.write(reinterpret_cast<const char*>(&dim64), 8);
    out.write(reinterpret_cast<const char*>(&TQRV_DATA_OFFSET), 8);

    // Vectors (N * dim * 4 bytes)
    for (size_t i = 0; i < n; ++i) {
        const char* data = hnsw.getDataByInternalId(static_cast<tableint>(i));
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
inline Status compressIndex(
    HierarchicalNSW<float>& hnsw,
    TurboQuantSpace& tq_space) {

    size_t n = hnsw.cur_element_count.load();
    size_t dim = tq_space.dim();
    size_t code_size = tq_space.codeSizeBytes();
    size_t data_size = hnsw.data_size_;

    if (code_size > data_size)
        return Status("compressIndex: TQ code size exceeds L2 data slot size");

    std::vector<float> float_buf(dim);
    std::vector<char> code_buf(code_size);

    for (size_t i = 0; i < n; ++i) {
        char* slot = hnsw.getDataByInternalId(static_cast<tableint>(i));

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
inline Status loadRawVectors(
    const std::string& path,
    const size_t* ids,
    size_t num_ids,
    size_t dim,
    float* out) {

    std::ifstream in(path, std::ios::binary);
    if (!in.good())
        return Status("loadRawVectors: cannot open file");

    // Read and validate header
    uint32_t magic, version;
    uint64_t n64, dim64, data_offset;
    in.read(reinterpret_cast<char*>(&magic), 4);
    in.read(reinterpret_cast<char*>(&version), 4);
    in.read(reinterpret_cast<char*>(&n64), 8);
    in.read(reinterpret_cast<char*>(&dim64), 8);
    in.read(reinterpret_cast<char*>(&data_offset), 8);

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

        std::streamoff offset = static_cast<std::streamoff>(
            data_offset + ids[i] * vec_bytes);
        in.seekg(offset, std::ios::beg);
        in.read(reinterpret_cast<char*>(out + i * dim), vec_bytes);

        if (!in.good())
            return Status("loadRawVectors: read error");
    }

    return OkStatus();
}

}  // namespace turboquant
}  // namespace hnswlib
