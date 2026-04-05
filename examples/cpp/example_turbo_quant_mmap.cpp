/// example_turbo_quant_mmap.cpp — TurboQuant search with mmap-based re-ranking
///
/// Problem:
///   Compressed HNSW indices (TurboQuant) reduce in-memory data size by ~1.9x,
///   but quantization noise degrades recall.  Exact L2 re-ranking of a short
///   candidate list restores recall to near-baseline levels.  The challenge is
///   accessing original vectors for re-ranking without keeping them all in RAM.
///
/// Solution: MappedRawVectors
///   Original vectors are saved to a flat file (.tqrv) and memory-mapped at
///   search time.  The OS page cache transparently manages which pages are
///   resident — hot vectors stay in RAM, cold ones are paged in on demand.
///
///   Benefits:
///     - HNSW index holds only TQ codes (268 B/vec for dim=128)
///     - Raw vectors live on disk; resident set grows only with working set
///     - Zero-copy access for float32 (pointer directly into mmap region)
///     - Automatic fp16 -> fp32 conversion when raw file uses float16
///     - Read-only mapping — fully thread-safe, no synchronization needed
///
/// Best practices:
///   1. Save raw vectors BEFORE compressIndex() — after compression, the
///      original float data in HNSW slots is overwritten.
///   2. Use MappedRawVectors (mmap) instead of loadRawVectors (fseek) for
///      production workloads.  mmap amortizes syscall overhead and lets the
///      OS prefetch intelligently.
///   3. For datasets that fit in RAM: the OS will keep the entire file in
///      page cache after the first pass — subsequent re-ranking is as fast
///      as reading from an in-memory array.
///   4. For datasets larger than RAM: re-ranking incurs random I/O. Consider
///      using float16 storage (DTYPE_FLOAT16) to halve the file size and
///      the number of page faults.  The conversion cost is negligible on
///      modern hardware (ARM NEON / x86 F16C).
///   5. One MappedRawVectors instance can serve all search threads — mmap
///      read-only regions are thread-safe by definition.
///   6. Typical search pattern: TQ search with ef > K to get a broad
///      candidate set, then re-rank by exact L2 to select final top-K.
///      Good starting point: ef = 5..10x of K.
///
/// This example demonstrates:
///   (a) Building an HNSW index with L2, saving raw vectors, compressing
///   (b) Opening the raw file via MappedRawVectors
///   (c) TQ search + mmap-based L2 re-ranking
///   (d) Comparison: no rerank vs mmap rerank vs L2 baseline

#include "../../hnswlib/turbo_quant_space.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <random>
#include <unordered_set>
#include <vector>

using namespace hnswlib::turboquant;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static inline float computeL2(const float *__restrict__ a,
                              const float *__restrict__ b, size_t dim) {
  size_t i = 0;
  float dist = 0.0f;
  for (; i + 64 <= dim; i += 64) {
    for (size_t j = 0; j < 64; ++j) {
      dist += a[i + j] * b[i + j];
    }
  }
  if (i < dim) {
    for (size_t j = 0; i + j < dim; ++j) {
      dist += a[i + j] * b[i + j];
    }
  }
  return dist;
}

static float measureRecall(
    const std::vector<std::vector<hnswlib::labeltype>> &results,
    const std::vector<std::vector<hnswlib::labeltype>> &gt, size_t k) {
  size_t correct = 0, total = 0;
  for (size_t q = 0; q < results.size(); ++q) {
    std::unordered_set<hnswlib::labeltype> gt_set(
        gt[q].begin(), gt[q].begin() + std::min(k, gt[q].size()));
    total += gt_set.size();
    for (size_t i = 0; i < k && i < results[q].size(); ++i) {
      if (gt_set.count(results[q][i]))
        ++correct;
    }
  }
  return static_cast<float>(correct) / static_cast<float>(total);
}

int main() {
  const size_t dim = 128;
  const size_t N = 10000;
  const size_t num_queries = 200;
  const size_t K = 10;
  const size_t M = 16;
  const size_t ef_construction = 200;
  const size_t rerank_ef = 100;

  std::cout << "=== TurboQuant + MappedRawVectors Example ===" << std::endl;
  std::cout << "dim=" << dim
            << "\nN=" << N
            << "\nqueries=" << num_queries
            << "\nK=" << K
            << "\nM=" << M
            << "\nef_construction" << ef_construction
            << "\nrerank_ef" << rerank_ef << std::endl;

  // -----------------------------------------------------------------------
  // Generate random data
  // -----------------------------------------------------------------------
  std::mt19937 rng(42);
  std::normal_distribution<float> dist(0.0f, 1.0f);

  std::vector<float> data(N * dim);
  for (size_t i = 0; i < N * dim; ++i)
    data[i] = dist(rng);

  std::vector<float> queries(num_queries * dim);
  for (size_t i = 0; i < num_queries * dim; ++i)
    queries[i] = dist(rng);

  // -----------------------------------------------------------------------
  // Brute-force ground truth
  // -----------------------------------------------------------------------
  std::cout << "Computing ground truth..." << std::flush;
  std::vector<std::vector<hnswlib::labeltype>> gt(num_queries);
  {
    hnswlib::L2Space l2space(dim);
    hnswlib::BruteforceSearch<float> bf(&l2space, N);
    for (size_t i = 0; i < N; ++i)
      bf.addPoint(data.data() + i * dim, i);
    for (size_t q = 0; q < num_queries; ++q) {
      auto result = bf.searchKnn(queries.data() + q * dim, K);
      gt[q].resize(result.size());
      size_t idx = result.size();
      while (!result.empty()) {
        gt[q][--idx] = result.top().second;
        result.pop();
      }
    }
  }
  std::cout << " done" << std::endl;

  // -----------------------------------------------------------------------
  // Build L2 index and measure baseline
  // -----------------------------------------------------------------------
  std::cout << "Building L2 index..." << std::flush;
  hnswlib::L2Space l2space(dim);
  hnswlib::HierarchicalNSW<float> hnsw(&l2space, N, M, ef_construction);
  for (size_t i = 0; i < N; ++i)
    hnsw.addPoint(data.data() + i * dim, i);
  std::cout << " done" << std::endl;

  hnsw.setEf(64);
  std::vector<std::vector<hnswlib::labeltype>> l2_results(num_queries);
  for (size_t q = 0; q < num_queries; ++q) {
    auto result = hnsw.searchKnn(queries.data() + q * dim, K);
    l2_results[q].resize(result.size());
    size_t idx = result.size();
    while (!result.empty()) {
      l2_results[q][--idx] = result.top().second;
      result.pop();
    }
  }
  float l2_recall = measureRecall(l2_results, gt, K);

  // -----------------------------------------------------------------------
  // Save raw vectors to disk (BEFORE compression!)
  // -----------------------------------------------------------------------
  const std::string raw_path = "example_mmap_raw.tqrv";
  std::cout << "Saving raw vectors to " << raw_path << "..." << std::flush;
  auto status = saveRawVectors(raw_path, hnsw, dim);
  if (!status.ok()) {
    std::cerr << "Error: " << status.message() << std::endl;
    return 1;
  }
  std::cout << " done" << std::endl;

  // -----------------------------------------------------------------------
  // Compress index in-place (float32 -> TQ8)
  // -----------------------------------------------------------------------
  const int bits = 8;
  TurboQuantSpace tq_space(dim, bits);
  std::cout << "Compressing (" << dim * sizeof(float) << "B -> "
            << tq_space.codeSizeBytes() << "B per vector, " << std::fixed
            << std::setprecision(1)
            << static_cast<float>(dim * sizeof(float)) /
                   tq_space.codeSizeBytes()
            << "x)..." << std::flush;
  status = compressIndex(hnsw, tq_space);
  if (!status.ok()) {
    std::cerr << "Error: " << status.message() << std::endl;
    return 1;
  }
  std::cout << " done" << std::endl;

  // -----------------------------------------------------------------------
  // Open raw vectors via mmap
  // -----------------------------------------------------------------------
  MappedRawVectors raw_store;
  status = raw_store.open(raw_path);
  if (!status.ok()) {
    std::cerr << "Error: " << status.message() << std::endl;
    return 1;
  }
  std::cout << "Mapped " << raw_store.num_vectors() << " vectors from "
            << raw_path << " (dtype="
            << (raw_store.dtype() == DTYPE_FLOAT32 ? "float32" : "float16")
            << ")" << std::endl;

  // -----------------------------------------------------------------------
  // TQ search without re-ranking
  // -----------------------------------------------------------------------
  hnsw.setEf(64);
  tq_space.setSearchMode(hnsw);
  std::vector<std::vector<hnswlib::labeltype>> tq_results(num_queries);
  for (size_t q = 0; q < num_queries; ++q) {
    const float *query = queries.data() + q * dim;
    auto pq = tq_space.prepareQuery(query);
    auto result = hnsw.searchKnn(&pq, K);

    tq_results[q].resize(result.size());
    size_t idx = result.size();
    while (!result.empty()) {
      tq_results[q][--idx] = result.top().second;
      result.pop();
    }
  }
  float tq_recall = measureRecall(tq_results, gt, K);

  // -----------------------------------------------------------------------
  // TQ search + mmap-based L2 re-ranking
  //
  // Pattern:
  //   1. TQ search with ef=rerank_ef -> broad candidate set
  //   2. For each candidate: get raw vector via mmap (zero-copy for fp32)
  //   3. Compute exact L2 distance
  //   4. partial_sort to select top-K
  // -----------------------------------------------------------------------
  hnsw.setEf(rerank_ef);
  std::vector<std::vector<hnswlib::labeltype>> rerank_results(num_queries);
  std::vector<float> vec_buf(dim); // reusable buffer (needed for fp16 path)

  for (size_t q = 0; q < num_queries; ++q) {
    const float *query = queries.data() + q * dim;

    // TQ search
    auto pq = tq_space.prepareQuery(query);
    auto result = hnsw.searchKnn(&pq, rerank_ef);

    // Re-rank using mmap'd raw vectors
    std::vector<std::pair<float, hnswlib::labeltype>> shortlist;
    shortlist.reserve(result.size());
    while (!result.empty()) {
      hnswlib::labeltype id = result.top().second;

      // Zero-copy for float32; falls back to get() + buffer for float16
      const float *raw_vec = raw_store.get_float32(id);
      if (raw_vec == nullptr) {
        raw_store.get(id, vec_buf.data());
        raw_vec = vec_buf.data();
      }

      float d = computeL2(query, raw_vec, dim);
      shortlist.push_back({d, id});
      result.pop();
    }

    // Select top-K
    std::partial_sort(shortlist.begin(),
                      shortlist.begin() + std::min(K, shortlist.size()),
                      shortlist.end());

    rerank_results[q].resize(std::min(K, shortlist.size()));
    for (size_t i = 0; i < rerank_results[q].size(); ++i)
      rerank_results[q][i] = shortlist[i].second;
  }
  float rerank_recall = measureRecall(rerank_results, gt, K);

  // -----------------------------------------------------------------------
  // Summary
  // -----------------------------------------------------------------------
  std::cout << std::endl;
  std::cout << "=== Results (recall@" << K << ") ===" << std::endl;
  std::cout << std::fixed << std::setprecision(4);
  std::cout << "  L2 baseline (ef=64):            " << l2_recall << std::endl;
  std::cout << "  TQ" << bits << " compressed (ef=64):       " << tq_recall
            << std::endl;
  std::cout << "  TQ" << bits << " + mmap rerank (ef=" << rerank_ef
            << "):  " << rerank_recall << std::endl;
  std::cout << std::endl;
  std::cout << "  Index memory: " << tq_space.codeSizeBytes() << " B/vec (vs "
            << dim * sizeof(float) << " B/vec for L2)" << std::endl;
  std::cout << "  Raw file:     " << raw_store.num_vectors() * dim * 4
            << " bytes on disk, OS-managed page cache" << std::endl;

  // Cleanup
  raw_store.close();
  std::remove(raw_path.c_str());

  return 0;
}
