/// example_turbo_quant.cpp — TurboQuant compressed HNSW with re-ranking
///
/// Demonstrates the full workflow:
///   1. Build HNSW with exact L2 distances
///   2. Save raw vectors for re-ranking
///   3. Compress index in-place (float32 → TurboQuant codes, 3.3x compression)
///   4. Save compressed index
///   5. Search with asymmetric TQ distance
///   6. Re-rank shortlist using exact L2 from saved raw vectors
///   7. Load compressed index from disk and verify

#include "../../hnswlib/space_turbo_quant.h"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <unordered_set>
#include <vector>

using namespace hnswlib::turboquant;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static float computeL2(const float *a, const float *b, size_t dim) {
  float dist = 0.0f;
  for (size_t i = 0; i < dim; ++i) {
    float d = a[i] - b[i];
    dist += d * d;
  }
  return dist;
}

/// Measure recall@K: fraction of true top-K neighbors found.
static float
measureRecall(const std::vector<std::vector<hnswlib::labeltype>> &results,
              const std::vector<std::vector<hnswlib::labeltype>> &gt,
              size_t k) {
  size_t correct = 0;
  size_t total = 0;
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
  // -----------------------------------------------------------------------
  // Parameters
  // -----------------------------------------------------------------------
  const size_t dim = 128;
  const size_t N = 10000;
  const size_t num_queries = 200;
  const size_t K = 10;
  const size_t M = 16;
  const size_t ef_construction = 200;
  const size_t rerank_ef = 100; // shortlist size for re-ranking

  std::cout << "=== TurboQuant Example ===" << std::endl;
  std::cout << "N=" << N << ", dim=" << dim << ", M=" << M
            << ", ef_construction=" << ef_construction << ", K=" << K
            << std::endl
            << std::endl;

  // -----------------------------------------------------------------------
  // Step 1: Generate random data
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
  // Step 2: Compute brute-force ground truth
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
  // Step 3: Build HNSW with exact L2 distances and measure L2 baseline
  // -----------------------------------------------------------------------
  std::cout << "Building L2 index..." << std::flush;
  hnswlib::L2Space l2space(dim);
  hnswlib::HierarchicalNSW<float> hnsw_l2(&l2space, N, M, ef_construction);
  for (size_t i = 0; i < N; ++i)
    hnsw_l2.addPoint(data.data() + i * dim, i);
  std::cout << " done" << std::endl;

  // Baseline L2 recall
  hnsw_l2.setEf(64);
  std::vector<std::vector<hnswlib::labeltype>> l2_results(num_queries);
  for (size_t q = 0; q < num_queries; ++q) {
    auto result = hnsw_l2.searchKnn(queries.data() + q * dim, K);
    l2_results[q].resize(result.size());
    size_t idx = result.size();
    while (!result.empty()) {
      l2_results[q][--idx] = result.top().second;
      result.pop();
    }
  }
  float l2_recall = measureRecall(l2_results, gt, K);

  // -----------------------------------------------------------------------
  // Steps 4-9: Run for each bit width (4-bit and 8-bit TurboQuant)
  // -----------------------------------------------------------------------
  struct TQResult {
    int bits;
    size_t code_size;
    float tq_recall;
    float rerank_recall;
    float loaded_recall;
  };
  std::vector<TQResult> tq_results_all;

  for (int bits_per_coord : {4, 8}) {
    std::cout << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "  TurboQuant " << bits_per_coord << " bits/coord ("
              << (bits_per_coord - 1) << "-bit SQ + 1-bit QJL)" << std::endl;
    std::cout << "============================================" << std::endl;

    // Rebuild HNSW from scratch (compress mutates in-place)
    hnswlib::HierarchicalNSW<float> hnsw(&l2space, N, M, ef_construction);
    for (size_t i = 0; i < N; ++i)
      hnsw.addPoint(data.data() + i * dim, i);

    // Save raw vectors for re-ranking
    std::string raw_path =
        "example_tq_raw_" + std::to_string(bits_per_coord) + ".tqrv";
    std::cout << "Saving raw vectors..." << std::flush;
    auto status = saveRawVectors(raw_path, hnsw, dim);
    if (!status.ok()) {
      std::cerr << "Error: " << status.message() << std::endl;
      return 1;
    }
    std::cout << " done" << std::endl;

    // Compress index in-place
    TurboQuantSpace tq_space(dim, bits_per_coord);
    std::cout << "Compressing (float32 " << dim * sizeof(float) << "B -> TQ "
              << tq_space.codeSizeBytes() << "B, " << std::fixed
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

    // Save compressed index
    std::string index_path =
        "example_tq_index_" + std::to_string(bits_per_coord) + ".bin";
    std::cout << "Saving compressed index..." << std::flush;
    hnsw.saveIndex(index_path);
    std::cout << " done" << std::endl;

    // TQ search (without re-ranking)
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

    // TQ search + L2 re-ranking
    hnsw.setEf(rerank_ef);
    std::vector<std::vector<hnswlib::labeltype>> rerank_results(num_queries);
    for (size_t q = 0; q < num_queries; ++q) {
      const float *query = queries.data() + q * dim;

      auto pq = tq_space.prepareQuery(query);
      auto result = hnsw.searchKnn(&pq, rerank_ef);

      std::vector<size_t> candidate_ids;
      candidate_ids.reserve(result.size());
      while (!result.empty()) {
        candidate_ids.push_back(result.top().second);
        result.pop();
      }

      std::vector<float> raw_vecs(candidate_ids.size() * dim);
      status = loadRawVectors(raw_path, candidate_ids.data(),
                              candidate_ids.size(), dim, raw_vecs.data());
      if (!status.ok()) {
        std::cerr << "Error: " << status.message() << std::endl;
        return 1;
      }

      std::vector<std::pair<float, hnswlib::labeltype>> shortlist;
      shortlist.reserve(candidate_ids.size());
      for (size_t i = 0; i < candidate_ids.size(); ++i) {
        float d = computeL2(query, raw_vecs.data() + i * dim, dim);
        shortlist.push_back(
            {d, static_cast<hnswlib::labeltype>(candidate_ids[i])});
      }
      std::partial_sort(shortlist.begin(),
                        shortlist.begin() + std::min(K, shortlist.size()),
                        shortlist.end());

      rerank_results[q].resize(std::min(K, shortlist.size()));
      for (size_t i = 0; i < rerank_results[q].size(); ++i)
        rerank_results[q][i] = shortlist[i].second;
    }
    float rerank_recall = measureRecall(rerank_results, gt, K);

    // Load compressed index from disk and verify
    std::cout << "Loading compressed index from disk..." << std::flush;
    TurboQuantSpace tq_space2(dim, bits_per_coord);
    hnswlib::HierarchicalNSW<float> hnsw2(&tq_space2, index_path);
    hnsw2.setEf(64);
    tq_space2.setSearchMode(hnsw2);
    std::cout << " done" << std::endl;

    std::vector<std::vector<hnswlib::labeltype>> loaded_results(num_queries);
    for (size_t q = 0; q < num_queries; ++q) {
      const float *query = queries.data() + q * dim;
      auto pq = tq_space2.prepareQuery(query);
      auto result = hnsw2.searchKnn(&pq, K);

      loaded_results[q].resize(result.size());
      size_t idx = result.size();
      while (!result.empty()) {
        loaded_results[q][--idx] = result.top().second;
        result.pop();
      }
    }
    float loaded_recall = measureRecall(loaded_results, gt, K);

    tq_results_all.push_back({bits_per_coord, tq_space.codeSizeBytes(),
                              tq_recall, rerank_recall, loaded_recall});

    // Cleanup temp files
    std::remove(raw_path.c_str());
    std::remove(index_path.c_str());
  }

  // -----------------------------------------------------------------------
  // Summary
  // -----------------------------------------------------------------------
  std::cout << std::endl;
  std::cout << "=== Results ===" << std::endl;
  std::cout << "  L2 baseline (ef=64):      recall@" << K << " = " << std::fixed
            << std::setprecision(4) << l2_recall << std::endl;

  for (const auto &r : tq_results_all) {
    std::cout << std::endl;
    std::cout << "  TQ " << r.bits << "-bit (ef=64):        recall@" << K
              << " = " << r.tq_recall << std::endl;
    std::cout << "  TQ " << r.bits << "-bit + rerank (ef=" << rerank_ef
              << "): recall@" << K << " = " << r.rerank_recall << std::endl;
    std::cout << "  TQ " << r.bits << "-bit loaded (ef=64): recall@" << K
              << " = " << r.loaded_recall << std::endl;
    std::cout << "  Compression: " << dim * sizeof(float) << " -> "
              << r.code_size << " bytes/vector (" << std::setprecision(1)
              << static_cast<float>(dim * sizeof(float)) / r.code_size << "x)"
              << std::endl;
  }

  return 0;
}
