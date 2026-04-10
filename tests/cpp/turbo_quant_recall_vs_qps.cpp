/// turbo_quant_recall_vs_qps.cpp — Multi-dataset recall vs QPS benchmark.
///
/// Compares L2Space (float32) vs TurboQuantSpace across 3 datasets.
/// Outputs CSV to stdout, progress to stderr.
///
/// Usage: turbo_quant_recall_vs_qps [base_dir]
///   base_dir defaults to "." and should contain bigann/ and glove/ subdirs.

#include "hnswlib/space_turbo_quant.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <queue>
#include <atomic>
#include <exception>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#ifdef __APPLE__
#include <mach/mach.h>
#endif

// Multithreaded executor (copied from examples/cpp/example_mt_search.cpp).
template <class Function>
inline void ParallelFor(size_t start, size_t end, size_t numThreads,
                        Function fn) {
  if (numThreads <= 0)
    numThreads = std::thread::hardware_concurrency();
  if (numThreads == 1) {
    for (size_t id = start; id < end; id++)
      fn(id, 0);
    return;
  }
  std::vector<std::thread> threads;
  std::atomic<size_t> current(start);
  std::exception_ptr lastException = nullptr;
  std::mutex lastExceptMutex;
  for (size_t threadId = 0; threadId < numThreads; ++threadId) {
    threads.push_back(std::thread([&, threadId] {
      while (true) {
        size_t id = current.fetch_add(1);
        if (id >= end)
          break;
        try {
          fn(id, threadId);
        } catch (...) {
          std::unique_lock<std::mutex> lk(lastExceptMutex);
          lastException = std::current_exception();
          current = end;
          break;
        }
      }
    }));
  }
  for (auto &t : threads)
    t.join();
  if (lastException)
    std::rethrow_exception(lastException);
}

// Global thread counts (set from CLI in main).
static size_t g_build_threads = 0; // 0 => hardware_concurrency
static size_t g_search_threads = 1;

using namespace hnswlib::turboquant;

// ---------------------------------------------------------------------------
// Helpers (from turbo_quant_bigann_test.cpp)
// ---------------------------------------------------------------------------

class StopWatch {
  std::chrono::high_resolution_clock::time_point t0_;

public:
  StopWatch() : t0_(std::chrono::high_resolution_clock::now()) {}
  void reset() { t0_ = std::chrono::high_resolution_clock::now(); }
  double elapsedSeconds() const {
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(t1 - t0_).count();
  }
};

static size_t getCurrentRSS() {
#ifdef __APPLE__
  struct mach_task_basic_info info;
  mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
  if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info,
                &count) == KERN_SUCCESS)
    return info.resident_size;
#elif defined(__linux__)
  long rss = 0;
  FILE *fp = fopen("/proc/self/statm", "r");
  if (fp) {
    if (fscanf(fp, "%*s%ld", &rss) == 1) {
      fclose(fp);
      return static_cast<size_t>(rss) * sysconf(_SC_PAGESIZE);
    }
    fclose(fp);
  }
#endif
  return 0;
}

// ---------------------------------------------------------------------------
// Data loading (bvecs / fvecs / ivecs)
// ---------------------------------------------------------------------------

static bool loadBvecs(const std::string &path,
                      std::vector<std::vector<float>> &out,
                      size_t max_elements = 0) {
  std::ifstream in(path, std::ios::binary);
  if (!in.good())
    return false;
  out.clear();
  while (in.good() && !in.eof()) {
    int dim = 0;
    in.read(reinterpret_cast<char *>(&dim), sizeof(int));
    if (!in.good())
      break;
    std::vector<uint8_t> buf(dim);
    in.read(reinterpret_cast<char *>(buf.data()), dim);
    if (!in.good())
      break;
    std::vector<float> vec(dim);
    for (int i = 0; i < dim; ++i)
      vec[i] = static_cast<float>(buf[i]);
    out.push_back(std::move(vec));
    if (max_elements > 0 && out.size() >= max_elements)
      break;
  }
  return !out.empty();
}

static bool loadFvecs(const std::string &path,
                      std::vector<std::vector<float>> &out,
                      size_t max_elements = 0) {
  std::ifstream in(path, std::ios::binary);
  if (!in.good())
    return false;
  out.clear();
  while (in.good() && !in.eof()) {
    int dim = 0;
    in.read(reinterpret_cast<char *>(&dim), sizeof(int));
    if (!in.good())
      break;
    std::vector<float> vec(dim);
    in.read(reinterpret_cast<char *>(vec.data()), dim * sizeof(float));
    if (!in.good())
      break;
    out.push_back(std::move(vec));
    if (max_elements > 0 && out.size() >= max_elements)
      break;
  }
  return !out.empty();
}

static bool loadIvecs(const std::string &path,
                      std::vector<std::vector<int>> &out,
                      size_t max_elements = 0) {
  std::ifstream in(path, std::ios::binary);
  if (!in.good())
    return false;
  out.clear();
  while (in.good() && !in.eof()) {
    int dim = 0;
    in.read(reinterpret_cast<char *>(&dim), sizeof(int));
    if (!in.good())
      break;
    std::vector<int> vec(dim);
    in.read(reinterpret_cast<char *>(vec.data()), dim * sizeof(int));
    if (!in.good())
      break;
    out.push_back(std::move(vec));
    if (max_elements > 0 && out.size() >= max_elements)
      break;
  }
  return !out.empty();
}

// ---------------------------------------------------------------------------
// Ground truth
// ---------------------------------------------------------------------------

static std::vector<std::vector<int>>
computeBruteForceGT(const std::string &tag,
                    const std::vector<std::vector<float>> &base,
                    const std::vector<std::vector<float>> &queries, size_t k) {
  const size_t N = base.size();
  const size_t Q = queries.size();
  const size_t D = base[0].size();

  std::cerr << "[" << tag << "] Computing brute-force GT (" << Q
            << " queries x " << N << " base)..." << std::flush;
  StopWatch sw;

  hnswlib::L2Space l2space(D);
  hnswlib::BruteforceSearch<float> bf(&l2space, N);
  for (size_t i = 0; i < N; ++i)
    bf.addPoint(base[i].data(), i);

  std::vector<std::vector<int>> gt(Q);
  for (size_t q = 0; q < Q; ++q) {
    auto result = bf.searchKnn(queries[q].data(), k);
    gt[q].resize(result.size());
    size_t idx = result.size();
    while (!result.empty()) {
      gt[q][--idx] = static_cast<int>(result.top().second);
      result.pop();
    }
  }

  std::cerr << " done (" << std::fixed << std::setprecision(1)
            << sw.elapsedSeconds() << " s)" << std::endl;
  return gt;
}

// ---------------------------------------------------------------------------
// Recall measurement
// ---------------------------------------------------------------------------

static float measureRecall(hnswlib::HierarchicalNSW<float> &hnsw,
                           const std::vector<std::vector<float>> &queries,
                           const std::vector<std::vector<int>> &gt, size_t k) {
  std::atomic<size_t> correct(0);
  std::atomic<size_t> total(0);
  ParallelFor(0, queries.size(), g_search_threads, [&](size_t q, size_t) {
    auto result = hnsw.searchKnn(queries[q].data(), k);
    std::unordered_set<hnswlib::labeltype> gt_set;
    for (size_t j = 0; j < k && j < gt[q].size(); ++j)
      gt_set.insert(static_cast<hnswlib::labeltype>(gt[q][j]));
    total.fetch_add(gt_set.size(), std::memory_order_relaxed);
    size_t local_correct = 0;
    while (!result.empty()) {
      if (gt_set.count(result.top().second))
        ++local_correct;
      result.pop();
    }
    correct.fetch_add(local_correct, std::memory_order_relaxed);
  });
  return static_cast<float>(correct.load()) / static_cast<float>(total.load());
}

static float measureRecallTQ(hnswlib::HierarchicalNSW<float> &hnsw,
                             TurboQuantSpace &space,
                             const std::vector<std::vector<float>> &queries,
                             const std::vector<std::vector<int>> &gt,
                             size_t k) {
  std::atomic<size_t> correct(0);
  std::atomic<size_t> total(0);
  ParallelFor(0, queries.size(), g_search_threads, [&](size_t q, size_t) {
    auto pq = space.prepareQuery(queries[q].data());
    auto result = hnsw.searchKnn(&pq, k);
    std::unordered_set<hnswlib::labeltype> gt_set;
    for (size_t j = 0; j < k && j < gt[q].size(); ++j)
      gt_set.insert(static_cast<hnswlib::labeltype>(gt[q][j]));
    total.fetch_add(gt_set.size(), std::memory_order_relaxed);
    size_t local_correct = 0;
    while (!result.empty()) {
      if (gt_set.count(result.top().second))
        ++local_correct;
      result.pop();
    }
    correct.fetch_add(local_correct, std::memory_order_relaxed);
  });
  return static_cast<float>(correct.load()) / static_cast<float>(total.load());
}

// ---------------------------------------------------------------------------
// CSV output helper
// ---------------------------------------------------------------------------

static void emitCSV(const std::string &dataset, const std::string &method,
                    size_t ef, float recall, double us_per_query,
                    double build_time_s, size_t memory_bytes_per_vec,
                    bool include_build_info) {
  std::cout << dataset << "," << method << "," << ef << "," << std::fixed
            << std::setprecision(4) << recall << "," << std::setprecision(1)
            << us_per_query;
  if (include_build_info) {
    std::cout << "," << std::setprecision(2) << build_time_s << ","
              << memory_bytes_per_vec;
  } else {
    std::cout << ",,";
  }
  std::cout << "," << g_search_threads << "," << g_build_threads << "\n";
}

// ---------------------------------------------------------------------------
// Benchmark: L2 baseline (float32 graph + float32 search)
// ---------------------------------------------------------------------------

static void benchmarkL2(const std::string &tag,
                        const std::vector<std::vector<float>> &base_data,
                        const std::vector<std::vector<float>> &queries,
                        const std::vector<std::vector<int>> &gt, size_t dim,
                        size_t M, size_t ef_construction, size_t K,
                        const std::vector<size_t> &ef_values) {

  const size_t N = base_data.size();
  std::cerr << "[" << tag << "] L2: building index (" << N << " vectors)..."
            << std::endl;

  hnswlib::L2Space l2space(dim);
  size_t rss_before = getCurrentRSS();
  StopWatch sw;

  hnswlib::HierarchicalNSW<float> hnsw(&l2space, N, M, ef_construction);
  hnsw.addPoint(base_data[0].data(), 0);
  ParallelFor(1, N, g_build_threads, [&](size_t i, size_t) {
    hnsw.addPoint(base_data[i].data(), i);
  });

  double build_time = sw.elapsedSeconds();
  size_t rss_after = getCurrentRSS();
  size_t mem_per_vec =
      (rss_after > rss_before) ? (rss_after - rss_before) / N : 0;

  std::cerr << "[" << tag << "] L2 build: " << std::fixed
            << std::setprecision(2) << build_time << " s" << std::endl;

  // Warmup
  hnsw.setEf(10);
  measureRecall(hnsw, queries, gt, K);
  std::cerr << "[" << tag << "] L2 warmup done" << std::endl;

  // Sweep
  bool first = true;
  for (size_t ef : ef_values) {
    hnsw.setEf(ef);
    sw.reset();
    float recall = measureRecall(hnsw, queries, gt, K);
    double us_per_q = sw.elapsedSeconds() / queries.size() * 1e6;
    emitCSV(tag, "l2", ef, recall, us_per_q, build_time, mem_per_vec, first);
    first = false;
    std::cerr << "[" << tag << "]   ef=" << ef << ": recall=" << std::fixed
              << std::setprecision(4) << recall << ", " << std::setprecision(1)
              << us_per_q << " us/q" << std::endl;
  }
}

// ---------------------------------------------------------------------------
// Benchmark: TQ graph + TQ search (fully compressed)
// ---------------------------------------------------------------------------

static void benchmarkTQ(const std::string &tag,
                        const std::vector<std::vector<float>> &base_data,
                        const std::vector<std::vector<float>> &queries,
                        const std::vector<std::vector<int>> &gt, size_t dim,
                        size_t M, size_t ef_construction, size_t K,
                        const std::vector<size_t> &ef_values,
                        int bits_per_coord = 4) {

  const size_t N = base_data.size();
  TurboQuantSpace tqspace(dim, bits_per_coord, 42, 137);
  size_t code_size = tqspace.codeSizeBytes();

  std::string method = "tq" + std::to_string(bits_per_coord);
  std::cerr << "[" << tag << "] " << method << ": encoding " << N
            << " vectors..." << std::endl;
  StopWatch sw;

  // Parallel encode (encodeVector is stateless)
  std::vector<std::vector<char>> codes(N, std::vector<char>(code_size));
  ParallelFor(0, N, g_build_threads, [&](size_t i, size_t) {
    tqspace.encodeVector(base_data[i].data(), codes[i].data());
  });

  double encode_time = sw.elapsedSeconds();
  std::cerr << "[" << tag << "] " << method << " encode: " << std::fixed
            << std::setprecision(2) << encode_time << " s" << std::endl;

  // Parallel insert (symmetric distance is thread-safe when
  // prepared_query==nullptr)
  std::cerr << "[" << tag << "] " << method << ": inserting into HNSW..."
            << std::endl;
  size_t rss_before = getCurrentRSS();
  sw.reset();

  hnswlib::HierarchicalNSW<float> hnsw(&tqspace, N, M, ef_construction);
  hnsw.addPoint(codes[0].data(), 0);
  ParallelFor(1, N, g_build_threads, [&](size_t i, size_t) {
    hnsw.addPoint(codes[i].data(), i);
  });

  double insert_time = sw.elapsedSeconds();
  double build_time = encode_time + insert_time;
  size_t rss_after = getCurrentRSS();
  size_t mem_per_vec =
      (rss_after > rss_before) ? (rss_after - rss_before) / N : 0;

  std::cerr << "[" << tag << "] " << method << " insert: " << std::fixed
            << std::setprecision(2) << insert_time
            << " s (total build: " << build_time << " s)" << std::endl;

  // Switch distance function from symmetric (build) to asymmetric (search).
  tqspace.setSearchMode(hnsw);

  // Warmup
  hnsw.setEf(10);
  measureRecallTQ(hnsw, tqspace, queries, gt, K);
  std::cerr << "[" << tag << "] " << method << " warmup done" << std::endl;

  // Sweep
  bool first = true;
  for (size_t ef : ef_values) {
    hnsw.setEf(ef);
    sw.reset();
    float recall = measureRecallTQ(hnsw, tqspace, queries, gt, K);
    double us_per_q = sw.elapsedSeconds() / queries.size() * 1e6;
    emitCSV(tag, method, ef, recall, us_per_q, build_time, mem_per_vec, first);
    first = false;
    std::cerr << "[" << tag << "]   ef=" << ef << ": recall=" << std::fixed
              << std::setprecision(4) << recall << ", " << std::setprecision(1)
              << us_per_q << " us/q" << std::endl;
  }
}

// ---------------------------------------------------------------------------
// Benchmark: L2 graph build + replace data with TQ codes + TQ search
// ---------------------------------------------------------------------------

static void benchmarkL2GraphTQ(const std::string &tag,
                               const std::vector<std::vector<float>> &base_data,
                               const std::vector<std::vector<float>> &queries,
                               const std::vector<std::vector<int>> &gt,
                               size_t dim, size_t M, size_t ef_construction,
                               size_t K, const std::vector<size_t> &ef_values,
                               int bits_per_coord = 4) {

  const size_t N = base_data.size();
  TurboQuantSpace tqspace(dim, bits_per_coord, 42, 137);
  size_t tq_code_size = tqspace.codeSizeBytes();
  size_t l2_data_size = dim * sizeof(float);

  std::string method = "l2graph_tq" + std::to_string(bits_per_coord);

  if (tq_code_size > l2_data_size) {
    std::cerr << "[" << tag << "] " << method << ": SKIP (TQ code "
              << tq_code_size << " B > L2 slot " << l2_data_size << " B)"
              << std::endl;
    return;
  }

  std::cerr << "[" << tag << "] " << method << ": building L2 graph (" << N
            << " vectors)..." << std::endl;

  hnswlib::L2Space l2space(dim);
  size_t rss_before = getCurrentRSS();
  StopWatch sw;

  hnswlib::HierarchicalNSW<float> hnsw(&l2space, N, M, ef_construction);
  hnsw.addPoint(base_data[0].data(), 0);
  ParallelFor(1, N, g_build_threads, [&](size_t i, size_t) {
    hnsw.addPoint(base_data[i].data(), i);
  });

  double l2_build_time = sw.elapsedSeconds();
  std::cerr << "[" << tag << "] " << method << ": L2 build: " << std::fixed
            << std::setprecision(2) << l2_build_time << " s" << std::endl;

  // Replace stored data with TQ codes
  std::cerr << "[" << tag << "] " << method
            << ": replacing data with TQ codes..." << std::endl;
  sw.reset();
  std::vector<char> code_buf(tq_code_size);
  for (size_t i = 0; i < N; ++i) {
    tqspace.encodeVector(base_data[i].data(), code_buf.data());
    char *slot = hnsw.getDataByInternalId(static_cast<hnswlib::tableint>(i));
    memset(slot, 0, l2_data_size);
    memcpy(slot, code_buf.data(), tq_code_size);
  }
  double encode_time = sw.elapsedSeconds();
  double build_time = l2_build_time + encode_time;

  std::cerr << "[" << tag << "] " << method
            << ": encode+replace: " << std::fixed << std::setprecision(2)
            << encode_time << " s (total: " << build_time << " s)" << std::endl;

  // Swap distance function to TQ asymmetric search (PreparedQuery × code).
  tqspace.setSearchMode(hnsw);

  size_t rss_after = getCurrentRSS();
  size_t mem_per_vec =
      (rss_after > rss_before) ? (rss_after - rss_before) / N : 0;

  // Warmup
  hnsw.setEf(10);
  measureRecallTQ(hnsw, tqspace, queries, gt, K);
  std::cerr << "[" << tag << "] " << method << " warmup done" << std::endl;

  // Sweep
  bool first = true;
  for (size_t ef : ef_values) {
    hnsw.setEf(ef);
    sw.reset();
    float recall = measureRecallTQ(hnsw, tqspace, queries, gt, K);
    double us_per_q = sw.elapsedSeconds() / queries.size() * 1e6;
    emitCSV(tag, method, ef, recall, us_per_q, build_time, mem_per_vec, first);
    first = false;
    std::cerr << "[" << tag << "]   ef=" << ef << ": recall=" << std::fixed
              << std::setprecision(4) << recall << ", " << std::setprecision(1)
              << us_per_q << " us/q" << std::endl;
  }
}

// ---------------------------------------------------------------------------
// Benchmark: TQ search + L2 re-ranking
// ---------------------------------------------------------------------------

static void benchmarkTQRerank(const std::string &tag,
                              const std::vector<std::vector<float>> &base_data,
                              const std::vector<std::vector<float>> &queries,
                              const std::vector<std::vector<int>> &gt,
                              size_t dim, size_t M, size_t ef_construction,
                              size_t K, const std::vector<size_t> &ef_values,
                              int bits_per_coord = 4) {

  const size_t N = base_data.size();
  TurboQuantSpace tqspace(dim, bits_per_coord, 42, 137);
  size_t code_size = tqspace.codeSizeBytes();

  std::string method = "tq" + std::to_string(bits_per_coord) + "_rerank";
  std::cerr << "[" << tag << "] " << method << ": encoding " << N
            << " vectors..." << std::endl;
  StopWatch sw;

  // Parallel encode
  std::vector<std::vector<char>> codes(N, std::vector<char>(code_size));
  ParallelFor(0, N, g_build_threads, [&](size_t i, size_t) {
    tqspace.encodeVector(base_data[i].data(), codes[i].data());
  });

  double encode_time = sw.elapsedSeconds();

  // Parallel insert
  std::cerr << "[" << tag << "] " << method << ": inserting into HNSW..."
            << std::endl;
  size_t rss_before = getCurrentRSS();
  sw.reset();

  hnswlib::HierarchicalNSW<float> hnsw(&tqspace, N, M, ef_construction);
  hnsw.addPoint(codes[0].data(), 0);
  ParallelFor(1, N, g_build_threads, [&](size_t i, size_t) {
    hnsw.addPoint(codes[i].data(), i);
  });

  double insert_time = sw.elapsedSeconds();
  double build_time = encode_time + insert_time;
  size_t rss_after = getCurrentRSS();
  // Memory includes TQ index + base_data for re-ranking
  size_t tq_mem = (rss_after > rss_before) ? (rss_after - rss_before) / N : 0;
  size_t mem_per_vec = tq_mem + dim * sizeof(float);

  std::cerr << "[" << tag << "] " << method << ": build: " << std::fixed
            << std::setprecision(2) << build_time << " s" << std::endl;

  // Free codes — no longer needed, index has its own copy
  codes.clear();
  codes.shrink_to_fit();

  // Warmup
  {
    hnsw.setEf(10);
    tqspace.setSearchMode(hnsw);
    for (size_t q = 0; q < queries.size(); ++q) {
      auto pq = tqspace.prepareQuery(queries[q].data());
      auto result = hnsw.searchKnn(&pq, 10);
      (void)result;
    }
  }
  std::cerr << "[" << tag << "] " << method << " warmup done" << std::endl;

  // Sweep: for each ef, retrieve ef candidates via TQ, re-rank by exact L2
  bool first = true;
  for (size_t ef : ef_values) {
    hnsw.setEf(ef);
    std::atomic<size_t> correct(0);
    std::atomic<size_t> total(0);

    sw.reset();
    ParallelFor(0, queries.size(), g_search_threads, [&](size_t q, size_t) {
      const float *query = queries[q].data();

      // TQ search: retrieve ef candidates
      auto pq = tqspace.prepareQuery(query);
      auto tq_result = hnsw.searchKnn(&pq, ef);

      // L2 re-rank
      std::vector<std::pair<float, hnswlib::labeltype>> shortlist;
      shortlist.reserve(tq_result.size());
      while (!tq_result.empty()) {
        hnswlib::labeltype id = tq_result.top().second;
        float dist = 0.0f;
        const float *bv = base_data[id].data();
        for (size_t d = 0; d < dim; ++d) {
          float diff = query[d] - bv[d];
          dist += diff * diff;
        }
        shortlist.push_back({dist, id});
        tq_result.pop();
      }
      std::partial_sort(shortlist.begin(),
                        shortlist.begin() + std::min(K, shortlist.size()),
                        shortlist.end());

      // Recall
      std::unordered_set<hnswlib::labeltype> gt_set;
      for (size_t j = 0; j < K && j < gt[q].size(); ++j)
        gt_set.insert(static_cast<hnswlib::labeltype>(gt[q][j]));
      total.fetch_add(gt_set.size(), std::memory_order_relaxed);
      size_t local_correct = 0;
      for (size_t i = 0; i < K && i < shortlist.size(); ++i) {
        if (gt_set.count(shortlist[i].second))
          ++local_correct;
      }
      correct.fetch_add(local_correct, std::memory_order_relaxed);
    });

    double us_per_q = sw.elapsedSeconds() / queries.size() * 1e6;
    float recall =
        static_cast<float>(correct.load()) / static_cast<float>(total.load());
    emitCSV(tag, method, ef, recall, us_per_q, build_time, mem_per_vec, first);
    first = false;
    std::cerr << "[" << tag << "]   ef=" << ef << ": recall=" << std::fixed
              << std::setprecision(4) << recall << ", " << std::setprecision(1)
              << us_per_q << " us/q" << std::endl;
  }
}

// ---------------------------------------------------------------------------
// Benchmark: L2 graph build + TQ search + L2 re-ranking
// Graph topology from exact L2 on raw vectors; candidate retrieval uses
// TQ asymmetric distance; final ordering uses exact L2 from base_data.
// ---------------------------------------------------------------------------

static void benchmarkL2GraphTQRerank(
    const std::string &tag,
    const std::vector<std::vector<float>> &base_data,
    const std::vector<std::vector<float>> &queries,
    const std::vector<std::vector<int>> &gt, size_t dim, size_t M,
    size_t ef_construction, size_t K, const std::vector<size_t> &ef_values,
    int bits_per_coord = 4) {

  const size_t N = base_data.size();
  TurboQuantSpace tqspace(dim, bits_per_coord, 42, 137);
  size_t tq_code_size = tqspace.codeSizeBytes();
  size_t l2_data_size = dim * sizeof(float);

  std::string method =
      "l2graph_tq" + std::to_string(bits_per_coord) + "_rerank";

  if (tq_code_size > l2_data_size) {
    std::cerr << "[" << tag << "] " << method << ": SKIP (TQ code "
              << tq_code_size << " B > L2 slot " << l2_data_size << " B)"
              << std::endl;
    return;
  }

  std::cerr << "[" << tag << "] " << method << ": building L2 graph (" << N
            << " vectors)..." << std::endl;

  hnswlib::L2Space l2space(dim);
  size_t rss_before = getCurrentRSS();
  StopWatch sw;

  hnswlib::HierarchicalNSW<float> hnsw(&l2space, N, M, ef_construction);
  hnsw.addPoint(base_data[0].data(), 0);
  ParallelFor(1, N, g_build_threads, [&](size_t i, size_t) {
    hnsw.addPoint(base_data[i].data(), i);
  });

  double l2_build_time = sw.elapsedSeconds();
  std::cerr << "[" << tag << "] " << method << ": L2 build: " << std::fixed
            << std::setprecision(2) << l2_build_time << " s" << std::endl;

  // Replace stored data with TQ codes
  std::cerr << "[" << tag << "] " << method
            << ": replacing data with TQ codes..." << std::endl;
  sw.reset();
  std::vector<char> code_buf(tq_code_size);
  for (size_t i = 0; i < N; ++i) {
    tqspace.encodeVector(base_data[i].data(), code_buf.data());
    char *slot = hnsw.getDataByInternalId(static_cast<hnswlib::tableint>(i));
    memset(slot, 0, l2_data_size);
    memcpy(slot, code_buf.data(), tq_code_size);
  }
  double encode_time = sw.elapsedSeconds();
  double build_time = l2_build_time + encode_time;

  std::cerr << "[" << tag << "] " << method
            << ": encode+replace: " << std::fixed << std::setprecision(2)
            << encode_time << " s (total: " << build_time << " s)" << std::endl;

  // Swap distance function to TQ asymmetric search (PreparedQuery × code).
  tqspace.setSearchMode(hnsw);

  size_t rss_after = getCurrentRSS();
  size_t tq_mem = (rss_after > rss_before) ? (rss_after - rss_before) / N : 0;
  size_t mem_per_vec = tq_mem + dim * sizeof(float);

  // Warmup
  {
    hnsw.setEf(10);
    for (size_t q = 0; q < queries.size(); ++q) {
      auto pq = tqspace.prepareQuery(queries[q].data());
      auto result = hnsw.searchKnn(&pq, 10);
      (void)result;
    }
  }
  std::cerr << "[" << tag << "] " << method << " warmup done" << std::endl;

  // Sweep: TQ search for ef candidates, re-rank by exact L2 from base_data
  bool first = true;
  for (size_t ef : ef_values) {
    hnsw.setEf(ef);
    std::atomic<size_t> correct(0);
    std::atomic<size_t> total(0);

    sw.reset();
    ParallelFor(0, queries.size(), g_search_threads, [&](size_t q, size_t) {
      const float *query = queries[q].data();

      auto pq = tqspace.prepareQuery(query);
      auto tq_result = hnsw.searchKnn(&pq, ef);

      std::vector<std::pair<float, hnswlib::labeltype>> shortlist;
      shortlist.reserve(tq_result.size());
      while (!tq_result.empty()) {
        hnswlib::labeltype id = tq_result.top().second;
        float dist = 0.0f;
        const float *bv = base_data[id].data();
        for (size_t d = 0; d < dim; ++d) {
          float diff = query[d] - bv[d];
          dist += diff * diff;
        }
        shortlist.push_back({dist, id});
        tq_result.pop();
      }
      std::partial_sort(shortlist.begin(),
                        shortlist.begin() + std::min(K, shortlist.size()),
                        shortlist.end());

      std::unordered_set<hnswlib::labeltype> gt_set;
      for (size_t j = 0; j < K && j < gt[q].size(); ++j)
        gt_set.insert(static_cast<hnswlib::labeltype>(gt[q][j]));
      total.fetch_add(gt_set.size(), std::memory_order_relaxed);
      size_t local_correct = 0;
      for (size_t i = 0; i < K && i < shortlist.size(); ++i) {
        if (gt_set.count(shortlist[i].second))
          ++local_correct;
      }
      correct.fetch_add(local_correct, std::memory_order_relaxed);
    });

    double us_per_q = sw.elapsedSeconds() / queries.size() * 1e6;
    float recall =
        static_cast<float>(correct.load()) / static_cast<float>(total.load());
    emitCSV(tag, method, ef, recall, us_per_q, build_time, mem_per_vec, first);
    first = false;
    std::cerr << "[" << tag << "]   ef=" << ef << ": recall=" << std::fixed
              << std::setprecision(4) << recall << ", " << std::setprecision(1)
              << us_per_q << " us/q" << std::endl;
  }
}

// ---------------------------------------------------------------------------
// Benchmark: TurboQuantIndex::buildFromL2 via the high-level API.
// Same semantics as benchmarkL2GraphTQRerank but built through
// initL2Build + parallel addL2Point + finalizeFromL2 + searchRerank.
// Used to verify that the new public API produces identical numbers
// to the hand-wired version.
// ---------------------------------------------------------------------------

static void benchmarkTQIndexBuildFromL2(
    const std::string &tag,
    const std::vector<std::vector<float>> &base_data,
    const std::vector<std::vector<float>> &queries,
    const std::vector<std::vector<int>> &gt, size_t dim, size_t M,
    size_t ef_construction, size_t K, const std::vector<size_t> &ef_values,
    int bits_per_coord = 4) {

  const size_t N = base_data.size();
  std::string method =
      "tqidx_l2g_rerank" + std::to_string(bits_per_coord);

  std::cerr << "[" << tag << "] " << method << ": initL2Build (" << N
            << " vectors)..." << std::endl;

  TurboQuantIndex tqidx(dim, bits_per_coord, 42, 137);
  size_t rss_before = getCurrentRSS();
  StopWatch sw;

  tqidx.initL2Build(N, M, ef_construction);

  // Flatten base_data into a contiguous buffer once — finalizeFromL2
  // needs a single pointer, and addL2Point reads per-row.
  std::vector<float> flat(N * dim);
  for (size_t i = 0; i < N; ++i)
    std::memcpy(flat.data() + i * dim, base_data[i].data(),
                dim * sizeof(float));

  // First point single-threaded (entry point).
  tqidx.addL2Point(flat.data(), 0);
  ParallelFor(1, N, g_build_threads, [&](size_t i, size_t) {
    tqidx.addL2Point(flat.data() + i * dim, i);
  });

  double l2_build_time = sw.elapsedSeconds();
  std::cerr << "[" << tag << "] " << method << ": L2 build: " << std::fixed
            << std::setprecision(2) << l2_build_time << " s" << std::endl;

  sw.reset();
  auto status = tqidx.finalizeFromL2(flat.data(), N, /*keep_raw=*/true);
  if (!status.ok()) {
    std::cerr << "[" << tag << "] " << method
              << ": finalizeFromL2 failed: " << status.message() << std::endl;
    return;
  }
  double finalize_time = sw.elapsedSeconds();
  double build_time = l2_build_time + finalize_time;

  std::cerr << "[" << tag << "] " << method
            << ": finalize: " << std::fixed << std::setprecision(2)
            << finalize_time << " s (total: " << build_time << " s)"
            << std::endl;

  size_t rss_after = getCurrentRSS();
  size_t mem_per_vec =
      (rss_after > rss_before) ? (rss_after - rss_before) / N : 0;

  // Warmup
  tqidx.setEf(10);
  for (size_t q = 0; q < queries.size(); ++q) {
    (void)tqidx.searchRerank(queries[q].data(), 10, 10);
  }
  std::cerr << "[" << tag << "] " << method << " warmup done" << std::endl;

  bool first = true;
  for (size_t ef : ef_values) {
    tqidx.setEf(ef);
    std::atomic<size_t> correct(0);
    std::atomic<size_t> total(0);

    sw.reset();
    ParallelFor(0, queries.size(), g_search_threads, [&](size_t q, size_t) {
      auto shortlist = tqidx.searchRerank(queries[q].data(), K, ef);

      std::unordered_set<hnswlib::labeltype> gt_set;
      for (size_t j = 0; j < K && j < gt[q].size(); ++j)
        gt_set.insert(static_cast<hnswlib::labeltype>(gt[q][j]));
      total.fetch_add(gt_set.size(), std::memory_order_relaxed);
      size_t local_correct = 0;
      for (size_t i = 0; i < K && i < shortlist.size(); ++i) {
        if (gt_set.count(shortlist[i].second))
          ++local_correct;
      }
      correct.fetch_add(local_correct, std::memory_order_relaxed);
    });

    double us_per_q = sw.elapsedSeconds() / queries.size() * 1e6;
    float recall =
        static_cast<float>(correct.load()) / static_cast<float>(total.load());
    emitCSV(tag, method, ef, recall, us_per_q, build_time, mem_per_vec, first);
    first = false;
    std::cerr << "[" << tag << "]   ef=" << ef << ": recall=" << std::fixed
              << std::setprecision(4) << recall << ", " << std::setprecision(1)
              << us_per_q << " us/q" << std::endl;
  }
}

// ---------------------------------------------------------------------------
// Dataset descriptor
// ---------------------------------------------------------------------------

struct DatasetDesc {
  std::string name;
  std::string base_path;
  std::string query_path;
  std::string gt_path; // empty if no pre-computed GT
  bool is_bvecs;
  size_t max_queries; // 0 = use all
};

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
  // Usage: turbo_quant_recall_vs_qps [base_dir] [search_threads] [build_threads]
  //   search_threads: 0 => hardware_concurrency (default: 1)
  //   build_threads:  0 => hardware_concurrency (default: 0)
  std::string base_dir = ".";
  if (argc > 1)
    base_dir = argv[1];
  if (argc > 2)
    g_search_threads = static_cast<size_t>(std::atoi(argv[2]));
  if (argc > 3)
    g_build_threads = static_cast<size_t>(std::atoi(argv[3]));

  // Strip trailing slash
  while (base_dir.size() > 1 && base_dir.back() == '/')
    base_dir.pop_back();

  size_t effective_search =
      g_search_threads == 0 ? std::thread::hardware_concurrency()
                            : g_search_threads;
  size_t effective_build =
      g_build_threads == 0 ? std::thread::hardware_concurrency()
                           : g_build_threads;
  std::cerr << "[config] search_threads=" << g_search_threads << " ("
            << effective_search << " effective), build_threads="
            << g_build_threads << " (" << effective_build << " effective)"
            << std::endl;

  const size_t M = 16;
  const size_t EF_CONSTRUCTION = 200;
  const size_t K = 10;
  const std::vector<size_t> ef_values = {10, 16,  24,  32,  48, 64,
                                         96, 128, 200, 300, 500};

  std::vector<DatasetDesc> datasets = {
      {"bigann_sift128", base_dir + "/bigann/bigann_base.bvecs",
       base_dir + "/bigann/bigann_query.bvecs",
       "",   // no pre-computed GT
       true, // bvecs
       10000},
      {
          "glove100_128", base_dir + "/glove/glove100_128/base.fvecs",
          base_dir + "/glove/glove100_128/query.fvecs",
          base_dir + "/glove/glove100_128/gt.ivecs",
          false, // fvecs
          0      // use all
      },
      {"glove200_256", base_dir + "/glove/glove200_256/base.fvecs",
       base_dir + "/glove/glove200_256/query.fvecs",
       base_dir + "/glove/glove200_256/gt.ivecs", false, 0}};

  // CSV header
  std::cout << "dataset,method,ef,recall,us_per_query,build_time_s,"
            << "memory_bytes_per_vec,search_threads,build_threads"
            << std::endl;

  for (const auto &ds : datasets) {
    // Check if dataset exists
    {
      std::ifstream test(ds.base_path, std::ios::binary);
      if (!test.good()) {
        std::cerr << "[" << ds.name << "] SKIP: cannot open " << ds.base_path
                  << std::endl;
        continue;
      }
    }

    // Load base
    std::cerr << "[" << ds.name << "] Loading base from " << ds.base_path
              << "..." << std::endl;
    std::vector<std::vector<float>> base_data;
    bool loaded = ds.is_bvecs ? loadBvecs(ds.base_path, base_data)
                              : loadFvecs(ds.base_path, base_data);
    if (!loaded) {
      std::cerr << "[" << ds.name << "] SKIP: failed to load base" << std::endl;
      continue;
    }
    size_t dim = base_data[0].size();
    size_t N = base_data.size();
    std::cerr << "[" << ds.name << "] Loaded " << N
              << " base vectors, dim=" << dim << std::endl;

    // Load queries
    std::cerr << "[" << ds.name << "] Loading queries from " << ds.query_path
              << "..." << std::endl;
    std::vector<std::vector<float>> queries;
    loaded = ds.is_bvecs ? loadBvecs(ds.query_path, queries, ds.max_queries)
                         : loadFvecs(ds.query_path, queries, ds.max_queries);
    if (!loaded) {
      std::cerr << "[" << ds.name << "] SKIP: failed to load queries"
                << std::endl;
      continue;
    }
    if (ds.max_queries > 0 && queries.size() > ds.max_queries)
      queries.resize(ds.max_queries);
    std::cerr << "[" << ds.name << "] Using " << queries.size() << " queries"
              << std::endl;

    // Load or compute ground truth
    std::vector<std::vector<int>> gt;
    if (!ds.gt_path.empty()) {
      std::ifstream gt_test(ds.gt_path, std::ios::binary);
      if (gt_test.good()) {
        gt_test.close();
        std::cerr << "[" << ds.name << "] Loading GT from " << ds.gt_path
                  << "..." << std::endl;
        loadIvecs(ds.gt_path, gt, queries.size());
        std::cerr << "[" << ds.name << "] Loaded " << gt.size() << " GT entries"
                  << std::endl;
      }
    }
    if (gt.empty()) {
      gt = computeBruteForceGT(ds.name, base_data, queries, K);
    }

    std::cerr << "[" << ds.name << "] === Starting benchmarks ===" << std::endl;

    benchmarkL2(ds.name, base_data, queries, gt, dim, M, EF_CONSTRUCTION, K,
                ef_values);

    for (int bits : {4, 8}) {
      benchmarkTQ(ds.name, base_data, queries, gt, dim, M, EF_CONSTRUCTION, K,
                  ef_values, bits);

      benchmarkL2GraphTQ(ds.name, base_data, queries, gt, dim, M,
                         EF_CONSTRUCTION, K, ef_values, bits);

      benchmarkTQRerank(ds.name, base_data, queries, gt, dim, M,
                        EF_CONSTRUCTION, K, ef_values, bits);

      benchmarkL2GraphTQRerank(ds.name, base_data, queries, gt, dim, M,
                               EF_CONSTRUCTION, K, ef_values, bits);

      benchmarkTQIndexBuildFromL2(ds.name, base_data, queries, gt, dim, M,
                                  EF_CONSTRUCTION, K, ef_values, bits);
    }

    std::cerr << "[" << ds.name << "] === Done ===" << std::endl;
  }

  return 0;
}
