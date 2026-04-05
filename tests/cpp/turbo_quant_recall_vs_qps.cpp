/// turbo_quant_recall_vs_qps.cpp — Multi-dataset recall vs QPS benchmark.
///
/// Compares L2Space (float32) vs TurboQuantSpace across 3 datasets.
/// Outputs CSV to stdout, progress to stderr.
///
/// Usage: turbo_quant_recall_vs_qps [base_dir]
///   base_dir defaults to "." and should contain bigann/ and glove/ subdirs.

#include "hnswlib/turbo_quant_space.h"

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
#include <string>
#include <unordered_set>
#include <vector>

#ifdef __APPLE__
#include <mach/mach.h>
#endif

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
  size_t correct = 0;
  size_t total = 0;
  for (size_t q = 0; q < queries.size(); ++q) {
    auto result = hnsw.searchKnn(queries[q].data(), k);
    std::unordered_set<hnswlib::labeltype> gt_set;
    for (size_t j = 0; j < k && j < gt[q].size(); ++j)
      gt_set.insert(static_cast<hnswlib::labeltype>(gt[q][j]));
    total += gt_set.size();
    while (!result.empty()) {
      if (gt_set.count(result.top().second))
        ++correct;
      result.pop();
    }
  }
  return static_cast<float>(correct) / static_cast<float>(total);
}

static float measureRecallTQ(hnswlib::HierarchicalNSW<float> &hnsw,
                             TurboQuantSpace &space,
                             const std::vector<std::vector<float>> &queries,
                             const std::vector<std::vector<int>> &gt,
                             size_t k) {
  size_t correct = 0;
  size_t total = 0;
  for (size_t q = 0; q < queries.size(); ++q) {
    auto pq = space.prepareQuery(queries[q].data());
    space.beginSearch(pq);
    auto result = hnsw.searchKnn(queries[q].data(), k);
    space.endSearch();
    std::unordered_set<hnswlib::labeltype> gt_set;
    for (size_t j = 0; j < k && j < gt[q].size(); ++j)
      gt_set.insert(static_cast<hnswlib::labeltype>(gt[q][j]));
    total += gt_set.size();
    while (!result.empty()) {
      if (gt_set.count(result.top().second))
        ++correct;
      result.pop();
    }
  }
  return static_cast<float>(correct) / static_cast<float>(total);
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
  std::cout << "\n";
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
#pragma omp parallel for schedule(dynamic)
  for (int i = 1; i < static_cast<int>(N); ++i)
    hnsw.addPoint(base_data[i].data(), static_cast<size_t>(i));

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
#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < static_cast<int>(N); ++i)
    tqspace.encodeVector(base_data[i].data(), codes[i].data());

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
#pragma omp parallel for schedule(dynamic)
  for (int i = 1; i < static_cast<int>(N); ++i)
    hnsw.addPoint(codes[i].data(), static_cast<size_t>(i));

  double insert_time = sw.elapsedSeconds();
  double build_time = encode_time + insert_time;
  size_t rss_after = getCurrentRSS();
  size_t mem_per_vec =
      (rss_after > rss_before) ? (rss_after - rss_before) / N : 0;

  std::cerr << "[" << tag << "] " << method << " insert: " << std::fixed
            << std::setprecision(2) << insert_time
            << " s (total build: " << build_time << " s)" << std::endl;

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
#pragma omp parallel for schedule(dynamic)
  for (int i = 1; i < static_cast<int>(N); ++i)
    hnsw.addPoint(base_data[i].data(), static_cast<size_t>(i));

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

  // Swap distance function to TQ
  hnsw.fstdistfunc_ = tqspace.get_dist_func();
  hnsw.dist_func_param_ = tqspace.get_dist_func_param();

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
#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < static_cast<int>(N); ++i)
    tqspace.encodeVector(base_data[i].data(), codes[i].data());

  double encode_time = sw.elapsedSeconds();

  // Parallel insert
  std::cerr << "[" << tag << "] " << method << ": inserting into HNSW..."
            << std::endl;
  size_t rss_before = getCurrentRSS();
  sw.reset();

  hnswlib::HierarchicalNSW<float> hnsw(&tqspace, N, M, ef_construction);
  hnsw.addPoint(codes[0].data(), 0);
#pragma omp parallel for schedule(dynamic)
  for (int i = 1; i < static_cast<int>(N); ++i)
    hnsw.addPoint(codes[i].data(), static_cast<size_t>(i));

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
    for (size_t q = 0; q < queries.size(); ++q) {
      auto pq = tqspace.prepareQuery(queries[q].data());
      tqspace.beginSearch(pq);
      auto result = hnsw.searchKnn(queries[q].data(), 10);
      tqspace.endSearch();
      // Discard
      (void)result;
    }
  }
  std::cerr << "[" << tag << "] " << method << " warmup done" << std::endl;

  // Sweep: for each ef, retrieve ef candidates via TQ, re-rank by exact L2
  bool first = true;
  for (size_t ef : ef_values) {
    hnsw.setEf(ef);
    size_t correct = 0;
    size_t total = 0;

    sw.reset();
    for (size_t q = 0; q < queries.size(); ++q) {
      const float *query = queries[q].data();

      // TQ search: retrieve ef candidates
      auto pq = tqspace.prepareQuery(query);
      tqspace.beginSearch(pq);
      auto tq_result = hnsw.searchKnn(query, ef);
      tqspace.endSearch();

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
      total += gt_set.size();
      for (size_t i = 0; i < K && i < shortlist.size(); ++i) {
        if (gt_set.count(shortlist[i].second))
          ++correct;
      }
    }

    double us_per_q = sw.elapsedSeconds() / queries.size() * 1e6;
    float recall = static_cast<float>(correct) / static_cast<float>(total);
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
  std::string base_dir = ".";
  if (argc > 1)
    base_dir = argv[1];

  // Strip trailing slash
  while (base_dir.size() > 1 && base_dir.back() == '/')
    base_dir.pop_back();

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
            << "memory_bytes_per_vec" << std::endl;

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
    }

    std::cerr << "[" << ds.name << "] === Done ===" << std::endl;
  }

  return 0;
}
