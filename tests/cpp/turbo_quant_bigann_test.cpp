/// turbo_quant_bigann_test.cpp — Comparison of L2Space vs TurboQuantSpace
/// on real BigANN SIFT1M data (128d, uint8 → float).
///
/// Reports recall@10 vs ef tradeoff, build time, search time, memory.
/// Requires BigANN data files in bigann/ directory
/// (run tests/cpp/download_bigann.py first).

#include "hnswlib/turbo_quant_space.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <queue>
#include <unordered_set>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <numeric>

#ifdef __APPLE__
#include <mach/mach.h>
#endif

using namespace hnswlib::turboquant;

// ---------------------------------------------------------------------------
// Helpers
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
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  (task_info_t)&info, &count) == KERN_SUCCESS)
        return info.resident_size;
#elif defined(__linux__)
    long rss = 0;
    FILE* fp = fopen("/proc/self/statm", "r");
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

static size_t fileSize(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    return f.good() ? static_cast<size_t>(f.tellg()) : 0;
}

// ---------------------------------------------------------------------------
// BigANN data loading (bvecs / ivecs / fvecs format)
// ---------------------------------------------------------------------------

/// Read bvecs file: each record is [dim: int32][bytes: uint8 × dim].
/// Returns vectors as float (normalized to [0,1] range or raw uint8→float).
static bool loadBvecs(const std::string& path,
                      std::vector<std::vector<float>>& out,
                      size_t max_elements = 0) {
    std::ifstream in(path, std::ios::binary);
    if (!in.good()) {
        std::cerr << "Cannot open " << path << std::endl;
        return false;
    }

    out.clear();
    while (in.good() && !in.eof()) {
        int dim = 0;
        in.read(reinterpret_cast<char*>(&dim), sizeof(int));
        if (!in.good()) break;

        std::vector<uint8_t> buf(dim);
        in.read(reinterpret_cast<char*>(buf.data()), dim);
        if (!in.good()) break;

        std::vector<float> vec(dim);
        for (int i = 0; i < dim; ++i)
            vec[i] = static_cast<float>(buf[i]);

        out.push_back(std::move(vec));
        if (max_elements > 0 && out.size() >= max_elements) break;
    }
    return !out.empty();
}

/// Read ivecs file: each record is [dim: int32][ints: int32 × dim].
static bool loadIvecs(const std::string& path,
                      std::vector<std::vector<int>>& out,
                      size_t max_elements = 0) {
    std::ifstream in(path, std::ios::binary);
    if (!in.good()) {
        std::cerr << "Cannot open " << path << std::endl;
        return false;
    }

    out.clear();
    while (in.good() && !in.eof()) {
        int dim = 0;
        in.read(reinterpret_cast<char*>(&dim), sizeof(int));
        if (!in.good()) break;

        std::vector<int> vec(dim);
        in.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(int));
        if (!in.good()) break;

        out.push_back(std::move(vec));
        if (max_elements > 0 && out.size() >= max_elements) break;
    }
    return !out.empty();
}

/// Read fvecs file: each record is [dim: int32][floats: float32 × dim].
static bool loadFvecs(const std::string& path,
                      std::vector<std::vector<float>>& out,
                      size_t max_elements = 0) {
    std::ifstream in(path, std::ios::binary);
    if (!in.good()) {
        std::cerr << "Cannot open " << path << std::endl;
        return false;
    }

    out.clear();
    while (in.good() && !in.eof()) {
        int dim = 0;
        in.read(reinterpret_cast<char*>(&dim), sizeof(int));
        if (!in.good()) break;

        std::vector<float> vec(dim);
        in.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(float));
        if (!in.good()) break;

        out.push_back(std::move(vec));
        if (max_elements > 0 && out.size() >= max_elements) break;
    }
    return !out.empty();
}

// TurboQuantSpace is now in hnswlib/turbo_quant_space.h

// ---------------------------------------------------------------------------
// Brute-force ground truth computation
// ---------------------------------------------------------------------------

/// Compute exact K nearest neighbors for each query using brute-force L2.
/// Returns gt[q] = vector of K nearest base vector IDs for query q.
static std::vector<std::vector<int>> computeBruteForceGT(
    const std::vector<std::vector<float>>& base,
    const std::vector<std::vector<float>>& queries,
    size_t k) {
    const size_t N = base.size();
    const size_t Q = queries.size();
    const size_t D = base[0].size();
    std::vector<std::vector<int>> gt(Q);

    std::cout << "Computing brute-force ground truth (" << Q << " queries x "
              << N << " base)..." << std::flush;
    StopWatch sw;

    hnswlib::L2Space l2space(D);
    hnswlib::BruteforceSearch<float> bf(&l2space, N);
    for (size_t i = 0; i < N; ++i)
        bf.addPoint(base[i].data(), i);

    for (size_t q = 0; q < Q; ++q) {
        auto result = bf.searchKnn(queries[q].data(), k);
        gt[q].resize(result.size());
        size_t idx = result.size();
        while (!result.empty()) {
            gt[q][--idx] = static_cast<int>(result.top().second);
            result.pop();
        }
    }

    std::cout << " done (" << std::fixed << std::setprecision(1)
              << sw.elapsedSeconds() << " s)" << std::endl;
    return gt;
}

// ---------------------------------------------------------------------------
// Recall measurement
// ---------------------------------------------------------------------------

/// Compute recall@k for HNSW search results against ground truth.
static float measureRecall(
    hnswlib::HierarchicalNSW<float>& hnsw,
    const std::vector<std::vector<float>>& queries,
    const std::vector<std::vector<int>>& gt,
    size_t k) {
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

/// Same as above but with TurboQuant query preparation.
static float measureRecallTQ(
    hnswlib::HierarchicalNSW<float>& hnsw,
    TurboQuantSpace& space,
    const std::vector<std::vector<float>>& queries,
    const std::vector<std::vector<int>>& gt,
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
// Main test
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    // Configuration
    size_t N = 1000000;  // number of base vectors (capped by file size)
    size_t NUM_QUERIES = 1000;  // use subset of queries for brute-force GT
    size_t M = 16;
    size_t EF_CONSTRUCTION = 200;
    size_t K = 10;

    // Usage: turbo_quant_bigann_test <data_dir> [N] [num_queries]
    //
    // Supported data_dir layouts:
    //   bvecs: <dir>/bigann_base.bvecs + bigann_query.bvecs
    //   fvecs: <dir>/base.fvecs + query.fvecs [+ gt.ivecs]

    std::string data_dir = "bigann";
    if (argc > 1) data_dir = argv[1];
    if (argc > 2) N = std::atol(argv[2]);
    if (argc > 3) NUM_QUERIES = std::atol(argv[3]);

    // Detect format: check for base.fvecs first, then bigann_base.bvecs
    std::string base_path, query_path, gt_path;
    bool use_fvecs = false;
    {
        std::string fvecs_base = data_dir + "/base.fvecs";
        std::ifstream test(fvecs_base, std::ios::binary);
        if (test.good()) {
            use_fvecs = true;
            base_path = fvecs_base;
            query_path = data_dir + "/query.fvecs";
            gt_path = data_dir + "/gt.ivecs";
        } else {
            base_path = data_dir + "/bigann_base.bvecs";
            query_path = data_dir + "/bigann_query.bvecs";
        }
    }

    std::cout << "=== TurboQuant vs L2Space benchmark ===" << std::endl;
    std::cout << "Data dir: " << data_dir << " (format: "
              << (use_fvecs ? "fvecs" : "bvecs") << ")" << std::endl;

    // -----------------------------------------------------------------------
    // Load data
    // -----------------------------------------------------------------------
    std::cout << "Loading base vectors from " << base_path << "..." << std::endl;
    std::vector<std::vector<float>> base_data;
    bool loaded = use_fvecs
        ? loadFvecs(base_path, base_data, N)
        : loadBvecs(base_path, base_data, N);
    if (!loaded) {
        std::cerr << "Failed to load base data." << std::endl;
        return 1;
    }
    size_t vecdim = base_data[0].size();
    N = base_data.size();
    std::cout << "  Loaded " << N << " vectors, dim=" << vecdim << std::endl;

    std::cout << "Loading queries from " << query_path << "..." << std::endl;
    std::vector<std::vector<float>> all_queries;
    loaded = use_fvecs
        ? loadFvecs(query_path, all_queries, NUM_QUERIES)
        : loadBvecs(query_path, all_queries, NUM_QUERIES);
    if (!loaded) {
        std::cerr << "Failed to load queries." << std::endl;
        return 1;
    }
    std::vector<std::vector<float>> queries(
        all_queries.begin(),
        all_queries.begin() + std::min(NUM_QUERIES, all_queries.size()));
    std::cout << "  Using " << queries.size() << " queries" << std::endl;

    std::cout << std::endl;
    std::cout << "N=" << N << ", dim=" << vecdim << ", M=" << M
              << ", ef_construction=" << EF_CONSTRUCTION
              << ", K=" << K << ", queries=" << queries.size() << std::endl;
    std::cout << std::endl;

    // -----------------------------------------------------------------------
    // Load or compute ground truth
    // -----------------------------------------------------------------------
    std::vector<std::vector<int>> gt;
    if (use_fvecs) {
        // Try loading pre-computed GT
        std::ifstream gt_test(gt_path, std::ios::binary);
        if (gt_test.good()) {
            gt_test.close();
            std::cout << "Loading ground truth from " << gt_path << "..."
                      << std::endl;
            loadIvecs(gt_path, gt, queries.size());
            std::cout << "  Loaded " << gt.size() << " GT entries" << std::endl;
        }
    }
    if (gt.empty()) {
        gt = computeBruteForceGT(base_data, queries, K);
    }
    std::cout << std::endl;

    // ef values for recall vs speed tradeoff
    std::vector<size_t> ef_values = {10, 16, 24, 32, 48, 64, 96, 128, 200, 300, 500};

    // -----------------------------------------------------------------------
    // Part A: L2Space (float32)
    // -----------------------------------------------------------------------
    std::cout << "============================================" << std::endl;
    std::cout << "  L2Space (float32, " << vecdim * 4 << " bytes/vec)"
              << std::endl;
    std::cout << "============================================" << std::endl;

    size_t rss_before, rss_after;
    {
        hnswlib::L2Space l2space(vecdim);
        rss_before = getCurrentRSS();

        StopWatch sw;
        hnswlib::HierarchicalNSW<float> hnsw_l2(&l2space, N, M, EF_CONSTRUCTION);

        hnsw_l2.addPoint(base_data[0].data(), 0);
#pragma omp parallel for schedule(dynamic)
        for (int i = 1; i < static_cast<int>(N); ++i) {
            hnsw_l2.addPoint(base_data[i].data(), static_cast<size_t>(i));
        }
        double build_time = sw.elapsedSeconds();
        rss_after = getCurrentRSS();

        std::cout << "  Build time: " << std::fixed << std::setprecision(2)
                  << build_time << " s" << std::endl;
        std::cout << "  RSS delta:  "
                  << ((rss_after > rss_before)
                      ? (rss_after - rss_before) / (1024.0 * 1024.0) : 0.0)
                  << " MB" << std::endl;
        std::cout << "  data_size:  " << l2space.get_data_size()
                  << " B/vec" << std::endl;

        // Save and report file size
        const std::string file_l2 = "bench_bigann_l2.bin";
        hnsw_l2.saveIndex(file_l2);
        std::cout << "  Index file: "
                  << fileSize(file_l2) / (1024.0 * 1024.0) << " MB"
                  << std::endl;
        std::remove(file_l2.c_str());

        // Recall vs ef
        std::cout << "\n  ef\trecall@" << K << "\ttime(us/q)" << std::endl;
        std::cout << "  ----\t--------\t----------" << std::endl;
        for (size_t ef : ef_values) {
            hnsw_l2.setEf(ef);
            sw.reset();
            float recall = measureRecall(hnsw_l2, queries, gt, K);
            double search_time = sw.elapsedSeconds();
            double us_per_query = search_time / queries.size() * 1e6;
            std::cout << "  " << ef << "\t" << std::fixed
                      << std::setprecision(4) << recall << "\t\t"
                      << std::setprecision(1) << us_per_query << std::endl;
        }
        std::cout << std::endl;
    }

    // -----------------------------------------------------------------------
    // Part B: TurboQuantSpace (4 bits/coord)
    // -----------------------------------------------------------------------
    std::cout << "============================================" << std::endl;
    std::cout << "  TurboQuantSpace (4 bits/coord)" << std::endl;
    std::cout << "============================================" << std::endl;

    {
        TurboQuantSpace tqspace(vecdim, 4, 42, 137);

        std::cout << "  data_size:  " << tqspace.get_data_size()
                  << " B/vec (compression: "
                  << std::fixed << std::setprecision(1)
                  << static_cast<float>(vecdim * 4) / tqspace.codeSizeBytes()
                  << "x)" << std::endl;

        rss_before = getCurrentRSS();
        StopWatch sw;
        hnswlib::HierarchicalNSW<float> hnsw_tq(&tqspace, N, M, EF_CONSTRUCTION);

        // Encode and insert (single-threaded encoding, parallel insert is
        // tricky because TurboQuantSpace::beginSearch/endSearch use shared
        // state — keep it simple).
        std::vector<char> buf(tqspace.codeSizeBytes());
        tqspace.encodeVector(base_data[0].data(), buf.data());
        hnsw_tq.addPoint(buf.data(), 0);

        for (size_t i = 1; i < N; ++i) {
            tqspace.encodeVector(base_data[i].data(), buf.data());
            hnsw_tq.addPoint(buf.data(), static_cast<size_t>(i));
            if ((i + 1) % 100000 == 0)
                std::cout << "    inserted " << (i + 1) / 1000 << "K..."
                          << std::endl;
        }
        double build_time = sw.elapsedSeconds();
        rss_after = getCurrentRSS();

        std::cout << "  Build time: " << std::fixed << std::setprecision(2)
                  << build_time << " s" << std::endl;
        std::cout << "  RSS delta:  "
                  << ((rss_after > rss_before)
                      ? (rss_after - rss_before) / (1024.0 * 1024.0) : 0.0)
                  << " MB" << std::endl;

        // Save and report file size
        const std::string file_tq = "bench_bigann_tq.bin";
        hnsw_tq.saveIndex(file_tq);
        std::cout << "  Index file: "
                  << fileSize(file_tq) / (1024.0 * 1024.0) << " MB"
                  << std::endl;
        std::remove(file_tq.c_str());

        // Recall vs ef
        std::cout << "\n  ef\trecall@" << K << "\ttime(us/q)" << std::endl;
        std::cout << "  ----\t--------\t----------" << std::endl;
        for (size_t ef : ef_values) {
            hnsw_tq.setEf(ef);
            sw.reset();
            float recall = measureRecallTQ(hnsw_tq, tqspace, queries, gt, K);
            double search_time = sw.elapsedSeconds();
            double us_per_query = search_time / queries.size() * 1e6;
            std::cout << "  " << ef << "\t" << std::fixed
                      << std::setprecision(4) << recall << "\t\t"
                      << std::setprecision(1) << us_per_query << std::endl;
        }
        std::cout << std::endl;
    }

    // -----------------------------------------------------------------------
    // Part C: Brute-force TQ recall (isolates distance quality from graph)
    // -----------------------------------------------------------------------
    std::cout << "============================================" << std::endl;
    std::cout << "  Brute-force TQ recall (distance quality)" << std::endl;
    std::cout << "============================================" << std::endl;

    {
        // Use a subset for brute-force feasibility
        size_t BF_N = std::min(N, static_cast<size_t>(50000));
        size_t BF_Q = std::min(queries.size(), static_cast<size_t>(200));
        std::cout << "  N=" << BF_N << ", queries=" << BF_Q << std::endl;

        TurboQuantSpace tqspace(vecdim, 4, 42, 137);

        // Compute brute-force GT for this subset
        std::vector<std::vector<float>> bf_base(
            base_data.begin(), base_data.begin() + BF_N);
        std::vector<std::vector<float>> bf_queries(
            queries.begin(), queries.begin() + BF_Q);
        auto bf_gt = computeBruteForceGT(bf_base, bf_queries, K);

        // Encode all vectors
        std::vector<std::vector<char>> codes(BF_N,
            std::vector<char>(tqspace.codeSizeBytes()));
        for (size_t i = 0; i < BF_N; ++i)
            tqspace.encodeVector(bf_base[i].data(), codes[i].data());

        // Brute-force TQ search: for each query, compute TQ distance to all
        // vectors and find top-K
        auto dist_func = tqspace.get_dist_func();
        auto* dist_param = tqspace.get_dist_func_param();

        size_t total_hits = 0;
        for (size_t q = 0; q < BF_Q; ++q) {
            auto pq = tqspace.prepareQuery(bf_queries[q].data());
            tqspace.beginSearch(pq);

            // Compute all distances
            std::vector<std::pair<float, size_t>> dists(BF_N);
            for (size_t i = 0; i < BF_N; ++i) {
                dists[i] = {dist_func(bf_queries[q].data(),
                                      codes[i].data(), dist_param), i};
            }
            tqspace.endSearch();

            std::partial_sort(dists.begin(), dists.begin() + K, dists.end());

            std::unordered_set<size_t> gt_set;
            for (size_t j = 0; j < K && j < bf_gt[q].size(); ++j)
                gt_set.insert(static_cast<size_t>(bf_gt[q][j]));

            for (size_t i = 0; i < K; ++i) {
                if (gt_set.count(dists[i].second))
                    ++total_hits;
            }
        }
        float bf_tq_recall = static_cast<float>(total_hits)
                           / static_cast<float>(BF_Q * K);

        std::cout << "  Brute-force TQ Recall@" << K << ": "
                  << std::fixed << std::setprecision(2)
                  << (bf_tq_recall * 100.0f) << "%" << std::endl;
        std::cout << "  (This isolates distance quality from graph quality)"
                  << std::endl;
        std::cout << std::endl;
    }

    // -----------------------------------------------------------------------
    // Part D: Hybrid — L2 graph + TQ search (isolates graph quality)
    // Uses a smaller subset to keep build time reasonable.
    // -----------------------------------------------------------------------
    std::cout << "============================================" << std::endl;
    std::cout << "  Hybrid: L2 graph + TQ search" << std::endl;
    std::cout << "============================================" << std::endl;

    {
        size_t HY_N = std::min(N, static_cast<size_t>(100000));
        size_t HY_Q = std::min(queries.size(), static_cast<size_t>(500));
        std::cout << "  N=" << HY_N << ", queries=" << HY_Q << std::endl;

        // Build L2 graph on raw data
        hnswlib::L2Space l2space(vecdim);
        hnswlib::HierarchicalNSW<float> hnsw_l2(&l2space, HY_N, M, EF_CONSTRUCTION);
        hnsw_l2.addPoint(base_data[0].data(), 0);
#pragma omp parallel for schedule(dynamic)
        for (int i = 1; i < static_cast<int>(HY_N); ++i)
            hnsw_l2.addPoint(base_data[i].data(), static_cast<size_t>(i));

        // Build TQ graph on encoded data
        TurboQuantSpace tqspace(vecdim, 4, 42, 137);
        hnswlib::HierarchicalNSW<float> hnsw_tq(&tqspace, HY_N, M, EF_CONSTRUCTION);
        std::vector<char> buf(tqspace.codeSizeBytes());
        tqspace.encodeVector(base_data[0].data(), buf.data());
        hnsw_tq.addPoint(buf.data(), 0);
        for (size_t i = 1; i < HY_N; ++i) {
            tqspace.encodeVector(base_data[i].data(), buf.data());
            hnsw_tq.addPoint(buf.data(), i);
        }

        // Compute GT for subset
        std::vector<std::vector<float>> hy_base(
            base_data.begin(), base_data.begin() + HY_N);
        std::vector<std::vector<float>> hy_queries(
            queries.begin(), queries.begin() + HY_Q);
        auto hy_gt = computeBruteForceGT(hy_base, hy_queries, K);

        // Search: L2 graph + L2 search (baseline)
        // Search: TQ graph + TQ search (current)
        // We already have both indices, just measure recall at ef=64, 200
        std::vector<size_t> hy_efs = {32, 64, 128, 200, 500};

        std::cout << "\n  Config\t\t\tef=32\tef=64\tef=128\tef=200\tef=500"
                  << std::endl;
        std::cout << "  -----\t\t\t\t-----\t-----\t------\t------\t------"
                  << std::endl;

        // L2 graph + L2 search
        std::cout << "  L2 graph + L2 search\t\t";
        for (size_t ef : hy_efs) {
            hnsw_l2.setEf(ef);
            float r = measureRecall(hnsw_l2, hy_queries, hy_gt, K);
            std::cout << std::fixed << std::setprecision(3) << r << "\t";
        }
        std::cout << std::endl;

        // TQ graph + TQ search
        std::cout << "  TQ graph + TQ search\t\t";
        for (size_t ef : hy_efs) {
            hnsw_tq.setEf(ef);
            float r = measureRecallTQ(hnsw_tq, tqspace, hy_queries, hy_gt, K);
            std::cout << std::fixed << std::setprecision(3) << r << "\t";
        }
        std::cout << std::endl;

        // Hybrid: L2 graph + TQ search
        // Replace the data in L2 index with encoded TQ data, but keep the graph.
        // Trick: we can't easily swap SpaceInterface, so instead we build a new
        // TQ HNSW but copy the graph from L2.
        // Simpler approach: just search the L2 graph but use TQ distance for
        // reranking. But that requires modifying HNSW internals.
        //
        // Alternative: build a new TQ HNSW from scratch but insert in the same
        // order the L2 graph would use — not possible directly.
        //
        // Simplest diagnostic: measure TQ brute-force recall at this scale.
        auto dist_func = tqspace.get_dist_func();
        auto* dist_param = tqspace.get_dist_func_param();

        // Encode all HY_N vectors
        std::vector<std::vector<char>> codes(HY_N,
            std::vector<char>(tqspace.codeSizeBytes()));
        for (size_t i = 0; i < HY_N; ++i)
            tqspace.encodeVector(hy_base[i].data(), codes[i].data());

        std::cout << "  TQ brute-force\t\t";
        {
            size_t total_hits = 0;
            for (size_t q = 0; q < HY_Q; ++q) {
                auto pq = tqspace.prepareQuery(hy_queries[q].data());
                tqspace.beginSearch(pq);
                std::vector<std::pair<float, size_t>> dists(HY_N);
                for (size_t i = 0; i < HY_N; ++i) {
                    dists[i] = {dist_func(hy_queries[q].data(),
                                          codes[i].data(), dist_param), i};
                }
                tqspace.endSearch();
                std::partial_sort(dists.begin(), dists.begin() + K, dists.end());

                std::unordered_set<size_t> gt_set;
                for (size_t j = 0; j < K && j < hy_gt[q].size(); ++j)
                    gt_set.insert(static_cast<size_t>(hy_gt[q][j]));
                for (size_t i = 0; i < K; ++i) {
                    if (gt_set.count(dists[i].second))
                        ++total_hits;
                }
            }
            float r = static_cast<float>(total_hits)
                    / static_cast<float>(HY_Q * K);
            // Print same value for all ef columns (brute-force doesn't depend on ef)
            for (size_t i = 0; i < hy_efs.size(); ++i)
                std::cout << std::fixed << std::setprecision(3) << r << "\t";
        }
        std::cout << std::endl;
        std::cout << std::endl;
    }

    // -----------------------------------------------------------------------
    // Part E: L2 graph build → replace data with TQ codes → TQ search
    // This is the correct approach: exact graph, compressed storage.
    // -----------------------------------------------------------------------
    std::cout << "============================================" << std::endl;
    std::cout << "  L2 graph + TQ data (exact build, compressed search)"
              << std::endl;
    std::cout << "============================================" << std::endl;

    {
        // Use full dataset
        TurboQuantSpace tqspace(vecdim, 4, 42, 137);
        size_t tq_code_size = tqspace.codeSizeBytes();
        size_t l2_data_size = vecdim * sizeof(float);

        // TQ code must fit in L2 data slot
        if (tq_code_size > l2_data_size) {
            std::cout << "  SKIP: TQ code (" << tq_code_size
                      << " B) > L2 slot (" << l2_data_size << " B)"
                      << std::endl;
        } else {
            std::cout << "  Building L2 graph on " << N << " vectors..."
                      << std::endl;

            // Step 1: Build graph with exact L2 distances
            hnswlib::L2Space l2space(vecdim);
            hnswlib::HierarchicalNSW<float> hnsw(&l2space, N, M, EF_CONSTRUCTION);

            StopWatch sw;
            hnsw.addPoint(base_data[0].data(), 0);
#pragma omp parallel for schedule(dynamic)
            for (int i = 1; i < static_cast<int>(N); ++i)
                hnsw.addPoint(base_data[i].data(), static_cast<size_t>(i));
            double build_time = sw.elapsedSeconds();
            std::cout << "  L2 build time: " << std::fixed
                      << std::setprecision(2) << build_time << " s"
                      << std::endl;

            // Step 2: Replace stored data with TQ codes
            std::cout << "  Replacing data with TQ codes..." << std::endl;
            sw.reset();
            std::vector<char> code_buf(tq_code_size);
            for (size_t i = 0; i < N; ++i) {
                tqspace.encodeVector(base_data[i].data(), code_buf.data());
                // Write TQ code into the data slot (zero-pad to l2_data_size)
                char* slot = hnsw.getDataByInternalId(
                    static_cast<hnswlib::tableint>(i));
                memset(slot, 0, l2_data_size);
                memcpy(slot, code_buf.data(), tq_code_size);
            }
            double encode_time = sw.elapsedSeconds();
            std::cout << "  Encode time: " << std::fixed
                      << std::setprecision(2) << encode_time << " s"
                      << std::endl;

            // Step 3: Switch distance function to TQ
            hnsw.fstdistfunc_ = tqspace.get_dist_func();
            hnsw.dist_func_param_ = tqspace.get_dist_func_param();

            // Step 4: Search with TQ distances
            std::cout << "\n  ef\trecall@" << K << "\ttime(us/q)" << std::endl;
            std::cout << "  ----\t--------\t----------" << std::endl;
            for (size_t ef : ef_values) {
                hnsw.setEf(ef);
                sw.reset();
                float recall = measureRecallTQ(hnsw, tqspace, queries, gt, K);
                double search_time = sw.elapsedSeconds();
                double us_per_query = search_time / queries.size() * 1e6;
                std::cout << "  " << ef << "\t" << std::fixed
                          << std::setprecision(4) << recall << "\t\t"
                          << std::setprecision(1) << us_per_query << std::endl;
            }
            std::cout << std::endl;
        }
    }

    // -----------------------------------------------------------------------
    // Part F: TQ search + L2 re-ranking from original data
    // TQ HNSW retrieves a shortlist, then re-ranks by exact L2 distance.
    // -----------------------------------------------------------------------
    std::cout << "============================================" << std::endl;
    std::cout << "  TQ search + L2 re-ranking" << std::endl;
    std::cout << "============================================" << std::endl;

    {
        TurboQuantSpace tqspace(vecdim, 4, 42, 137);

        // Build TQ graph
        std::cout << "  Building TQ graph on " << N << " vectors..."
                  << std::endl;
        hnswlib::HierarchicalNSW<float> hnsw_tq(&tqspace, N, M, EF_CONSTRUCTION);
        std::vector<char> buf(tqspace.codeSizeBytes());
        tqspace.encodeVector(base_data[0].data(), buf.data());
        hnsw_tq.addPoint(buf.data(), 0);
        for (size_t i = 1; i < N; ++i) {
            tqspace.encodeVector(base_data[i].data(), buf.data());
            hnsw_tq.addPoint(buf.data(), i);
            if ((i + 1) % 100000 == 0)
                std::cout << "    inserted " << (i + 1) / 1000 << "K..."
                          << std::endl;
        }

        // Re-ranking: TQ retrieves shortlist of size `ef`, then L2 re-rank
        // to find true top-K.
        std::vector<size_t> rerank_efs = {20, 50, 100, 200, 500, 1000};

        std::cout << "\n  ef(TQ)\trecall@" << K
                  << "\ttime(us/q)\t(shortlist → L2 re-rank → top-"
                  << K << ")" << std::endl;
        std::cout << "  --------\t--------\t----------" << std::endl;

        for (size_t ef : rerank_efs) {
            hnsw_tq.setEf(ef);
            size_t correct = 0;
            size_t total = 0;

            StopWatch sw;
            for (size_t q = 0; q < queries.size(); ++q) {
                const float* query = queries[q].data();

                // Step 1: TQ search → shortlist of size ef
                auto pq = tqspace.prepareQuery(query);
                tqspace.beginSearch(pq);
                auto tq_result = hnsw_tq.searchKnn(query, ef);
                tqspace.endSearch();

                // Step 2: L2 re-rank shortlist using original float data
                std::vector<std::pair<float, hnswlib::labeltype>> shortlist;
                shortlist.reserve(tq_result.size());
                while (!tq_result.empty()) {
                    hnswlib::labeltype id = tq_result.top().second;
                    // Compute exact L2 distance
                    float dist = 0.0f;
                    const float* base_vec = base_data[id].data();
                    for (size_t d = 0; d < vecdim; ++d) {
                        float diff = query[d] - base_vec[d];
                        dist += diff * diff;
                    }
                    shortlist.push_back({dist, id});
                    tq_result.pop();
                }

                // Step 3: Pick true top-K by exact L2
                std::partial_sort(shortlist.begin(),
                    shortlist.begin() + std::min(K, shortlist.size()),
                    shortlist.end());

                // Measure recall
                std::unordered_set<hnswlib::labeltype> gt_set;
                for (size_t j = 0; j < K && j < gt[q].size(); ++j)
                    gt_set.insert(static_cast<hnswlib::labeltype>(gt[q][j]));
                total += gt_set.size();
                for (size_t i = 0; i < K && i < shortlist.size(); ++i) {
                    if (gt_set.count(shortlist[i].second))
                        ++correct;
                }
            }
            double search_time = sw.elapsedSeconds();
            double us_per_query = search_time / queries.size() * 1e6;
            float recall = static_cast<float>(correct)
                         / static_cast<float>(total);
            std::cout << "  " << ef << "\t\t" << std::fixed
                      << std::setprecision(4) << recall << "\t\t"
                      << std::setprecision(1) << us_per_query << std::endl;
        }
        std::cout << std::endl;
    }

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    std::cout << "============================================" << std::endl;
    std::cout << "Memory comparison (per vector, data only):" << std::endl;
    std::cout << "  L2Space:        " << vecdim * sizeof(float)
              << " bytes" << std::endl;
    std::cout << "  TurboQuantSpace: "
              << TurboQuantCode::codeSizeBytes(vecdim) << " bytes ("
              << std::fixed << std::setprecision(1)
              << static_cast<float>(vecdim * sizeof(float))
                 / TurboQuantCode::codeSizeBytes(vecdim)
              << "x compression)" << std::endl;

    return 0;
}
