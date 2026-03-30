/// convert_glove.cpp — Convert GloVe text file to binary fvecs format.
/// Optionally pads dimension to next power of 2 (for TurboQuant).
/// Also splits off query vectors and computes brute-force ground truth.
///
/// Usage: convert_glove <input.txt> <output_dir> [pad_dim] [num_queries]
///
/// Output files:
///   <output_dir>/base.fvecs   — base vectors (N - num_queries)
///   <output_dir>/query.fvecs  — query vectors (num_queries)
///   <output_dir>/gt.ivecs     — ground truth top-100 for each query

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <queue>
#include <chrono>

struct FvecsWriter {
    FILE* f;
    int dim;
    FvecsWriter(const std::string& path, int d) : dim(d) {
        f = fopen(path.c_str(), "wb");
        if (!f) { fprintf(stderr, "Cannot open %s\n", path.c_str()); exit(1); }
    }
    ~FvecsWriter() { if (f) fclose(f); }
    void write(const float* vec) {
        fwrite(&dim, sizeof(int), 1, f);
        fwrite(vec, sizeof(float), dim, f);
    }
};

struct IvecsWriter {
    FILE* f;
    int dim;
    IvecsWriter(const std::string& path, int d) : dim(d) {
        f = fopen(path.c_str(), "wb");
        if (!f) { fprintf(stderr, "Cannot open %s\n", path.c_str()); exit(1); }
    }
    ~IvecsWriter() { if (f) fclose(f); }
    void write(const int* vec) {
        fwrite(&dim, sizeof(int), 1, f);
        fwrite(vec, sizeof(int), dim, f);
    }
};

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <glove.txt> <output_dir> [pad_dim] [num_queries]\n",
                argv[0]);
        return 1;
    }

    const char* input_path = argv[1];
    const char* output_dir = argv[2];
    int pad_dim = (argc > 3) ? atoi(argv[3]) : 0;
    int num_queries = (argc > 4) ? atoi(argv[4]) : 1000;
    int gt_k = 100;

    // Read all vectors
    printf("Reading %s...\n", input_path);
    std::ifstream fin(input_path);
    if (!fin.good()) { fprintf(stderr, "Cannot open %s\n", input_path); return 1; }

    std::vector<std::vector<float>> vectors;
    std::string line;
    int raw_dim = 0;

    while (std::getline(fin, line)) {
        std::istringstream iss(line);
        std::string word;
        iss >> word;  // skip word

        std::vector<float> vec;
        float val;
        while (iss >> val) vec.push_back(val);

        if (raw_dim == 0) {
            raw_dim = static_cast<int>(vec.size());
            printf("  Raw dimension: %d\n", raw_dim);
        }

        // Pad if requested
        if (pad_dim > 0 && pad_dim > raw_dim) {
            vec.resize(pad_dim, 0.0f);
        }

        vectors.push_back(std::move(vec));
    }
    fin.close();

    int final_dim = pad_dim > 0 ? pad_dim : raw_dim;
    printf("  Loaded %zu vectors, final dim=%d\n", vectors.size(), final_dim);

    // Shuffle deterministically for train/test split
    // Use last num_queries as queries (GloVe is already shuffled by frequency)
    size_t N = vectors.size();
    if (static_cast<size_t>(num_queries) >= N) {
        fprintf(stderr, "num_queries >= N\n");
        return 1;
    }
    size_t base_n = N - num_queries;

    // Write base vectors
    std::string base_path = std::string(output_dir) + "/base.fvecs";
    printf("Writing %zu base vectors to %s\n", base_n, base_path.c_str());
    {
        FvecsWriter w(base_path, final_dim);
        for (size_t i = 0; i < base_n; ++i)
            w.write(vectors[i].data());
    }

    // Write query vectors
    std::string query_path = std::string(output_dir) + "/query.fvecs";
    printf("Writing %d query vectors to %s\n", num_queries, query_path.c_str());
    {
        FvecsWriter w(query_path, final_dim);
        for (size_t i = base_n; i < N; ++i)
            w.write(vectors[i].data());
    }

    // Compute brute-force ground truth (top-100 NN for each query)
    printf("Computing ground truth (top-%d, %d queries x %zu base)...\n",
           gt_k, num_queries, base_n);
    auto t0 = std::chrono::high_resolution_clock::now();

    std::string gt_path = std::string(output_dir) + "/gt.ivecs";
    {
        IvecsWriter w(gt_path, gt_k);
        for (int q = 0; q < num_queries; ++q) {
            const float* query = vectors[base_n + q].data();

            // Max-heap of (dist, id), size K
            std::priority_queue<std::pair<float, int>> topk;
            for (size_t i = 0; i < base_n; ++i) {
                float dist = 0.0f;
                for (int d = 0; d < final_dim; ++d) {
                    float diff = query[d] - vectors[i][d];
                    dist += diff * diff;
                }
                if (static_cast<int>(topk.size()) < gt_k) {
                    topk.push({dist, static_cast<int>(i)});
                } else if (dist < topk.top().first) {
                    topk.pop();
                    topk.push({dist, static_cast<int>(i)});
                }
            }

            // Extract sorted by distance (ascending)
            std::vector<std::pair<float, int>> sorted;
            while (!topk.empty()) {
                sorted.push_back(topk.top());
                topk.pop();
            }
            std::reverse(sorted.begin(), sorted.end());

            std::vector<int> ids(gt_k);
            for (int j = 0; j < gt_k; ++j)
                ids[j] = sorted[j].second;
            w.write(ids.data());

            if ((q + 1) % 100 == 0)
                printf("  %d/%d queries done\n", q + 1, num_queries);
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double gt_sec = std::chrono::duration<double>(t1 - t0).count();
    printf("Ground truth computed in %.1f s\n", gt_sec);

    printf("\nDone! Files:\n");
    printf("  %s (%zu vectors, dim=%d)\n", base_path.c_str(), base_n, final_dim);
    printf("  %s (%d vectors, dim=%d)\n", query_path.c_str(), num_queries, final_dim);
    printf("  %s (%d queries, top-%d)\n", gt_path.c_str(), num_queries, gt_k);

    return 0;
}
