/// turbo_quant_test.cpp — Correctness tests for TurboQuant Algorithm 2
/// Synthetic face-embedding-like data: d=128, L2-normalized vectors
///
/// Tests:
///   1. WHT orthogonality (energy preservation)
///   2. SQ distortion (MSE in rotated space)
///   3. Unbiasedness: E[⟨y, x̃⟩] = ⟨y, x⟩
///   4. Lossless identity: MSE + exact residual = exact IP
///   5. Asymmetric IP Pearson correlation
///   6. Recall@K — L2 ranking preservation
///   +  Memory footprint report (informational, no pass/fail)

#include "hnswlib/space_turbo_quant.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

#ifdef __APPLE__
#include <mach/mach.h>
#endif

using namespace hnswlib::turboquant;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static float dot(const float *a, const float *b, size_t d) {
  float s = 0.0f;
  for (size_t i = 0; i < d; ++i)
    s += a[i] * b[i];
  return s;
}

static float l2_dist(const float *a, const float *b, size_t d) {
  float s = 0.0f;
  for (size_t i = 0; i < d; ++i) {
    float diff = a[i] - b[i];
    s += diff * diff;
  }
  return s;
}

static void normalize(float *v, size_t d) {
  float n = std::sqrt(dot(v, v, d));
  if (n > 1e-10f) {
    for (size_t i = 0; i < d; ++i)
      v[i] /= n;
  }
}

/// Generates clustered unit vectors simulating face descriptors.
static std::vector<std::vector<float>>
generateFaceEmbeddings(size_t n, size_t d, size_t num_clusters = 50,
                       uint64_t seed = 12345) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> gauss(0.0f, 1.0f);

  std::vector<std::vector<float>> centers(num_clusters, std::vector<float>(d));
  for (auto &c : centers) {
    for (auto &v : c)
      v = gauss(rng);
    normalize(c.data(), d);
  }

  std::uniform_int_distribution<size_t> cluster_dist(0, num_clusters - 1);
  std::vector<std::vector<float>> embeddings(n, std::vector<float>(d));
  for (size_t i = 0; i < n; ++i) {
    size_t cluster = cluster_dist(rng);
    for (size_t j = 0; j < d; ++j) {
      embeddings[i][j] = centers[cluster][j] + gauss(rng) * 0.3f;
    }
    normalize(embeddings[i].data(), d);
  }
  return embeddings;
}

// ---------------------------------------------------------------------------
// Test 1: WHT energy preservation
// ---------------------------------------------------------------------------
bool test_wht_energy() {
  std::cout << "=== Test 1: WHT energy preservation ===" << std::endl;

  constexpr size_t D = 128;
  std::mt19937_64 rng(42);
  std::normal_distribution<float> gauss(0.0f, 1.0f);

  uint64_t rot_seed = 42;
  auto rot_signs = generateSigns(D, rot_seed);
  float max_err = 0.0f;
  for (int trial = 0; trial < 100; ++trial) {
    std::vector<float> v(D);
    for (auto &x : v)
      x = gauss(rng);
    float energy_before = dot(v.data(), v.data(), D);

    std::vector<float> rotated(v);
    randomizedHadamard(rotated.data(), rot_signs.data(), D);
    float energy_after = dot(rotated.data(), rotated.data(), D);

    float rel_err = std::abs(energy_after - energy_before) / energy_before;
    max_err = std::max(max_err, rel_err);
  }

  constexpr float THRESHOLD = 1e-5f;
  bool pass = max_err < THRESHOLD;
  std::cout << "  Max relative energy error: " << std::scientific << max_err;
  if (!pass) {
    std::cout << "  FAIL (expected < " << THRESHOLD << ")";
  } else {
    std::cout << "  PASS";
  }
  std::cout << std::endl;
  return pass;
}

// ---------------------------------------------------------------------------
// Test 2: SQ distortion (MSE stage only, b-1 bits)
// ---------------------------------------------------------------------------
bool test_sq_distortion(int bits_per_coord = 4) {
  int mse_bits = bits_per_coord - 1;
  std::cout << "=== Test 2: SQ distortion (" << mse_bits
            << "-bit MSE stage, b=" << bits_per_coord << ") ===" << std::endl;

  constexpr size_t D = 128;
  constexpr size_t N = 1000;

  auto embeddings = generateFaceEmbeddings(N, D);
  uint64_t rot_seed = 42;
  uint64_t qjl_seed = 137;
  auto rot_signs = generateSigns(D, rot_seed);
  TurboQuantSpace space(D, bits_per_coord, rot_seed, qjl_seed);

  std::vector<char> buf(space.codeSizeBytes());

  double total_mse = 0.0;
  for (size_t i = 0; i < N; ++i) {
    space.encodeVector(embeddings[i].data(), buf.data());
    TurboQuantCode code(buf.data(), D, bits_per_coord);

    // Reconstruct MSE part in rotated space
    std::vector<float> rotated(embeddings[i]);
    float inv_norm = 1.0f / code.norm();
    for (size_t j = 0; j < D; ++j)
      rotated[j] *= inv_norm;
    randomizedHadamard(rotated.data(), rot_signs.data(), D);

    // Dequantize: centroid[sq_idx] * sigma
    const float *centroids = space.centroids();
    float sigma = code.sigma();
    float mse = 0.0f;
    for (size_t j = 0; j < D; ++j) {
      float recon_j = centroids[code.sqIndex(j, bits_per_coord)] * sigma;
      float diff = rotated[j] - recon_j;
      mse += diff * diff;
    }
    mse /= static_cast<float>(D);
    total_mse += mse;
  }
  double avg_mse = total_mse / N;

  // Threshold scales with bit depth: more bits → lower MSE
  double threshold = (mse_bits <= 3) ? 0.05 : (mse_bits <= 4) ? 0.01 : 0.001;
  bool pass = avg_mse < threshold;
  std::cout << "  Average MSE: " << std::fixed << std::setprecision(6)
            << avg_mse;
  if (!pass) {
    std::cout << "  FAIL (expected < " << threshold << ")";
  } else {
    std::cout << "  PASS";
  }
  std::cout << std::endl;
  return pass;
}

// ---------------------------------------------------------------------------
// Test 3: Unbiasedness — E[⟨y, x̃⟩] = ⟨y, x⟩
// Uses multiple random seeds to average over QJL randomness
// ---------------------------------------------------------------------------
bool test_unbiasedness(int bits_per_coord = 4) {
  std::cout << "=== Test 3: Unbiasedness of IP estimator (b=" << bits_per_coord
            << ") ===" << std::endl;

  constexpr size_t D = 128;
  constexpr size_t NUM_PAIRS = 50;
  constexpr size_t NUM_SEEDS = 500;

  auto embeddings = generateFaceEmbeddings(NUM_PAIRS * 2, D);

  double total_bias = 0.0;
  double max_bias = 0.0;

  for (size_t p = 0; p < NUM_PAIRS; ++p) {
    const float *x = embeddings[2 * p].data();
    const float *y = embeddings[2 * p + 1].data();

    float exact_ip = dot(y, x, D);

    // Average over multiple QJL seeds to estimate E[⟨y, x̃⟩]
    // Use distSearch to compute L2, then recover IP:
    //   L2 = ||y||^2 + ||x||^2 - 2*IP  =>  IP = (||y||^2 + ||x||^2 - L2) / 2
    float y_norm_sq = dot(y, y, D);
    float x_norm_sq = dot(x, x, D);

    double avg_approx = 0.0;
    for (size_t s = 0; s < NUM_SEEDS; ++s) {
      TurboQuantSpace space(D, bits_per_coord, /*rot_seed=*/42,
                            /*qjl_seed=*/1000 + s);
      std::vector<char> buf(space.codeSizeBytes());
      space.encodeVector(x, buf.data());
      auto pq = space.prepareQuery(y);
      auto dist_func = space.get_search_dist_func();
      float l2 = dist_func(&pq, buf.data(), &space);
      float approx_ip = (y_norm_sq + x_norm_sq - l2) / 2.0f;
      avg_approx += approx_ip;
    }
    avg_approx /= NUM_SEEDS;

    double bias = std::abs(avg_approx - exact_ip);
    total_bias += bias;
    max_bias = std::max(max_bias, bias);
  }

  double avg_bias = total_bias / NUM_PAIRS;

  // Unbiased: average bias should be small relative to typical IP magnitude
  constexpr double THRESHOLD = 0.005;
  bool pass = avg_bias < THRESHOLD;
  std::cout << "  Average |bias| over " << NUM_SEEDS << " seeds: " << std::fixed
            << std::setprecision(6) << avg_bias;
  if (!pass) {
    std::cout << "  FAIL (expected < " << THRESHOLD << ")";
  } else {
    std::cout << "  PASS";
  }
  std::cout << std::endl;
  std::cout << "  Max |bias|: " << std::setprecision(6) << max_bias
            << std::endl;
  return pass;
}

// ---------------------------------------------------------------------------
// Test 3b: Lossless identity — MSE_IP + exact_residual_IP = exact_IP
// Deterministic check: no QJL randomness involved
// ---------------------------------------------------------------------------
bool test_lossless_identity(int bits_per_coord = 4) {
  std::cout << "=== Test 3b: MSE + residual = exact (lossless identity, b="
            << bits_per_coord << ") ===" << std::endl;

  constexpr size_t D = 512;
  constexpr size_t N = 20000;

  auto embeddings = generateFaceEmbeddings(N, D);
  uint64_t rot_seed = 42;
  uint64_t qjl_seed = 137;
  auto rot_signs = generateSigns(D, rot_seed);
  TurboQuantSpace space(D, bits_per_coord, rot_seed, qjl_seed);
  const float *centroids = space.centroids();

  std::vector<char> buf(space.codeSizeBytes());
  double max_err = 0.0;

  for (size_t i = 0; i < N; ++i) {
    for (size_t j = i + 1; j < std::min(N, i + 5); ++j) {
      const float *x = embeddings[i].data();
      const float *y = embeddings[j].data();
      float exact_ip = dot(x, y, D);

      space.encodeVector(x, buf.data());
      TurboQuantCode code(buf.data(), D, bits_per_coord);

      // Compute MSE IP in rotated space
      float x_norm = code.norm();
      float sigma = code.sigma();
      float y_norm_sq = 0.0f;
      for (size_t k = 0; k < D; ++k)
        y_norm_sq += y[k] * y[k];
      float y_norm = std::sqrt(y_norm_sq);

      std::vector<float> y_rot(D);
      float y_inv = 1.0f / y_norm;
      for (size_t k = 0; k < D; ++k)
        y_rot[k] = y[k] * y_inv;
      randomizedHadamard(y_rot.data(), rot_signs.data(), D);

      // Rotate x
      std::vector<float> x_rot(D);
      float x_inv = 1.0f / x_norm;
      for (size_t k = 0; k < D; ++k)
        x_rot[k] = x[k] * x_inv;
      randomizedHadamard(x_rot.data(), rot_signs.data(), D);

      float ip_mse = 0.0f;
      float ip_res = 0.0f;
      for (size_t k = 0; k < D; ++k) {
        float cv_k = centroids[code.sqIndex(k, bits_per_coord)] * sigma;
        ip_mse += y_rot[k] * cv_k;
        ip_res += y_rot[k] * (x_rot[k] - cv_k);
      }
      float reconstructed = (ip_mse + ip_res) * x_norm * y_norm;
      double err = std::abs(reconstructed - exact_ip);
      max_err = std::max(max_err, err);
    }
  }

  // Should be exact up to float32 precision
  constexpr double THRESHOLD = 1e-4;
  bool pass = max_err < THRESHOLD;
  std::cout << "  Max |MSE+residual - exact|: " << std::scientific << max_err;
  if (!pass) {
    std::cout << "  FAIL (expected < " << THRESHOLD << ")";
  } else {
    std::cout << "  PASS";
  }
  std::cout << std::endl;
  return pass;
}

// ---------------------------------------------------------------------------
// Test 4: IP correlation (single seed, practical scenario)
// ---------------------------------------------------------------------------
bool test_ip_correlation(int bits_per_coord = 4) {
  std::cout << "=== Test 4: Asymmetric IP correlation (b=" << bits_per_coord
            << ") ===" << std::endl;

  constexpr size_t D = 128;
  constexpr size_t N = 500;
  constexpr size_t NUM_QUERIES = 20;

  auto embeddings = generateFaceEmbeddings(N, D);
  uint64_t rot_seed = 42;
  uint64_t qjl_seed = 137;
  TurboQuantSpace space(D, bits_per_coord, rot_seed, qjl_seed);

  size_t code_size = space.codeSizeBytes();
  std::vector<std::vector<char>> codes(N, std::vector<char>(code_size));
  for (size_t i = 0; i < N; ++i) {
    space.encodeVector(embeddings[i].data(), codes[i].data());
  }

  double sum_corr = 0.0;
  std::mt19937_64 rng(999);
  std::uniform_int_distribution<size_t> query_dist(0, N - 1);
  auto dist_func = space.get_search_dist_func();

  for (size_t q = 0; q < NUM_QUERIES; ++q) {
    size_t qi = query_dist(rng);
    const float *query = embeddings[qi].data();
    auto pq = space.prepareQuery(query);
    float q_norm_sq = dot(query, query, D);

    std::vector<float> exact_ips(N), approx_ips(N);
    for (size_t i = 0; i < N; ++i) {
      exact_ips[i] = dot(query, embeddings[i].data(), D);
      // Recover IP from L2: IP = (||q||^2 + ||x||^2 - L2) / 2
      float x_norm_sq = dot(embeddings[i].data(), embeddings[i].data(), D);
      float l2 = dist_func(&pq, codes[i].data(), &space);
      approx_ips[i] = (q_norm_sq + x_norm_sq - l2) / 2.0f;
    }

    // Pearson correlation
    float mean_e = 0, mean_a = 0;
    for (size_t i = 0; i < N; ++i) {
      mean_e += exact_ips[i];
      mean_a += approx_ips[i];
    }
    mean_e /= N;
    mean_a /= N;

    float cov = 0, var_e = 0, var_a = 0;
    for (size_t i = 0; i < N; ++i) {
      float de = exact_ips[i] - mean_e;
      float da = approx_ips[i] - mean_a;
      cov += de * da;
      var_e += de * de;
      var_a += da * da;
    }
    float corr = cov / (std::sqrt(var_e) * std::sqrt(var_a) + 1e-10f);
    sum_corr += corr;
  }

  double avg_corr = sum_corr / NUM_QUERIES;

  constexpr double THRESHOLD = 0.85;
  bool pass = avg_corr > THRESHOLD;
  std::cout << "  Average Pearson correlation: " << std::fixed
            << std::setprecision(4) << avg_corr;
  if (!pass) {
    std::cout << "  FAIL (expected > " << THRESHOLD << ")";
  } else {
    std::cout << "  PASS";
  }
  std::cout << std::endl;
  return pass;
}

// ---------------------------------------------------------------------------
// Test 5: Recall@K
// ---------------------------------------------------------------------------
bool test_recall_at_k(int bits_per_coord = 4) {
  std::cout << "=== Test 5: Recall@K (L2 ranking, b=" << bits_per_coord
            << ") ===" << std::endl;

  constexpr size_t D = 128;
  constexpr size_t N = 2000;
  constexpr size_t NUM_QUERIES = 50;
  constexpr size_t K = 10;

  auto embeddings = generateFaceEmbeddings(N, D, 100);
  uint64_t rot_seed = 42;
  uint64_t qjl_seed = 137;
  TurboQuantSpace space(D, bits_per_coord, rot_seed, qjl_seed);

  size_t code_size = space.codeSizeBytes();
  std::vector<std::vector<char>> codes(N, std::vector<char>(code_size));
  for (size_t i = 0; i < N; ++i) {
    space.encodeVector(embeddings[i].data(), codes[i].data());
  }

  auto dist_func = space.get_search_dist_func();

  std::mt19937_64 rng(777);
  std::uniform_int_distribution<size_t> query_dist(0, N - 1);

  size_t total_hits = 0;

  for (size_t q = 0; q < NUM_QUERIES; ++q) {
    size_t qi = query_dist(rng);
    const float *query = embeddings[qi].data();
    auto pq = space.prepareQuery(query);

    // Exact top-K
    std::vector<std::pair<float, size_t>> exact_dists(N);
    for (size_t i = 0; i < N; ++i) {
      exact_dists[i] = {l2_dist(query, embeddings[i].data(), D), i};
    }
    std::partial_sort(exact_dists.begin(), exact_dists.begin() + K,
                      exact_dists.end());

    // Approximate top-K
    std::vector<std::pair<float, size_t>> approx_dists(N);
    for (size_t i = 0; i < N; ++i) {
      approx_dists[i] = {dist_func(&pq, codes[i].data(), &space), i};
    }
    std::partial_sort(approx_dists.begin(), approx_dists.begin() + K,
                      approx_dists.end());

    // Count hits
    for (size_t i = 0; i < K; ++i) {
      for (size_t j = 0; j < K; ++j) {
        if (approx_dists[i].second == exact_dists[j].second) {
          ++total_hits;
          break;
        }
      }
    }
  }

  float recall =
      static_cast<float>(total_hits) / static_cast<float>(NUM_QUERIES * K);

  constexpr float THRESHOLD = 0.60f;
  bool pass = recall > THRESHOLD;
  std::cout << "  Recall@" << K << ": " << std::fixed << std::setprecision(2)
            << (recall * 100.0f) << "%";
  if (!pass) {
    std::cout << "  FAIL (expected > " << (THRESHOLD * 100.0f) << "%)";
  } else {
    std::cout << "  PASS";
  }
  std::cout << std::endl;
  return pass;
}

// ---------------------------------------------------------------------------
// Memory footprint report (informational, no pass/fail)
// ---------------------------------------------------------------------------
void test_memory_footprint() {
  std::cout << "=== Memory footprint ===" << std::endl;

  constexpr size_t D = 128;
  constexpr int BITS = 8;

  size_t raw_bytes = D * sizeof(float);
  size_t tq_bytes = TurboQuantCode::codeSizeBytes(D, BITS);
  int mse_bits = BITS - 1;

  std::cout << "  Dimension:       " << D << std::endl;
  std::cout << "  Bit budget:      " << BITS
            << " (MSE: " << mse_bits << " + QJL: 1)" << std::endl;
  std::cout << "  Raw float32:     " << raw_bytes << " bytes" << std::endl;
  std::cout << "  TurboQuant code: " << tq_bytes << " bytes" << std::endl;
  std::cout << "  Effective bpc:   " << std::fixed << std::setprecision(1)
            << static_cast<float>(tq_bytes * 8) / D << " bits/coord" << std::endl;
  std::cout << "  Compression:     " << std::setprecision(1)
            << static_cast<float>(raw_bytes) / tq_bytes << "x" << std::endl;
  std::cout << "  Per 1M vectors:  " << std::setprecision(0)
            << (raw_bytes * 1e6 / (1024.0 * 1024.0)) << " MB -> "
            << (tq_bytes * 1e6 / (1024.0 * 1024.0)) << " MB" << std::endl;

  // Tight bit-packed estimate (if SQ indices were sub-byte packed)
  size_t tight_sq = (D * mse_bits + 7) / 8;
  size_t tight_qjl = (D + 7) / 8;
  size_t tight_total = tight_sq + tight_qjl + sizeof(float) * 3;
  std::cout << "\n  Tight bit-packed estimate: " << tight_total << " bytes ("
            << std::setprecision(1)
            << static_cast<float>(raw_bytes) / tight_total << "x)" << std::endl;
}

// TurboQuantSpace is now in hnswlib/turbo_quant_space.h

// ---------------------------------------------------------------------------
// Test 6: Serialization round-trip
// ---------------------------------------------------------------------------
bool test_encode_determinism(int bits_per_coord = 4) {
  std::cout << "=== Test 6: Encode determinism (b=" << bits_per_coord
            << ") ===" << std::endl;

  constexpr size_t D = 128;
  constexpr size_t N = 100;

  auto embeddings = generateFaceEmbeddings(N, D);
  uint64_t rot_seed = 42;
  uint64_t qjl_seed = 137;
  TurboQuantSpace space(D, bits_per_coord, rot_seed, qjl_seed);

  size_t code_size = space.codeSizeBytes();
  bool all_match = true;

  for (size_t i = 0; i < N; ++i) {
    std::vector<char> buf1(code_size), buf2(code_size);
    space.encodeVector(embeddings[i].data(), buf1.data());
    space.encodeVector(embeddings[i].data(), buf2.data());

    if (std::memcmp(buf1.data(), buf2.data(), code_size) != 0) {
      all_match = false;
      break;
    }
  }

  // Also verify two spaces with same seeds produce identical codes
  TurboQuantSpace space2(D, bits_per_coord, rot_seed, qjl_seed);
  for (size_t i = 0; i < N && all_match; ++i) {
    std::vector<char> buf1(code_size), buf2(code_size);
    space.encodeVector(embeddings[i].data(), buf1.data());
    space2.encodeVector(embeddings[i].data(), buf2.data());

    if (std::memcmp(buf1.data(), buf2.data(), code_size) != 0) {
      all_match = false;
      break;
    }
  }

  std::cout << "  Deterministic encoding: " << (all_match ? "yes" : "NO");
  if (!all_match) {
    std::cout << "  FAIL";
  } else {
    std::cout << "  PASS";
  }
  std::cout << std::endl;
  return all_match;
}

// ---------------------------------------------------------------------------
// Test 7: HNSW integration — TurboQuantSpace + HierarchicalNSW
// ---------------------------------------------------------------------------
bool test_hnsw_integration(int bits_per_coord = 4) {
  std::cout << "=== Test 7: HNSW integration (TurboQuantSpace, b="
            << bits_per_coord << ") ===" << std::endl;

  constexpr size_t D = 128;
  constexpr size_t N = 500;
  constexpr size_t NUM_QUERIES = 20;
  constexpr size_t K = 10;
  constexpr size_t M = 16;
  constexpr size_t EF_CONSTRUCTION = 200;

  auto embeddings = generateFaceEmbeddings(N, D, 100);

  uint64_t rot_seed = 42;
  uint64_t qjl_seed = 137;
  TurboQuantSpace space(D, bits_per_coord, rot_seed, qjl_seed);

  hnswlib::HierarchicalNSW<float> hnsw(&space, N, M, EF_CONSTRUCTION);

  // Encode and insert
  std::vector<char> buf(space.codeSizeBytes());
  for (size_t i = 0; i < N; ++i) {
    space.encodeVector(embeddings[i].data(), buf.data());
    hnsw.addPoint(buf.data(), i);
  }

  hnsw.setEf(64);
  space.setSearchMode(hnsw);

  // Search and measure recall
  std::mt19937_64 rng(555);
  std::uniform_int_distribution<size_t> query_dist(0, N - 1);
  size_t total_hits = 0;

  for (size_t q = 0; q < NUM_QUERIES; ++q) {
    size_t qi = query_dist(rng);
    const float *query = embeddings[qi].data();

    // Exact top-K by brute-force L2
    std::vector<std::pair<float, size_t>> exact_dists(N);
    for (size_t i = 0; i < N; ++i) {
      exact_dists[i] = {l2_dist(query, embeddings[i].data(), D), i};
    }
    std::partial_sort(exact_dists.begin(), exact_dists.begin() + K,
                      exact_dists.end());

    // HNSW search with prepared query
    auto pq = space.prepareQuery(query);
    auto result = hnsw.searchKnn(&pq, K);

    // Collect HNSW results
    std::vector<size_t> hnsw_ids;
    while (!result.empty()) {
      hnsw_ids.push_back(result.top().second);
      result.pop();
    }

    // Count hits
    for (size_t i = 0; i < K; ++i) {
      for (size_t j = 0; j < hnsw_ids.size(); ++j) {
        if (hnsw_ids[j] == exact_dists[i].second) {
          ++total_hits;
          break;
        }
      }
    }
  }

  float recall =
      static_cast<float>(total_hits) / static_cast<float>(NUM_QUERIES * K);

  // Memory comparison
  size_t raw_bytes = D * sizeof(float);
  size_t tq_bytes = space.codeSizeBytes();
  std::cout << "  Memory: " << raw_bytes << " B/vec (raw float) vs " << tq_bytes
            << " B/vec (TurboQuant) = " << std::fixed << std::setprecision(1)
            << static_cast<float>(raw_bytes) / tq_bytes << "x compression"
            << std::endl;

  constexpr float THRESHOLD = 0.45f;
  bool pass = recall > THRESHOLD;
  std::cout << "  Recall@" << K << ": " << std::fixed << std::setprecision(2)
            << (recall * 100.0f) << "%";
  if (!pass) {
    std::cout << "  FAIL (expected > " << (THRESHOLD * 100.0f) << "%)";
  } else {
    std::cout << "  PASS";
  }
  std::cout << std::endl;
  return pass;
}

// ---------------------------------------------------------------------------
// Test 8: Save/load index — verify persistence preserves search quality
// ---------------------------------------------------------------------------
bool test_save_load_index(int bits_per_coord = 4) {
  std::cout << "=== Test 8: Save/load index (b=" << bits_per_coord
            << ") ===" << std::endl;

  constexpr size_t D = 128;
  constexpr size_t N = 200;
  constexpr size_t NUM_QUERIES = 10;
  constexpr size_t K = 10;
  constexpr size_t M = 16;
  constexpr size_t EF_CONSTRUCTION = 200;

  auto embeddings = generateFaceEmbeddings(N, D, 50);

  uint64_t rot_seed = 42;
  uint64_t qjl_seed = 137;
  TurboQuantSpace space(D, bits_per_coord, rot_seed, qjl_seed);

  // Build index
  hnswlib::HierarchicalNSW<float> hnsw(&space, N, M, EF_CONSTRUCTION);
  std::vector<char> buf(space.codeSizeBytes());
  for (size_t i = 0; i < N; ++i) {
    space.encodeVector(embeddings[i].data(), buf.data());
    hnsw.addPoint(buf.data(), i);
  }
  hnsw.setEf(64);
  space.setSearchMode(hnsw);

  // Search before save — collect results for comparison
  std::mt19937_64 rng(333);
  std::uniform_int_distribution<size_t> query_dist(0, N - 1);
  std::vector<size_t> query_indices(NUM_QUERIES);
  for (size_t q = 0; q < NUM_QUERIES; ++q)
    query_indices[q] = query_dist(rng);

  std::vector<std::vector<size_t>> results_before(NUM_QUERIES);
  for (size_t q = 0; q < NUM_QUERIES; ++q) {
    const float *query = embeddings[query_indices[q]].data();
    auto pq = space.prepareQuery(query);
    auto result = hnsw.searchKnn(&pq, K);
    while (!result.empty()) {
      results_before[q].push_back(result.top().second);
      result.pop();
    }
    std::sort(results_before[q].begin(), results_before[q].end());
  }

  // Save to file
  const std::string filename = "test_tq_index.bin";
  hnsw.saveIndex(filename);

  // Load into a new index (same space, fresh HierarchicalNSW)
  hnswlib::HierarchicalNSW<float> hnsw2(&space, filename);
  hnsw2.setEf(64);
  space.setSearchMode(hnsw2);

  // Search after load — results must be identical
  bool all_match = true;
  size_t total_hits_vs_exact = 0;

  for (size_t q = 0; q < NUM_QUERIES; ++q) {
    const float *query = embeddings[query_indices[q]].data();

    auto pq = space.prepareQuery(query);
    auto result = hnsw2.searchKnn(&pq, K);

    std::vector<size_t> results_after;
    while (!result.empty()) {
      results_after.push_back(result.top().second);
      result.pop();
    }
    std::sort(results_after.begin(), results_after.end());

    if (results_after != results_before[q]) {
      all_match = false;
    }

    // Also check recall vs exact brute-force
    std::vector<std::pair<float, size_t>> exact_dists(N);
    for (size_t i = 0; i < N; ++i) {
      exact_dists[i] = {l2_dist(query, embeddings[i].data(), D), i};
    }
    std::partial_sort(exact_dists.begin(), exact_dists.begin() + K,
                      exact_dists.end());
    for (size_t i = 0; i < K; ++i) {
      for (size_t j = 0; j < results_after.size(); ++j) {
        if (results_after[j] == exact_dists[i].second) {
          ++total_hits_vs_exact;
          break;
        }
      }
    }
  }

  float recall = static_cast<float>(total_hits_vs_exact) /
                 static_cast<float>(NUM_QUERIES * K);

  // Clean up
  std::remove(filename.c_str());

  bool pass = all_match && (recall > 0.45f);
  std::cout << "  Results match before/after: " << (all_match ? "yes" : "NO")
            << std::endl;
  std::cout << "  Recall@" << K << " after load: " << std::fixed
            << std::setprecision(2) << (recall * 100.0f) << "%";
  if (!pass) {
    std::cout << "  FAIL";
  } else {
    std::cout << "  PASS";
  }
  std::cout << std::endl;
  return pass;
}

// ---------------------------------------------------------------------------
// Large-scale comparison: L2Space vs TurboQuantSpace
// 1M vectors, dim=1024.  Reports memory, file size, build time, search time,
// recall@10.
// ---------------------------------------------------------------------------

/// Returns resident set size (RSS) in bytes. macOS only; returns 0 elsewhere.
static size_t getCurrentRSS() {
#ifdef __APPLE__
  struct mach_task_basic_info info;
  mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
  if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info,
                &count) == KERN_SUCCESS) {
    return info.resident_size;
  }
#endif
  return 0;
}

static size_t fileSize(const std::string &path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  return f.good() ? static_cast<size_t>(f.tellg()) : 0;
}

/// Generates a single random unit vector (no storage of the full dataset).
static void generateRandomVector(float *out, size_t d, std::mt19937_64 &rng) {
  std::normal_distribution<float> gauss(0.0f, 1.0f);
  for (size_t i = 0; i < d; ++i)
    out[i] = gauss(rng);
  normalize(out, d);
}

void test_large_scale_comparison() {
  std::cout << "=== Large-scale comparison: L2Space vs TurboQuantSpace ==="
            << std::endl;

  constexpr size_t D = 128;
  constexpr size_t N = 10000;
  constexpr size_t NUM_QUERIES = 10;
  constexpr size_t K = 10;
  constexpr size_t M = 16;
  constexpr size_t EF_CONSTRUCTION = 100;
  constexpr size_t EF_SEARCH = 64;

  const std::string file_l2 = "bench_l2.bin";
  const std::string file_tq = "bench_tq.bin";

  // Pre-generate query vectors (small — 100 × 1024 floats = 400 KB)
  std::mt19937_64 qrng(99999);
  std::vector<std::vector<float>> queries(NUM_QUERIES, std::vector<float>(D));
  for (size_t q = 0; q < NUM_QUERIES; ++q) {
    generateRandomVector(queries[q].data(), D, qrng);
  }

  // -----------------------------------------------------------------------
  // Part A: Standard L2Space HNSW
  // -----------------------------------------------------------------------
  std::cout << "\n  --- L2Space (raw float32) ---" << std::endl;
  {
    hnswlib::L2Space l2space(D);
    size_t rss_before = getCurrentRSS();

    auto t0 = std::chrono::high_resolution_clock::now();
    hnswlib::HierarchicalNSW<float> hnsw(&l2space, N, M, EF_CONSTRUCTION);

    std::mt19937_64 rng(42);
    std::vector<float> vec(D);
    for (size_t i = 0; i < N; ++i) {
      generateRandomVector(vec.data(), D, rng);
      hnsw.addPoint(vec.data(), i);
      if ((i + 1) % 10000 == 0) {
        std::cout << "    inserted " << (i + 1) / 1000 << "K..." << std::endl;
      }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double build_sec = std::chrono::duration<double>(t1 - t0).count();

    size_t rss_after = getCurrentRSS();
    size_t rss_delta = (rss_after > rss_before) ? (rss_after - rss_before) : 0;

    // Save
    hnsw.saveIndex(file_l2);
    size_t fsize = fileSize(file_l2);

    // Search
    hnsw.setEf(EF_SEARCH);
    t0 = std::chrono::high_resolution_clock::now();
    for (size_t q = 0; q < NUM_QUERIES; ++q) {
      auto result = hnsw.searchKnn(queries[q].data(), K);
      (void)result;
    }
    t1 = std::chrono::high_resolution_clock::now();
    double search_sec = std::chrono::duration<double>(t1 - t0).count();

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "    Build time:    " << build_sec << " s" << std::endl;
    std::cout << "    RSS delta:     " << (rss_delta / (1024.0 * 1024.0))
              << " MB" << std::endl;
    std::cout << "    data_size/vec: " << l2space.get_data_size() << " B"
              << std::endl;
    std::cout << "    Index file:    " << (fsize / (1024.0 * 1024.0)) << " MB"
              << std::endl;
    std::cout << "    Search time:   " << (search_sec / NUM_QUERIES * 1000.0)
              << " ms/query (" << NUM_QUERIES << " queries, ef=" << EF_SEARCH
              << ")" << std::endl;

    // Cleanup — release memory before TurboQuant build
    std::remove(file_l2.c_str());
  }

  // -----------------------------------------------------------------------
  // Part B: TurboQuantSpace HNSW
  // -----------------------------------------------------------------------
  std::cout << "\n  --- TurboQuantSpace (7-bit SQ + 1-bit QJL) ---"
            << std::endl;
  {
    uint64_t rot_seed = 42;
    uint64_t qjl_seed = 137;
    TurboQuantSpace tqspace(D, 8, rot_seed, qjl_seed);
    size_t rss_before = getCurrentRSS();

    auto t0 = std::chrono::high_resolution_clock::now();
    hnswlib::HierarchicalNSW<float> hnsw(&tqspace, N, M, EF_CONSTRUCTION);

    std::mt19937_64 rng(42); // same seed → same data
    std::vector<float> vec(D);
    std::vector<char> buf(tqspace.codeSizeBytes());
    for (size_t i = 0; i < N; ++i) {
      generateRandomVector(vec.data(), D, rng);
      tqspace.encodeVector(vec.data(), buf.data());
      hnsw.addPoint(buf.data(), i);
      if ((i + 1) % 10000 == 0) {
        std::cout << "    inserted " << (i + 1) / 1000 << "K..." << std::endl;
      }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double build_sec = std::chrono::duration<double>(t1 - t0).count();

    size_t rss_after = getCurrentRSS();
    size_t rss_delta = (rss_after > rss_before) ? (rss_after - rss_before) : 0;

    // Save
    hnsw.saveIndex(file_tq);
    size_t fsize = fileSize(file_tq);

    // Search
    hnsw.setEf(EF_SEARCH);
    tqspace.setSearchMode(hnsw);
    t0 = std::chrono::high_resolution_clock::now();
    for (size_t q = 0; q < NUM_QUERIES; ++q) {
      auto pq = tqspace.prepareQuery(queries[q].data());
      auto result = hnsw.searchKnn(&pq, K);
      (void)result;
    }
    t1 = std::chrono::high_resolution_clock::now();
    double search_sec = std::chrono::duration<double>(t1 - t0).count();

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "    Build time:    " << build_sec << " s" << std::endl;
    std::cout << "    RSS delta:     " << (rss_delta / (1024.0 * 1024.0))
              << " MB" << std::endl;
    std::cout << "    data_size/vec: " << tqspace.get_data_size() << " B"
              << std::endl;
    std::cout << "    Index file:    " << (fsize / (1024.0 * 1024.0)) << " MB"
              << std::endl;
    std::cout << "    Search time:   " << (search_sec / NUM_QUERIES * 1000.0)
              << " ms/query (" << NUM_QUERIES << " queries, ef=" << EF_SEARCH
              << ")" << std::endl;

    // Recall: brute-force exact L2 over 1M for each query
    // Regenerate data with the same seed
    std::cout << "    Computing recall (brute-force)..." << std::endl;
    std::mt19937_64 rng2(42);
    // Store all vectors for brute-force — but 1M×1024 = 4GB is too much.
    // Instead, compute exact distances on-the-fly per query.
    size_t total_hits = 0;
    for (size_t q = 0; q < NUM_QUERIES; ++q) {
      const float *query = queries[q].data();

      // HNSW result
      auto pq = tqspace.prepareQuery(query);
      auto result = hnsw.searchKnn(&pq, K);
      std::vector<size_t> hnsw_ids;
      while (!result.empty()) {
        hnsw_ids.push_back(result.top().second);
        result.pop();
      }

      // Exact top-K: scan all vectors, keep a partial top-K heap
      std::vector<std::pair<float, size_t>> topk;
      topk.reserve(K + 1);
      std::mt19937_64 rng_scan(42);
      std::vector<float> scan_vec(D);
      for (size_t i = 0; i < N; ++i) {
        generateRandomVector(scan_vec.data(), D, rng_scan);
        float dist = l2_dist(query, scan_vec.data(), D);
        if (topk.size() < K) {
          topk.push_back({dist, i});
          if (topk.size() == K) {
            std::make_heap(topk.begin(), topk.end());
          }
        } else if (dist < topk.front().first) {
          std::pop_heap(topk.begin(), topk.end());
          topk.back() = {dist, i};
          std::push_heap(topk.begin(), topk.end());
        }
      }

      for (size_t ti = 0; ti < topk.size(); ++ti) {
        for (size_t j = 0; j < hnsw_ids.size(); ++j) {
          if (hnsw_ids[j] == topk[ti].second) {
            ++total_hits;
            break;
          }
        }
      }
    }
    float recall =
        static_cast<float>(total_hits) / static_cast<float>(NUM_QUERIES * K);
    std::cout << "    Recall@" << K << ":     " << std::setprecision(2)
              << (recall * 100.0f) << "%" << std::endl;

    std::remove(file_tq.c_str());
  }
}

// ---------------------------------------------------------------------------
// Performance benchmark (informational, no pass/fail)
// ---------------------------------------------------------------------------
void test_benchmark() {
  std::cout << "=== Performance benchmark ===" << std::endl;

  constexpr size_t D = 128;
  constexpr size_t N = 1000;
  constexpr size_t NUM_QUERIES = 50;

  auto embeddings = generateFaceEmbeddings(N, D, 50);
  uint64_t rot_seed = 42;
  uint64_t qjl_seed = 137;
  TurboQuantSpace space(D, 8, rot_seed, qjl_seed);

  size_t code_size = space.codeSizeBytes();
  std::vector<std::vector<char>> codes(N, std::vector<char>(code_size));

  // 1. Encode throughput
  auto t0 = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < N; ++i) {
    space.encodeVector(embeddings[i].data(), codes[i].data());
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  double encode_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

  // 2. Distance computation throughput (prepared query)
  auto dist_func = space.get_search_dist_func();
  auto *dist_param = space.get_dist_func_param();
  size_t total_comps = NUM_QUERIES * N;

  volatile float sink = 0.0f;
  t0 = std::chrono::high_resolution_clock::now();
  for (size_t q = 0; q < NUM_QUERIES; ++q) {
    auto pq = space.prepareQuery(embeddings[q].data());
    for (size_t i = 0; i < N; ++i) {
      sink = dist_func(&pq, codes[i].data(), dist_param);
    }
  }
  t1 = std::chrono::high_resolution_clock::now();
  double dist_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
  (void)sink;

  // Report
  std::cout << std::fixed << std::setprecision(1);
  std::cout << "  encode:            " << std::setw(8) << (encode_us / N)
            << " us/vec  (" << static_cast<size_t>(N * 1e6 / encode_us)
            << " vec/s)" << std::endl;
  std::cout << "  distSearch:        " << std::setw(8)
            << (dist_us / total_comps * 1000.0) << " ns/cmp  ("
            << static_cast<size_t>(total_comps * 1e6 / dist_us) << " cmp/s)"
            << std::endl;
}

// ---------------------------------------------------------------------------
// Test: LUT correctness — verify LUT-based distSearch matches reference
// ---------------------------------------------------------------------------
bool test_lut_correctness(int bits_per_coord = 4) {
  // int mse_bits = bits_per_coord - 1;
  std::cout << "=== Test: LUT distSearch correctness (b=" << bits_per_coord
            << ") ===" << std::endl;

  constexpr size_t D = 128;
  constexpr size_t N = 200;
  constexpr size_t NUM_QUERIES = 10;
  constexpr float TOL =
      1e-3f; // float rounding from factoring sigma out of loop

  auto embeddings = generateFaceEmbeddings(N + NUM_QUERIES, D);

  TurboQuantSpace tq_space(D, bits_per_coord, 42, 137);
  size_t code_size = tq_space.codeSizeBytes();

  // Encode all vectors
  std::vector<std::vector<char>> codes(N, std::vector<char>(code_size));
  for (size_t i = 0; i < N; ++i) {
    tq_space.encodeVector(embeddings[i].data(), codes[i].data());
  }

  // Reference: compute distance using direct formula
  // ip_mse_ref = sigma * Σ_i q_rot[i] * centroids[sq_packed[i]]
  const float *centroids = tq_space.centroids();

  float max_rel_err = 0.0f;
  int mismatches = 0;

  for (size_t q = 0; q < NUM_QUERIES; ++q) {
    const float *query = embeddings[N + q].data();
    auto pq = tq_space.prepareQuery(query);

    // Helper: extract unit = (sq_idx << 1) | qjl_bit for coord d, honoring
    // the packed-nibble layout when b<=4.
    auto extractUnit = [&](const uint8_t *packed, size_t d) -> uint8_t {
      if (tq_space.packedNibbles()) {
        uint8_t byte = packed[d >> 1];
        return (d & 1) ? (byte >> 4) : (byte & 0x0F);
      }
      return packed[d];
    };

    for (size_t i = 0; i < N; ++i) {
      const char *buf = codes[i].data();
      const uint8_t *packed = reinterpret_cast<const uint8_t *>(buf);
      const float *meta = reinterpret_cast<const float *>(
          buf + tq_space.packedBytes());
      const float x_norm = meta[0];
      const float gamma = meta[1];
      const float sigma = meta[2];

      // Reference MSE term (direct compute)
      float ip_mse_ref = 0.0f;
      for (size_t d = 0; d < D; ++d)
        ip_mse_ref += pq.q_rot[d] * centroids[extractUnit(packed, d) >> 1] * sigma;

      // Verify full distance via TurboQuantSpace search dispatch
      float dist_dispatch = tq_space.get_search_dist_func()(&pq, buf,
                                                          tq_space.get_dist_func_param());

      // Reference full distance
      float dot_qjl = 0.0f;
      for (size_t d = 0; d < D; ++d) {
        float sign = (extractUnit(packed, d) & 1) ? 1.0f : -1.0f;
        dot_qjl += pq.s_q[d] * sign;
      }
      float scale = std::sqrt(static_cast<float>(M_PI) / 2.0f) /
                    std::sqrt(static_cast<float>(D));
      float correction = scale * gamma * dot_qjl;
      float ip_ref = (ip_mse_ref + correction) * x_norm * pq.q_norm;
      float dist_ref =
          std::max(0.0f, pq.q_norm_sq + x_norm * x_norm - 2.0f * ip_ref);

      float dist_err = std::abs(dist_dispatch - dist_ref);
      float dist_denom = std::max(dist_ref, 1e-10f);
      float rel_err = dist_err / dist_denom;
      if (rel_err > max_rel_err)
        max_rel_err = rel_err;
      if (rel_err > TOL)
        ++mismatches;
    }
  }

  bool pass = (mismatches == 0);
  std::cout << "  Max relative error (distance): " << std::scientific
            << max_rel_err << std::endl;
  std::cout << "  Mismatches: " << mismatches << " / " << (NUM_QUERIES * N)
            << std::endl;
  std::cout << "  " << (pass ? "PASS" : "FAIL") << std::endl;
  return pass;
}

// ---------------------------------------------------------------------------
// Test 9: TurboQuantIndex — build, save, load, search, searchRerank
// ---------------------------------------------------------------------------
bool test_turbo_quant_index() {
  std::cout << "=== Test 9: TurboQuantIndex ===" << std::endl;

  constexpr size_t D = 128;
  constexpr size_t N = 2000;
  constexpr size_t NUM_QUERIES = 50;
  constexpr size_t K = 10;
  constexpr size_t M = 16;
  constexpr size_t EF_CONSTRUCTION = 200;

  auto embeddings = generateFaceEmbeddings(N, D, 100);

  // Flat contiguous array
  std::vector<float> flat(N * D);
  for (size_t i = 0; i < N; ++i)
    std::memcpy(flat.data() + i * D, embeddings[i].data(), D * sizeof(float));

  // Build
  TurboQuantIndex idx(D, 8);
  auto st = idx.build(flat.data(), N, M, EF_CONSTRUCTION);
  if (!st.ok()) {
    std::cout << "  Build failed: " << st.message() << "  FAIL" << std::endl;
    return false;
  }
  std::cout << "  Built: " << idx.numElements() << " vectors, "
            << idx.codeSizeBytes() << " B/code" << std::endl;

  // Save with raw vectors
  const std::string idx_path = "test_tqi_index.bin";
  const std::string raw_path = "test_tqi_raw.tqrv";
  st = idx.save(idx_path, raw_path, flat.data(), N);
  if (!st.ok()) {
    std::cout << "  Save failed: " << st.message() << "  FAIL" << std::endl;
    return false;
  }

  // Load
  TurboQuantIndex idx2(D, 8);
  st = idx2.load(idx_path, raw_path);
  if (!st.ok()) {
    std::cout << "  Load failed: " << st.message() << "  FAIL" << std::endl;
    return false;
  }
  std::cout << "  Loaded: " << idx2.numElements() << " vectors, hasRaw="
            << idx2.hasRawVectors() << std::endl;

  // Exact brute-force ground truth
  std::mt19937_64 rng(777);
  std::uniform_int_distribution<size_t> query_dist(0, N - 1);

  idx2.setEf(64);
  size_t search_hits = 0;
  size_t rerank_hits = 0;

  for (size_t q = 0; q < NUM_QUERIES; ++q) {
    size_t qi = query_dist(rng);
    const float *query = embeddings[qi].data();

    // Exact top-K
    std::vector<std::pair<float, size_t>> exact_dists(N);
    for (size_t i = 0; i < N; ++i)
      exact_dists[i] = {l2_dist(query, embeddings[i].data(), D), i};
    std::partial_sort(exact_dists.begin(), exact_dists.begin() + K,
                      exact_dists.end());

    // TQ search
    auto result = idx2.search(query, K);
    std::vector<size_t> search_ids;
    while (!result.empty()) {
      search_ids.push_back(result.top().second);
      result.pop();
    }
    for (size_t i = 0; i < K; ++i)
      for (size_t j = 0; j < search_ids.size(); ++j)
        if (search_ids[j] == exact_dists[i].second) { ++search_hits; break; }

    // Rerank search
    auto rerank = idx2.searchRerank(query, K, 100);
    for (size_t i = 0; i < K; ++i)
      for (size_t j = 0; j < rerank.size(); ++j)
        if (rerank[j].second == exact_dists[i].second) { ++rerank_hits; break; }
  }

  float search_recall = static_cast<float>(search_hits) /
                         static_cast<float>(NUM_QUERIES * K);
  float rerank_recall = static_cast<float>(rerank_hits) /
                         static_cast<float>(NUM_QUERIES * K);

  // Cleanup
  std::remove(idx_path.c_str());
  std::remove(raw_path.c_str());

  // Rerank should be better than or equal to TQ-only
  bool pass = search_recall > 0.50f && rerank_recall >= search_recall;
  std::cout << "  Search recall@" << K << ": " << std::fixed
            << std::setprecision(2) << (search_recall * 100.0f) << "%"
            << std::endl;
  std::cout << "  Rerank recall@" << K << ": " << (rerank_recall * 100.0f)
            << "%";
  if (!pass) {
    std::cout << "  FAIL";
  } else {
    std::cout << "  PASS";
  }
  std::cout << std::endl;
  return pass;
}

// ---------------------------------------------------------------------------
// Test 10: TurboQuantIndex::buildFromL2 — L2 graph + TQ search + L2 rerank
// Verifies that:
//   - initL2Build / addL2Point / finalizeFromL2 compile and run end-to-end
//   - buildFromL2() one-shot helper works
//   - searchRerank() reads from the in-memory raw data stashed by finalize
//   - recall from this path is at least as good as tq-graph rerank
// ---------------------------------------------------------------------------
bool test_turbo_quant_index_l2graph() {
  std::cout << "=== Test 10: TurboQuantIndex::buildFromL2 ===" << std::endl;

  constexpr size_t D = 128;
  constexpr size_t N = 2000;
  constexpr size_t NUM_QUERIES = 50;
  constexpr size_t K = 10;
  constexpr size_t M = 16;
  constexpr size_t EF_CONSTRUCTION = 200;

  auto embeddings = generateFaceEmbeddings(N, D, 100);

  // Flat contiguous buffer
  std::vector<float> flat(N * D);
  for (size_t i = 0; i < N; ++i)
    std::memcpy(flat.data() + i * D, embeddings[i].data(), D * sizeof(float));

  // ---- Path A: low-level API (initL2Build + addL2Point + finalizeFromL2) --
  TurboQuantIndex idx_lowlevel(D, 8);
  idx_lowlevel.initL2Build(N, M, EF_CONSTRUCTION);
  for (size_t i = 0; i < N; ++i)
    idx_lowlevel.addL2Point(flat.data() + i * D, i);
  auto st = idx_lowlevel.finalizeFromL2(flat.data(), N, /*keep_raw=*/true);
  if (!st.ok()) {
    std::cout << "  finalizeFromL2 failed: " << st.message() << "  FAIL"
              << std::endl;
    return false;
  }
  if (!idx_lowlevel.hasRawVectors()) {
    std::cout << "  hasRawVectors() returned false after keep_raw=true  FAIL"
              << std::endl;
    return false;
  }

  // ---- Path B: high-level one-shot helper --------------------------------
  TurboQuantIndex idx_oneshot(D, 8);
  st = idx_oneshot.buildFromL2(flat.data(), N, M, EF_CONSTRUCTION,
                               /*keep_raw=*/true);
  if (!st.ok()) {
    std::cout << "  buildFromL2 failed: " << st.message() << "  FAIL"
              << std::endl;
    return false;
  }

  // ---- Measure recall on both -------------------------------------------
  std::mt19937_64 rng(999);
  std::uniform_int_distribution<size_t> query_dist(0, N - 1);

  idx_lowlevel.setEf(64);
  idx_oneshot.setEf(64);

  size_t rerank_hits_lowlevel = 0;
  size_t rerank_hits_oneshot = 0;
  size_t search_hits_lowlevel = 0;

  for (size_t q = 0; q < NUM_QUERIES; ++q) {
    size_t qi = query_dist(rng);
    const float *query = embeddings[qi].data();

    // Exact top-K
    std::vector<std::pair<float, size_t>> exact_dists(N);
    for (size_t i = 0; i < N; ++i)
      exact_dists[i] = {l2_dist(query, embeddings[i].data(), D), i};
    std::partial_sort(exact_dists.begin(), exact_dists.begin() + K,
                      exact_dists.end());

    // Pure TQ search (no rerank) on the low-level index
    auto search_result = idx_lowlevel.search(query, K);
    while (!search_result.empty()) {
      size_t id = search_result.top().second;
      for (size_t i = 0; i < K; ++i) {
        if (exact_dists[i].second == id) { ++search_hits_lowlevel; break; }
      }
      search_result.pop();
    }

    // Rerank path — low-level index
    auto rerank_ll = idx_lowlevel.searchRerank(query, K, 100);
    for (const auto &p : rerank_ll)
      for (size_t i = 0; i < K; ++i)
        if (exact_dists[i].second == p.second) { ++rerank_hits_lowlevel; break; }

    // Rerank path — one-shot index
    auto rerank_os = idx_oneshot.searchRerank(query, K, 100);
    for (const auto &p : rerank_os)
      for (size_t i = 0; i < K; ++i)
        if (exact_dists[i].second == p.second) { ++rerank_hits_oneshot; break; }
  }

  float search_recall =
      static_cast<float>(search_hits_lowlevel) / (NUM_QUERIES * K);
  float rerank_recall_ll =
      static_cast<float>(rerank_hits_lowlevel) / (NUM_QUERIES * K);
  float rerank_recall_os =
      static_cast<float>(rerank_hits_oneshot) / (NUM_QUERIES * K);

  std::cout << "  Code size:             " << idx_lowlevel.codeSizeBytes()
            << " B/vec" << std::endl;
  std::cout << "  TQ-only search recall: " << std::fixed
            << std::setprecision(2) << (search_recall * 100.0f) << "%"
            << std::endl;
  std::cout << "  Rerank recall (llapi): " << (rerank_recall_ll * 100.0f)
            << "%" << std::endl;
  std::cout << "  Rerank recall (1shot): " << (rerank_recall_os * 100.0f)
            << "%" << std::endl;

  // L2-graph rerank should dominate TQ-only search.
  bool pass = rerank_recall_ll >= search_recall &&
              rerank_recall_os >= search_recall &&
              rerank_recall_ll > 0.90f && rerank_recall_os > 0.90f;
  std::cout << "  " << (pass ? "PASS" : "FAIL") << std::endl;
  return pass;
}

// ---------------------------------------------------------------------------
// Multithread benchmark: build + search scaling at 1,4,8,16,32 threads
// Compares L2Space (float32) vs TurboQuantSpace (q8) to test the hypothesis
// that compressed codes improve memory-bandwidth-limited throughput.
// ---------------------------------------------------------------------------
void test_multithread_scaling() {
  std::cout << "=== Multithread scaling: L2 vs TurboQuant (q8) ===" << std::endl;

  constexpr size_t D = 128;
  constexpr size_t N = 50000;
  constexpr size_t NUM_QUERIES = 5000;
  constexpr size_t K = 10;
  constexpr size_t M = 16;
  constexpr size_t EF_CONSTRUCTION = 100;
  constexpr size_t EF_SEARCH = 64;
  constexpr int NUM_RUNS = 3; // median of N runs for stable results

  const std::vector<int> thread_counts = {1, 4, 8, 16, 32, 64};

  // Pre-generate all data
  std::cout << "  Generating " << N << " vectors (d=" << D << ")..." << std::endl;
  auto embeddings = generateFaceEmbeddings(N, D, 200);

  std::mt19937_64 qrng(99999);
  std::vector<std::vector<float>> queries(NUM_QUERIES, std::vector<float>(D));
  for (size_t q = 0; q < NUM_QUERIES; ++q)
    generateRandomVector(queries[q].data(), D, qrng);

  // Pre-encode TQ codes for build benchmark
  uint64_t rot_seed = 42;
  uint64_t qjl_seed = 137;
  TurboQuantSpace tq_space_ref(D, 8, rot_seed, qjl_seed);
  size_t code_size = tq_space_ref.codeSizeBytes();
  std::vector<std::vector<char>> tq_codes(N, std::vector<char>(code_size));
  for (size_t i = 0; i < N; ++i)
    tq_space_ref.encodeVector(embeddings[i].data(), tq_codes[i].data());

  std::cout << "  data_size/vec: L2=" << D * sizeof(float)
            << " B, TQ=" << code_size << " B ("
            << std::fixed << std::setprecision(1)
            << float(D * sizeof(float)) / code_size << "x compression)"
            << std::endl;

  // Header
  std::cout << "\n  "
            << std::setw(8) << "threads"
            << std::setw(14) << "L2_build_s"
            << std::setw(14) << "TQ_build_s"
            << std::setw(14) << "L2_qps"
            << std::setw(14) << "TQ_qps"
            << std::setw(12) << "build_ratio"
            << std::setw(12) << "search_ratio"
            << std::endl;
  std::cout << "  " << std::string(88, '-') << std::endl;

  // Helper: median of measurements
  auto medianOf = [](std::vector<double> &v) -> double {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
  };

  // Build L2 index once (single-threaded — build scaling tested separately)
  std::cout << "  Building L2 index..." << std::flush;
  hnswlib::L2Space l2space(D);
  hnswlib::HierarchicalNSW<float> hnsw_l2(&l2space, N, M, EF_CONSTRUCTION);
  for (size_t i = 0; i < N; ++i)
    hnsw_l2.addPoint(embeddings[i].data(), i);
  hnsw_l2.setEf(EF_SEARCH);
  std::cout << " done" << std::endl;

  // Build TQ index once
  std::cout << "  Building TQ index..." << std::flush;
  TurboQuantSpace tq_space_bench(D, 8, rot_seed, qjl_seed);
  hnswlib::HierarchicalNSW<float> hnsw_tq(&tq_space_bench, N, M, EF_CONSTRUCTION);
  for (size_t i = 0; i < N; ++i)
    hnsw_tq.addPoint(tq_codes[i].data(), i);
  hnsw_tq.setEf(EF_SEARCH);
  tq_space_bench.setSearchMode(hnsw_tq);
  std::cout << " done" << std::endl;

  // Build scaling (still per thread_count, cannot reuse)
  std::cout << "\n  --- Build scaling ---\n  "
            << std::setw(8) << "threads"
            << std::setw(14) << "L2_build_s"
            << std::setw(14) << "TQ_build_s"
            << std::setw(12) << "build_ratio"
            << std::endl;
  std::cout << "  " << std::string(48, '-') << std::endl;

  for (int num_threads : thread_counts) {
    double l2_build_sec;
    {
      hnswlib::L2Space ls(D);
      hnswlib::HierarchicalNSW<float> h(&ls, N, M, EF_CONSTRUCTION);
      h.addPoint(embeddings[0].data(), 0);
      auto t0 = std::chrono::high_resolution_clock::now();
      std::vector<std::thread> threads;
      size_t per_thread = (N - 1) / num_threads;
      for (int t = 0; t < num_threads; ++t) {
        size_t start = 1 + t * per_thread;
        size_t end = (t == num_threads - 1) ? N : start + per_thread;
        threads.emplace_back([&, start, end]() {
          for (size_t i = start; i < end; ++i)
            h.addPoint(embeddings[i].data(), i);
        });
      }
      for (auto &th : threads) th.join();
      l2_build_sec = std::chrono::duration<double>(
          std::chrono::high_resolution_clock::now() - t0).count();
    }

    double tq_build_sec;
    {
      TurboQuantSpace ts(D, 8, rot_seed, qjl_seed);
      hnswlib::HierarchicalNSW<float> h(&ts, N, M, EF_CONSTRUCTION);
      h.addPoint(tq_codes[0].data(), 0);
      auto t0 = std::chrono::high_resolution_clock::now();
      std::vector<std::thread> threads;
      size_t per_thread = (N - 1) / num_threads;
      for (int t = 0; t < num_threads; ++t) {
        size_t start = 1 + t * per_thread;
        size_t end = (t == num_threads - 1) ? N : start + per_thread;
        threads.emplace_back([&, start, end]() {
          for (size_t i = start; i < end; ++i)
            h.addPoint(tq_codes[i].data(), i);
        });
      }
      for (auto &th : threads) th.join();
      tq_build_sec = std::chrono::duration<double>(
          std::chrono::high_resolution_clock::now() - t0).count();
    }

    std::cout << "  "
              << std::setw(8) << num_threads
              << std::setw(14) << std::fixed << std::setprecision(3) << l2_build_sec
              << std::setw(14) << tq_build_sec
              << std::setw(12) << std::setprecision(2) << l2_build_sec / tq_build_sec
              << std::endl;
  }

  // Search scaling (median of NUM_RUNS runs, reusing pre-built indexes)
  std::cout << "\n  --- Search scaling (median of " << NUM_RUNS << " runs) ---\n  "
            << std::setw(8) << "threads"
            << std::setw(14) << "L2_qps"
            << std::setw(14) << "TQ_qps"
            << std::setw(12) << "search_ratio"
            << std::endl;
  std::cout << "  " << std::string(48, '-') << std::endl;

  for (int num_threads : thread_counts) {
    std::vector<double> l2_times(NUM_RUNS), tq_times(NUM_RUNS);

    for (int run = 0; run < NUM_RUNS; ++run) {
      // L2 search
      {
        auto t0 = std::chrono::high_resolution_clock::now();
        std::vector<std::thread> threads;
        size_t per_thread = NUM_QUERIES / num_threads;
        for (int t = 0; t < num_threads; ++t) {
          size_t start = t * per_thread;
          size_t end = (t == num_threads - 1) ? NUM_QUERIES : start + per_thread;
          threads.emplace_back([&, start, end]() {
            for (size_t q = start; q < end; ++q) {
              auto result = hnsw_l2.searchKnn(queries[q].data(), K);
              (void)result;
            }
          });
        }
        for (auto &th : threads) th.join();
        l2_times[run] = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - t0).count();
      }
      // TQ search
      {
        auto t0 = std::chrono::high_resolution_clock::now();
        std::vector<std::thread> threads;
        size_t per_thread = NUM_QUERIES / num_threads;
        for (int t = 0; t < num_threads; ++t) {
          size_t start = t * per_thread;
          size_t end = (t == num_threads - 1) ? NUM_QUERIES : start + per_thread;
          threads.emplace_back([&, start, end]() {
            for (size_t q = start; q < end; ++q) {
              auto pq = tq_space_bench.prepareQuery(queries[q].data());
              auto result = hnsw_tq.searchKnn(&pq, K);
              (void)result;
            }
          });
        }
        for (auto &th : threads) th.join();
        tq_times[run] = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - t0).count();
      }
    }

    double l2_sec = medianOf(l2_times);
    double tq_sec = medianOf(tq_times);
    double l2_qps = NUM_QUERIES / l2_sec;
    double tq_qps = NUM_QUERIES / tq_sec;

    std::cout << "  "
              << std::setw(8) << num_threads
              << std::setw(14) << std::setprecision(0) << l2_qps
              << std::setw(14) << tq_qps
              << std::setw(12) << std::setprecision(2) << tq_qps / l2_qps
              << std::endl;
  }

  std::cout << "\n  build_ratio = L2_time / TQ_time (>1 means TQ builds faster)"
            << std::endl;
  std::cout << "  search_ratio = TQ_qps / L2_qps (>1 means TQ searches faster)"
            << std::endl;
}

// ---------------------------------------------------------------------------
int main() {
  std::cout << "TurboQuant Algorithm 2 — Correctness Tests\n"
            << "d=128, testing b=4 (3+1) and b=8 (7+1)\n"
            << "============================================\n"
            << std::endl;

  int passed = 0, total = 0;
  auto run = [&](bool result) {
    ++total;
    if (result)
      ++passed;
    std::cout << std::endl;
  };

  // WHT is independent of bit width
  run(test_wht_energy());

  // Run parametric tests for b=4 and b=8
  for (int b : {4, 8}) {
    std::cout << "---- bits_per_coord = " << b << " (" << (b - 1)
              << "-bit MSE + 1-bit QJL) ----\n"
              << std::endl;

    run(test_sq_distortion(b));
    run(test_unbiasedness(b));
    run(test_lossless_identity(b));
    run(test_ip_correlation(b));
    run(test_recall_at_k(b));
    run(test_encode_determinism(b));
    run(test_hnsw_integration(b));
    run(test_save_load_index(b));
    run(test_lut_correctness(b));
  }

  run(test_turbo_quant_index());
  run(test_turbo_quant_index_l2graph());

  test_memory_footprint();
  std::cout << std::endl;
  test_benchmark();
  std::cout << std::endl;
  test_large_scale_comparison();
  std::cout << std::endl;
  test_multithread_scaling();
  std::cout << std::endl;

  std::cout << "============================================" << std::endl;
  std::cout << "Results: " << passed << "/" << total << " tests passed"
            << std::endl;

  return (passed == total) ? 0 : 1;
}
