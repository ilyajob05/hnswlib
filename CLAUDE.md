# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

hnswlib is a header-only C++ library implementing the Hierarchical Navigable Small World (HNSW) algorithm for fast approximate nearest neighbor search. It includes Python bindings via pybind11.

## Build & Test Commands

### Python bindings
```bash
pip install .                    # Build and install from source
HNSWLIB_NO_NATIVE=1 pip install . # Build without -march=native (for portable builds)
```

### Python tests
```bash
# Run all Python tests
python -m unittest discover --start-directory tests/python --pattern "bindings_test*.py"

# Run a single test file
python -m unittest tests/python/bindings_test_recall.py
```

### C++ build and tests
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo  # or: -DHNSWLIB_ENABLE_EXCEPTIONS=OFF
make    # or: ninja -v (if using -G Ninja)
ctest --build-config RelWithDebInfo

# Update test requires generated data
cd tests/cpp && python update_gen_data.py && cd ../../build
./test_updates
./test_updates update
```

### CMake options
- `HNSWLIB_ENABLE_EXCEPTIONS` (ON/OFF) — controls exception support; some targets always enable exceptions
- `ENABLE_ASAN`, `ENABLE_UBSAN`, `ENABLE_TSAN`, `ENABLE_MSAN` — sanitizer flags

## Architecture

### Core C++ library (`hnswlib/`)
Header-only, no dependencies beyond C++11. All classes are in the `hnswlib` namespace.

- **`hnswlib.h`** — Central header. Defines base abstractions: `SpaceInterface<dist_t>` (distance function contract), `AlgorithmInterface<dist_t>` (search/insert contract with both exception-throwing and `NoExceptions` variants returning `Status`/`StatusOr`), `BaseFilterFunctor` (search-time filtering), `BaseSearchStopCondition` (custom stop criteria). Also includes SIMD detection (SSE/AVX/AVX512) and all other headers.
- **`hnswalg.h`** — `HierarchicalNSW<dist_t>`: the main HNSW index. Multi-layer graph with configurable M, ef_construction, ef. Supports concurrent insertions (mutex per link list + label operation locks), element deletion/replacement, and serialization.
- **`bruteforce.h`** — `BruteforceSearch<dist_t>`: brute-force baseline implementing the same `AlgorithmInterface`.
- **`space_l2.h`** / **`space_ip.h`** — `L2Space`, `InnerProductSpace` (+ `int` variants) implementing `SpaceInterface`. Contain SIMD-optimized distance functions selected at runtime based on CPU capabilities and dimension.
- **`visited_list_pool.h`** — Thread-safe pool of visited-node bitsets used during graph traversal.
- **`stop_condition.h`** — Implementations of `BaseSearchStopCondition` for epsilon search and multi-vector search.

### Python bindings (`python_bindings/`)
- **`bindings.cpp`** — pybind11 module exposing `Index` and `BFIndex` classes. Implements its own `ParallelFor` (no OpenMP dependency on macOS). The `Index` class wraps `HierarchicalNSW<float>` and adds cosine normalization.
- **`__init__.py`** — Re-exports the C extension module.
- Build system: `setup.py` + `pyproject.toml`. Requires numpy and pybind11 at build time.

### Key design points
- Distance type is templated (`dist_t`), but Python bindings only use `float`.
- Cosine distance is implemented as inner product on normalized vectors (normalization happens in the Python binding layer).
- The API has dual interfaces: exception-throwing methods (`addPoint`, `searchKnn`) and `NoExceptions` variants returning `Status`/`StatusOr`, controlled by `HNSWLIB_ENABLE_EXCEPTIONS`.
- Python bindings always compile with exceptions enabled (`HNSWLIB_ENABLE_EXCEPTIONS=ON`); the no-exceptions mode is for embedded C++ use only. <!-- reflected: 2026-03-27 -->
- `AlgorithmInterface` is an internal abstraction with exactly two implementations (`HierarchicalNSW`, `BruteforceSearch`); do not design for external subclasses. <!-- reflected: 2026-03-27 -->
- Thread safety: `add_items` is thread-safe with other `add_items` calls but NOT with `knn_query`.

## Development Workflow

- PRs should target the `develop` branch, not `master`.
- CI runs on ubuntu/windows/macos with Python 3.9-3.13 and multiple C++ compiler/sanitizer/exception combinations.
- When adding new Python functionality, add a test in `tests/python/` matching `bindings_test*.py`.
- C++ tests go in `tests/cpp/` and should be registered in `CMakeLists.txt` under `TEST_NAMES`.
- C++ examples go in `examples/cpp/` and should be registered under `EXAMPLE_NAMES`.
- Release changelogs should only include library-facing changes; omit CI/infrastructure updates. <!-- reflected: 2026-03-27 -->
- Releases follow Git Flow: cherry-pick to `release/X.Y.Z` from `develop`, merge to `master`, tag on `master`, delete the release branch. <!-- reflected: 2026-03-27 -->
- Version string lives only in `setup.py` (`__version__`); update it on the release branch before merging to `master`. <!-- reflected: 2026-03-27 -->
