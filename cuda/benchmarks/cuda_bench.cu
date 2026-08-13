#include "neuroseek_cuda.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

struct Case { uint32_t rows; uint32_t dims; };

int run_case(Case item, int repeats) {
    std::vector<float> vectors(static_cast<size_t>(item.rows) * item.dims, 0.01f);
    std::vector<float> query(item.dims, 0.02f);
    std::vector<float> scores(item.rows);
    // One unrecorded call creates the CUDA context and avoids reporting its
    // one-time setup cost as steady-state kernel latency.
    if (neuroseek_cuda_exact_scores(vectors.data(), query.data(), item.rows, item.dims, scores.data()) != NEUROSEEK_CUDA_OK) {
        std::fprintf(stderr, "CUDA warmup failed: %s\n", neuroseek_cuda_last_error());
        return 1;
    }
    auto started = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < repeats; ++iteration) {
        if (neuroseek_cuda_exact_scores(vectors.data(), query.data(), item.rows, item.dims, scores.data()) != NEUROSEEK_CUDA_OK) {
            std::fprintf(stderr, "CUDA exact_scores failed: %s\n", neuroseek_cuda_last_error());
            return 1;
        }
    }
    const double elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    const double mean_ms = elapsed_ms / repeats;
    const double vectors_per_second = mean_ms > 0.0 ? (1000.0 * item.rows / mean_ms) : 0.0;
    // JSONL is intentionally machine-readable input for bench_cost_model.py.
    std::printf("{\"operation\":\"exact_scores\",\"rows\":%u,\"dims\":%u,\"mean_ms\":%.6f,\"throughput_vectors_per_second\":%.3f}\n",
                item.rows, item.dims, mean_ms, vectors_per_second);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    constexpr std::array<Case, 6> standard{{
        {16'384, 32}, {16'384, 64}, {16'384, 128},
        {65'536, 32}, {65'536, 64}, {65'536, 128},
    }};
    const bool single = argc == 3;
    if (argc != 1 && !single) {
        std::fprintf(stderr, "usage: %s [rows dims]\n", argv[0]);
        return 2;
    }
    if (single) {
        char* end = nullptr;
        const auto rows = std::strtoul(argv[1], &end, 10);
        if (*argv[1] == '\0' || *end != '\0' || rows == 0 || rows > UINT32_MAX) {
            std::fprintf(stderr, "rows must be a positive uint32\n");
            return 2;
        }
        const auto dims = std::strtoul(argv[2], &end, 10);
        if (*argv[2] == '\0' || *end != '\0' || dims == 0 || dims > UINT32_MAX) {
            std::fprintf(stderr, "dims must be a positive uint32\n");
            return 2;
        }
        return run_case({static_cast<uint32_t>(rows), static_cast<uint32_t>(dims)}, 12);
    }
    for (Case item : standard) {
        if (run_case(item, 12) != 0) return 1;
    }
    return 0;
}
