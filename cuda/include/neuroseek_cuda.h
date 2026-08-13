#pragma once

#include <stdint.h>

// Deliberately narrow C ABI: all pointers are host pointers on entry/return.
// The implementation owns temporary device buffers and synchronizes before it
// returns, so callers never observe an asynchronous CUDA failure as success.
// Production bindings may add stream-aware APIs only with equivalent checks.
#ifdef __cplusplus
extern "C" {
#endif

typedef enum neuroseek_cuda_status {
  NEUROSEEK_CUDA_OK = 0,
  NEUROSEEK_CUDA_INVALID_ARGUMENT = 1,
  NEUROSEEK_CUDA_UNAVAILABLE = 2,
  NEUROSEEK_CUDA_RUNTIME_ERROR = 3,
  NEUROSEEK_CUDA_OUTPUT_CAPACITY = 4
} neuroseek_cuda_status;

// Opaque ownership handle for a GPU-resident, relation-tagged CSR. A session
// owns a single traversal direction (forward or reverse); create two sessions
// when both directions must be hot. All calls are synchronous and report
// asynchronous CUDA errors before returning.
typedef struct neuroseek_cuda_csr_session neuroseek_cuda_csr_session;

const char* neuroseek_cuda_last_error(void);
neuroseek_cuda_status neuroseek_cuda_device_count(int* count);

// Expands a frontier in a relation-tagged CSR. `offsets` must have exactly
// node_count + 1 entries, targets/relations exactly edge_count entries, and
// all offsets must be monotonic from zero to edge_count. `relation` is -1 for
// any relation, otherwise it must fit uint16. Output can contain duplicates;
// VM-level deterministic compaction owns semantic deduplication.
//
// This synchronous ABI accepts host pointers (including read-only mmap
// buffers). It validates the complete CSR before launching a kernel and owns
// all temporary device memory. `output_len` always receives the *required*
// number of results, including when output_capacity is too small. Empty
// frontiers and zero-capacity probes are valid.
neuroseek_cuda_status neuroseek_cuda_expand(
    const uint64_t* offsets, const uint32_t* targets, const uint16_t* relations,
    uint32_t node_count, uint64_t edge_count,
    const uint32_t* frontier, uint32_t frontier_len, int32_t relation,
    uint32_t* output, uint64_t output_capacity, uint64_t* output_len);

// Validates and uploads a complete CSR exactly once. The host arrays may be
// released after successful creation. Call destroy exactly once for each
// successful create; destroy accepts NULL as a no-op for cleanup paths.
neuroseek_cuda_status neuroseek_cuda_csr_session_create(
    const uint64_t* offsets, const uint32_t* targets, const uint16_t* relations,
    uint32_t node_count, uint64_t edge_count,
    neuroseek_cuda_csr_session** session);
neuroseek_cuda_status neuroseek_cuda_csr_session_destroy(neuroseek_cuda_csr_session* session);

// Expands against GPU-resident CSR without re-uploading graph arrays. Only the
// host frontier and returned results cross the API boundary. Semantics and
// capacity behavior are identical to neuroseek_cuda_expand.
neuroseek_cuda_status neuroseek_cuda_csr_session_expand(
    const neuroseek_cuda_csr_session* session,
    const uint32_t* frontier, uint32_t frontier_len, int32_t relation,
    uint32_t* output, uint64_t output_capacity, uint64_t* output_len);

// Computes exact dot-product scores for fp32 row-major vectors. It is the
// mandatory ANN fallback, not an approximate index.
neuroseek_cuda_status neuroseek_cuda_exact_scores(
    const float* vectors, const float* query, uint32_t rows, uint32_t dims,
    float* scores);

// Stable tie rule: greater score first, then lower index. k must be <= 1024.
neuroseek_cuda_status neuroseek_cuda_topk(
    const float* scores, uint32_t rows, uint32_t k, uint32_t* indices,
    float* values);
#ifdef __cplusplus
}
#endif
