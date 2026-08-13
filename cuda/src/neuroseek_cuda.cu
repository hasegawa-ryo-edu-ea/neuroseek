#include "neuroseek_cuda.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <new>
#include <vector>
#include <math.h>

struct neuroseek_cuda_csr_session {
  uint64_t* offsets = nullptr;
  uint32_t* targets = nullptr;
  uint16_t* relations = nullptr;
  uint32_t node_count = 0;
  uint64_t edge_count = 0;
};

namespace {
thread_local char g_error[256] = "";
neuroseek_cuda_status fail(neuroseek_cuda_status s, const char* where, cudaError_t e) {
  std::snprintf(g_error, sizeof(g_error), "%s: %s", where, cudaGetErrorString(e)); return s;
}
neuroseek_cuda_status invalid(const char* message) {
  std::snprintf(g_error, sizeof(g_error), "%s", message); return NEUROSEEK_CUDA_INVALID_ARGUMENT;
}
void clear_error() { g_error[0] = '\0'; }
neuroseek_cuda_status ensure_available() {
  int n = 0;
  const cudaError_t e = cudaGetDeviceCount(&n);
  if (e != cudaSuccess) return fail(NEUROSEEK_CUDA_UNAVAILABLE, "cudaGetDeviceCount", e);
  if (n <= 0) {
    std::snprintf(g_error, sizeof(g_error), "cudaGetDeviceCount: no CUDA device is visible");
    return NEUROSEEK_CUDA_UNAVAILABLE;
  }
  return NEUROSEEK_CUDA_OK;
}
template<class T> cudaError_t copy_to_device(T** d, const T* h, size_t n) {
  *d = nullptr;
  cudaError_t e = cudaMalloc((void**)d, n * sizeof(T));
  if (e != cudaSuccess) return e;
  e = cudaMemcpy(*d, h, n * sizeof(T), cudaMemcpyHostToDevice);
  if (e != cudaSuccess) { cudaFree(*d); *d = nullptr; }
  return e;
}
template<class T> void free_d(T* p) { if (p) cudaFree(p); }

neuroseek_cuda_status validate_csr(const uint64_t* offsets, const uint32_t* targets,
                                   const uint16_t* relations, uint32_t node_count,
                                   uint64_t edge_count, const char* operation) {
  if (!offsets || (edge_count && (!targets || !relations))) {
    char message[128]; std::snprintf(message, sizeof(message), "%s: null CSR pointer", operation);
    return invalid(message);
  }
  if (edge_count > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    char message[128]; std::snprintf(message, sizeof(message), "%s: edge_count exceeds addressable size", operation);
    return invalid(message);
  }
  if (offsets[0] != 0) {
    char message[128]; std::snprintf(message, sizeof(message), "%s: offsets[0] must be zero", operation);
    return invalid(message);
  }
  for (uint64_t i = 0; i < static_cast<uint64_t>(node_count); ++i) {
    if (offsets[i] > offsets[i + 1] || offsets[i + 1] > edge_count) {
      char message[128]; std::snprintf(message, sizeof(message), "%s: non-monotonic or out-of-range offsets", operation);
      return invalid(message);
    }
  }
  if (offsets[node_count] != edge_count) {
    char message[128]; std::snprintf(message, sizeof(message), "%s: final offset does not equal edge_count", operation);
    return invalid(message);
  }
  return NEUROSEEK_CUDA_OK;
}

neuroseek_cuda_status validate_expand_request(uint32_t node_count, const uint32_t* frontier,
                                               uint32_t frontier_len, int32_t relation,
                                               uint32_t* output, uint64_t output_capacity,
                                               uint64_t* output_len, const char* operation) {
  if (!output_len || (frontier_len && !frontier) || (output_capacity && !output)) {
    char message[128]; std::snprintf(message, sizeof(message), "%s: null required pointer", operation);
    return invalid(message);
  }
  *output_len = 0;
  if (relation < -1 || relation > std::numeric_limits<uint16_t>::max()) {
    char message[128]; std::snprintf(message, sizeof(message), "%s: relation out of range", operation);
    return invalid(message);
  }
  for (uint32_t i = 0; i < frontier_len; ++i) if (frontier[i] >= node_count) {
    char message[128]; std::snprintf(message, sizeof(message), "%s: frontier node out of range", operation);
    return invalid(message);
  }
  return NEUROSEEK_CUDA_OK;
}

__global__ void expand_kernel(const uint64_t* offsets, const uint32_t* targets, const uint16_t* rels,
                              const uint32_t* frontier, uint32_t n, int32_t wanted,
                              uint32_t* out, uint64_t capacity, unsigned long long* count);

neuroseek_cuda_status run_expand(const uint64_t* doff, const uint32_t* dt, const uint16_t* dr,
                                 uint32_t node_count, const uint32_t* frontier, uint32_t frontier_len,
                                 int32_t relation, uint32_t* output, uint64_t output_capacity,
                                 uint64_t* output_len, const char* operation) {
  if (frontier_len == 0) { clear_error(); return NEUROSEEK_CUDA_OK; }
  uint32_t *df = nullptr, *dout = nullptr; unsigned long long* dc = nullptr;
  cudaError_t e;
  if (output_capacity > static_cast<uint64_t>(std::numeric_limits<size_t>::max() / sizeof(uint32_t))) return invalid("expand: output capacity exceeds addressable size");
  e = copy_to_device(&df, frontier, frontier_len);
  if (e == cudaSuccess) e = cudaMalloc((void**)&dout, std::max<uint64_t>(1, output_capacity) * sizeof(uint32_t));
  if (e == cudaSuccess) e = cudaMalloc((void**)&dc, sizeof(unsigned long long));
  if (e != cudaSuccess) {
    free_d(df); free_d(dout); free_d(dc);
    return fail(NEUROSEEK_CUDA_RUNTIME_ERROR, operation, e);
  }
  e = cudaMemset(dc, 0, sizeof(unsigned long long));
  if (e == cudaSuccess) expand_kernel<<<(frontier_len + 255) / 256, 256>>>(doff, dt, dr, df, frontier_len, relation, dout, output_capacity, dc);
  if (e == cudaSuccess) e = cudaGetLastError();
  if (e == cudaSuccess) e = cudaDeviceSynchronize();
  if (e == cudaSuccess) e = cudaMemcpy(output_len, dc, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
  if (e == cudaSuccess && *output_len && *output_len <= output_capacity) e = cudaMemcpy(output, dout, static_cast<size_t>(*output_len) * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  free_d(df); free_d(dout); free_d(dc);
  if (e != cudaSuccess) return fail(NEUROSEEK_CUDA_RUNTIME_ERROR, operation, e);
  if (*output_len > output_capacity) return NEUROSEEK_CUDA_OUTPUT_CAPACITY;
  clear_error(); return NEUROSEEK_CUDA_OK;
}

__global__ void expand_kernel(const uint64_t* offsets, const uint32_t* targets, const uint16_t* rels,
                              const uint32_t* frontier, uint32_t n, int32_t wanted,
                              uint32_t* out, uint64_t capacity, unsigned long long* count) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const uint32_t node = frontier[i];
  for (uint64_t e = offsets[node]; e < offsets[node + 1]; ++e) {
    if (wanted >= 0 && rels[e] != static_cast<uint16_t>(wanted)) continue;
    const unsigned long long pos = atomicAdd(count, 1ULL);
    if (pos < capacity) out[pos] = targets[e];
  }
}
__global__ void score_kernel(const float* vectors, const float* query, uint32_t rows, uint32_t dims, float* scores) {
  uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  float sum = 0.f; const float* v = vectors + static_cast<size_t>(row) * dims;
  for (uint32_t d=0; d<dims; ++d) sum = fmaf(v[d], query[d], sum);
  scores[row] = sum;
}
// Intentionally single-threaded device selection: it is a correctness-safe
// small-k fallback, not the high-throughput large-k path.
__global__ void topk_kernel(const float* scores, uint32_t rows, uint32_t k, uint32_t* indices, float* values) {
  if (blockIdx.x || threadIdx.x) return;
  for (uint32_t rank=0; rank<k; ++rank) {
    float best = -INFINITY; uint32_t bi = 0;
    for (uint32_t i=0; i<rows; ++i) { float s=scores[i]; bool used=false; for(uint32_t j=0;j<rank;++j) if(indices[j]==i) {used=true;break;} if(!used && (s>best || (s==best && i<bi))) {best=s;bi=i;} }
    indices[rank]=bi; values[rank]=best;
  }
}
}

extern "C" const char* neuroseek_cuda_last_error(void) { return g_error; }
extern "C" neuroseek_cuda_status neuroseek_cuda_device_count(int* count) {
  if (!count) return invalid("device_count: count is null");
  cudaError_t e=cudaGetDeviceCount(count);
  if (e != cudaSuccess) return fail(NEUROSEEK_CUDA_UNAVAILABLE,"cudaGetDeviceCount",e);
  clear_error(); return NEUROSEEK_CUDA_OK;
}
extern "C" neuroseek_cuda_status neuroseek_cuda_expand(const uint64_t* offsets,const uint32_t* targets,const uint16_t* relations,uint32_t node_count,uint64_t edge_count,const uint32_t* frontier,uint32_t frontier_len,int32_t relation,uint32_t* output,uint64_t output_capacity,uint64_t* output_len) {
  if (const auto valid = validate_expand_request(node_count, frontier, frontier_len, relation, output, output_capacity, output_len, "expand"); valid != NEUROSEEK_CUDA_OK) return valid;
  if (const auto valid = validate_csr(offsets, targets, relations, node_count, edge_count, "expand"); valid != NEUROSEEK_CUDA_OK) return valid;
  if (frontier_len == 0) { clear_error(); return NEUROSEEK_CUDA_OK; }
  if (const auto ready = ensure_available(); ready != NEUROSEEK_CUDA_OK) return ready;
  uint64_t *doff=nullptr; uint32_t *dt=nullptr; uint16_t*dr=nullptr;
  const size_t off_n = static_cast<size_t>(node_count) + 1;
  const size_t edge_n = static_cast<size_t>(edge_count);
  static const uint32_t dummy_target = 0; static const uint16_t dummy_relation = 0;
  cudaError_t upload_error = copy_to_device(&doff, offsets, off_n);
  if (upload_error == cudaSuccess) upload_error = copy_to_device(&dt, edge_n ? targets : &dummy_target, edge_n ? edge_n : 1);
  if (upload_error == cudaSuccess) upload_error = copy_to_device(&dr, edge_n ? relations : &dummy_relation, edge_n ? edge_n : 1);
  if (upload_error != cudaSuccess) { free_d(doff);free_d(dt);free_d(dr); return fail(NEUROSEEK_CUDA_RUNTIME_ERROR,"expand graph upload",upload_error); }
  const auto result = run_expand(doff, dt, dr, node_count, frontier, frontier_len, relation, output, output_capacity, output_len, "expand");
  free_d(doff);free_d(dt);free_d(dr); return result;
}
extern "C" neuroseek_cuda_status neuroseek_cuda_csr_session_create(
    const uint64_t* offsets, const uint32_t* targets, const uint16_t* relations,
    uint32_t node_count, uint64_t edge_count, neuroseek_cuda_csr_session** session) {
  if (!session) return invalid("csr_session_create: session output is null");
  *session = nullptr;
  if (const auto valid = validate_csr(offsets, targets, relations, node_count, edge_count, "csr_session_create"); valid != NEUROSEEK_CUDA_OK) return valid;
  if (const auto ready = ensure_available(); ready != NEUROSEEK_CUDA_OK) return ready;
  auto* created = new (std::nothrow) neuroseek_cuda_csr_session();
  if (!created) return invalid("csr_session_create: host allocation failed");
  const size_t offset_count = static_cast<size_t>(node_count) + 1;
  const size_t edge_count_size = static_cast<size_t>(edge_count);
  static const uint32_t dummy_target = 0;
  static const uint16_t dummy_relation = 0;
  cudaError_t upload_error = copy_to_device(&created->offsets, offsets, offset_count);
  if (upload_error == cudaSuccess) upload_error = copy_to_device(&created->targets, edge_count_size ? targets : &dummy_target, edge_count_size ? edge_count_size : 1);
  if (upload_error == cudaSuccess) upload_error = copy_to_device(&created->relations, edge_count_size ? relations : &dummy_relation, edge_count_size ? edge_count_size : 1);
  if (upload_error != cudaSuccess) {
    free_d(created->offsets); free_d(created->targets); free_d(created->relations); delete created;
    return fail(NEUROSEEK_CUDA_RUNTIME_ERROR, "csr_session_create graph upload", upload_error);
  }
  created->node_count = node_count;
  created->edge_count = edge_count;
  *session = created;
  clear_error(); return NEUROSEEK_CUDA_OK;
}
extern "C" neuroseek_cuda_status neuroseek_cuda_csr_session_destroy(neuroseek_cuda_csr_session* session) {
  if (!session) { clear_error(); return NEUROSEEK_CUDA_OK; }
  cudaError_t first_error = cudaSuccess;
  const cudaError_t offsets_error = cudaFree(session->offsets);
  if (offsets_error != cudaSuccess) first_error = offsets_error;
  const cudaError_t targets_error = cudaFree(session->targets);
  if (first_error == cudaSuccess && targets_error != cudaSuccess) first_error = targets_error;
  const cudaError_t relations_error = cudaFree(session->relations);
  if (first_error == cudaSuccess && relations_error != cudaSuccess) first_error = relations_error;
  delete session;
  if (first_error != cudaSuccess) return fail(NEUROSEEK_CUDA_RUNTIME_ERROR, "csr_session_destroy", first_error);
  clear_error(); return NEUROSEEK_CUDA_OK;
}
extern "C" neuroseek_cuda_status neuroseek_cuda_csr_session_expand(
    const neuroseek_cuda_csr_session* session, const uint32_t* frontier,
    uint32_t frontier_len, int32_t relation, uint32_t* output,
    uint64_t output_capacity, uint64_t* output_len) {
  if (!session) return invalid("csr_session_expand: session is null");
  if (const auto valid = validate_expand_request(session->node_count, frontier, frontier_len, relation, output, output_capacity, output_len, "csr_session_expand"); valid != NEUROSEEK_CUDA_OK) return valid;
  if (frontier_len == 0) { clear_error(); return NEUROSEEK_CUDA_OK; }
  if (const auto ready = ensure_available(); ready != NEUROSEEK_CUDA_OK) return ready;
  return run_expand(session->offsets, session->targets, session->relations, session->node_count,
                    frontier, frontier_len, relation, output, output_capacity, output_len,
                    "csr_session_expand");
}
extern "C" neuroseek_cuda_status neuroseek_cuda_exact_scores(const float* vectors,const float* query,uint32_t rows,uint32_t dims,float* scores) {
  if(!vectors||!query||!scores||!rows||!dims) return invalid("exact_scores: null or empty input"); if (const auto ready = ensure_available(); ready != NEUROSEEK_CUDA_OK) return ready;
  float *dv=nullptr,*dq=nullptr,*ds=nullptr; cudaError_t e;
  e=copy_to_device(&dv,vectors,(size_t)rows*dims); if(e==cudaSuccess)e=copy_to_device(&dq,query,dims); if(e==cudaSuccess)e=cudaMalloc((void**)&ds,rows*sizeof(float)); if(e!=cudaSuccess){free_d(dv);free_d(dq);free_d(ds);return fail(NEUROSEEK_CUDA_RUNTIME_ERROR,"score allocation/copy",e);}
  score_kernel<<<(rows+255)/256,256>>>(dv,dq,rows,dims,ds); e=cudaGetLastError(); if(e==cudaSuccess)e=cudaDeviceSynchronize(); if(e==cudaSuccess)e=cudaMemcpy(scores,ds,rows*sizeof(float),cudaMemcpyDeviceToHost); free_d(dv);free_d(dq);free_d(ds); if(e!=cudaSuccess)return fail(NEUROSEEK_CUDA_RUNTIME_ERROR,"exact_scores",e); clear_error(); return NEUROSEEK_CUDA_OK;
}
extern "C" neuroseek_cuda_status neuroseek_cuda_topk(const float* scores,uint32_t rows,uint32_t k,uint32_t* indices,float* values) {
  if(!scores||!indices||!values||!rows||!k||k>rows||k>1024)return invalid("topk: invalid shape or k"); if (const auto ready = ensure_available(); ready != NEUROSEEK_CUDA_OK) return ready;
  float *ds=nullptr,*dv=nullptr; uint32_t*di=nullptr; cudaError_t e=copy_to_device(&ds,scores,rows); if(e==cudaSuccess)e=cudaMalloc((void**)&di,k*sizeof(uint32_t));if(e==cudaSuccess)e=cudaMalloc((void**)&dv,k*sizeof(float));if(e!=cudaSuccess){free_d(ds);free_d(di);free_d(dv);return fail(NEUROSEEK_CUDA_RUNTIME_ERROR,"topk allocation/copy",e);} topk_kernel<<<1,1>>>(ds,rows,k,di,dv); e=cudaGetLastError();if(e==cudaSuccess)e=cudaDeviceSynchronize();if(e==cudaSuccess)e=cudaMemcpy(indices,di,k*sizeof(uint32_t),cudaMemcpyDeviceToHost);if(e==cudaSuccess)e=cudaMemcpy(values,dv,k*sizeof(float),cudaMemcpyDeviceToHost);free_d(ds);free_d(di);free_d(dv);if(e!=cudaSuccess)return fail(NEUROSEEK_CUDA_RUNTIME_ERROR,"topk",e); clear_error(); return NEUROSEEK_CUDA_OK;
}
