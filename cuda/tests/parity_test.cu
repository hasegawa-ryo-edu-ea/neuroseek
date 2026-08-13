#include "neuroseek_cuda.h"
#include <cmath>
#include <cstdio>
#include <vector>
int main() {
  int devices=0; if(neuroseek_cuda_device_count(&devices)!=NEUROSEEK_CUDA_OK || devices<1) { std::fprintf(stderr,"CUDA unavailable: %s\n",neuroseek_cuda_last_error()); return 77; }
  uint64_t offsets[]={0,2,3,3}; uint32_t targets[]={1,2,2}; uint16_t rels[]={7,8,7}; uint32_t frontier[]={0,1}, got[4]={}; uint64_t count=0;
  if(neuroseek_cuda_expand(offsets,targets,rels,3,3,frontier,2,7,got,4,&count)!=NEUROSEEK_CUDA_OK || count!=2 || got[0]!=1 || got[1]!=2) { std::fprintf(stderr,"expand parity failed: %s\n",neuroseek_cuda_last_error()); return 1; }
  uint32_t invalid_frontier[]={3};
  if(neuroseek_cuda_expand(offsets,targets,rels,3,3,invalid_frontier,1,7,got,4,&count)!=NEUROSEEK_CUDA_INVALID_ARGUMENT) { std::fprintf(stderr,"out-of-range frontier was accepted\n"); return 5; }
  // Empty work and a zero-capacity size probe are valid, deterministic API
  // cases. The latter must disclose rather than truncate its result.
  if(neuroseek_cuda_expand(offsets,targets,rels,3,3,nullptr,0,7,nullptr,0,&count)!=NEUROSEEK_CUDA_OK || count!=0) return 6;
  if(neuroseek_cuda_expand(offsets,targets,rels,3,3,frontier,2,-1,nullptr,0,&count)!=NEUROSEEK_CUDA_OUTPUT_CAPACITY || count!=3) return 7;
  uint64_t bad_offsets[]={0,3,2,3};
  if(neuroseek_cuda_expand(bad_offsets,targets,rels,3,3,frontier,2,7,got,4,&count)!=NEUROSEEK_CUDA_INVALID_ARGUMENT) return 8;
  neuroseek_cuda_csr_session* session=nullptr;
  if(neuroseek_cuda_csr_session_create(offsets,targets,rels,3,3,&session)!=NEUROSEEK_CUDA_OK || !session) { std::fprintf(stderr,"session create failed: %s\n", neuroseek_cuda_last_error()); return 9; }
  if(neuroseek_cuda_csr_session_expand(session,frontier,2,7,got,4,&count)!=NEUROSEEK_CUDA_OK || count!=2 || got[0]!=1 || got[1]!=2) { std::fprintf(stderr,"session expand parity failed: %s\n", neuroseek_cuda_last_error()); neuroseek_cuda_csr_session_destroy(session); return 10; }
  if(neuroseek_cuda_csr_session_destroy(session)!=NEUROSEEK_CUDA_OK) return 11;
  if(neuroseek_cuda_csr_session_expand(nullptr,frontier,2,7,got,4,&count)!=NEUROSEEK_CUDA_INVALID_ARGUMENT) return 12;
  float vectors[]={1,0, 0,1, 1,1}; float query[]={1,0.5f}; float scores[3]; if(neuroseek_cuda_exact_scores(vectors,query,3,2,scores)!=NEUROSEEK_CUDA_OK) return 2;
  float expected[]={1.f,.5f,1.5f}; for(int i=0;i<3;i++) if(std::fabs(scores[i]-expected[i])>1e-5f) return 3;
  uint32_t idx[2]; float val[2]; if(neuroseek_cuda_topk(scores,3,2,idx,val)!=NEUROSEEK_CUDA_OK || idx[0]!=2 || idx[1]!=0) return 4;
  std::puts("CUDA parity passed"); return 0;
}
