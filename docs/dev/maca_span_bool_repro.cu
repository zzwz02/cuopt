// Minimal reproduction of the cuOpt C500 routing crash root cause.
//
// cuOpt's vendored raft::span stores a `cuda::std::span<T> base_` and used to
// construct it with brace-init in the ctor initializer list:
//     span(pointer ptr, size_type count) : base_{ptr, count} {}
// For T=bool this is a *narrowing list-initialization* (bool* -> bool). Newer
// CCCL/CUDA rejects it at compile time; older MACA CCCL compiles it but builds a
// GARBAGE span (wrong .data()/.size()). cuOpt's per-vehicle order_match is a
// span<bool const>, so on C500 it became a wild (host) pointer -> ATU fault.
//
// Build with -DUSE_BRACE to test the original (buggy) brace-init; without it to
// test the fix (parenthesised, direct-init).

#include <cuda/std/span>
#include <cstdio>
#include <type_traits>
#include <cuda_runtime.h>

template <typename T>
struct SpanWrap {
  cuda::std::span<T> base_;
#ifdef USE_BRACE
  __host__ __device__ SpanWrap(T* p, size_t n) : base_{p, n} {}   // original (buggy for bool)
#else
  __host__ __device__ SpanWrap(T* p, size_t n) : base_(p, n) {}   // the fix
#endif
};

template <typename T>
__global__ void probe(T* p, size_t n, void** out_data, unsigned long long* out_size)
{
  SpanWrap<T> s(p, n);
  out_data[0] = (void*)s.base_.data();
  out_size[0] = (unsigned long long)s.base_.size();
}

template <typename T>
int run(const char* tname)
{
  std::remove_const_t<T>* d;
  cudaMalloc(&d, 16);
  void** dd;
  unsigned long long* ds;
  cudaMalloc(&dd, sizeof(void*));
  cudaMalloc(&ds, sizeof(unsigned long long));
  probe<T><<<1, 1>>>((T*)d, 3, dd, ds);
  cudaDeviceSynchronize();
  void* hd;
  unsigned long long hs;
  cudaMemcpy(&hd, dd, sizeof(void*), cudaMemcpyDeviceToHost);
  cudaMemcpy(&hs, ds, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
  bool ok = (hd == (void*)d) && (hs == 3);
  printf("span<%-5s> expected{data=%p size=3}  got{data=%p size=%llu}  -> %s\n",
         tname, (void*)d, hd, hs, ok ? "OK" : "*** WRONG ***");
  cudaFree(d); cudaFree(dd); cudaFree(ds);
  return ok ? 0 : 1;
}

int main()
{
#ifdef USE_BRACE
  printf("=== brace-init  base_{ptr,count}  (cuOpt original raft::span) ===\n");
#else
  printf("=== paren-init  base_(ptr,count)  (the fix) ===\n");
#endif
  int bad = 0;
  bad += run<const int>("cint");
  bad += run<const bool>("cbool");
  bad += run<bool>("bool");
  printf("%s\n", bad ? "RESULT: BUG REPRODUCED (a span is corrupted)" : "RESULT: all spans correct");
  return bad;
}
