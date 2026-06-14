// Replay the EXACT cusparseXcsrsort inputs cuopt dumped (m,n,nnz,offsets,indices)
// to see whether the C500 onesweep ATU fault is caused by the inputs themselves or by
// cuopt's process/graph state. Build:
//   source maca_env.sh
//   /opt/maca/tools/cu-bridge/tools/pre_make nvcc docs/dev/mccub_csrsort_repro.cu \
//       -lcusparse -o /tmp/csrsort_repro
// Run: /tmp/csrsort_repro /tmp/csrsort_000.bin 3   # file, repeat count
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <cusparse.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#define CK(x)                                                       \
  do {                                                              \
    cudaError_t e = (x);                                            \
    if (e) {                                                        \
      printf("CUDA err %s at %d: %s\n", #x, __LINE__, cudaGetErrorString(e)); \
    }                                                               \
  } while (0)

int main(int argc, char** argv)
{
  const char* fn = (argc > 1) ? argv[1] : "/tmp/csrsort_000.bin";
  int reps       = (argc > 2) ? atoi(argv[2]) : 3;
  int fresh_handle = (argc > 3) ? atoi(argv[3]) : 1;  // 1 = create/destroy handle each rep (as cuopt does)

  FILE* f = fopen(fn, "rb");
  if (!f) { printf("cannot open %s\n", fn); return 2; }
  int64_t meta[4];
  fread(meta, sizeof(int64_t), 4, f);
  int m = (int)meta[0], n = (int)meta[1], nnz = (int)meta[2], offlen = (int)meta[3];
  std::vector<int> off(offlen), ind(nnz);
  fread(off.data(), 4, offlen, f);
  fread(ind.data(), 4, nnz, f);
  fclose(f);
  printf("loaded %s: m=%d n=%d nnz=%d offlen=%d fresh_handle=%d reps=%d\n",
         fn, m, n, nnz, offlen, fresh_handle, reps);

  int *d_off, *d_ind, *d_P;
  CK(cudaMalloc(&d_off, (size_t)offlen * 4));
  CK(cudaMalloc(&d_ind, (size_t)nnz * 4));
  CK(cudaMalloc(&d_P, (size_t)nnz * 4));
  CK(cudaMemcpy(d_off, off.data(), (size_t)offlen * 4, cudaMemcpyHostToDevice));

  for (int r = 0; r < reps; r++) {
    // csrsort sorts indices in place; reload each rep so every rep is identical to a fresh solve
    CK(cudaMemcpy(d_ind, ind.data(), (size_t)nnz * 4, cudaMemcpyHostToDevice));
    thrust::sequence(thrust::device, d_P, d_P + nnz);

    cusparseHandle_t h;
    cusparseCreate(&h);
    cusparseMatDescr_t descr;
    cusparseCreateMatDescr(&descr);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);

    size_t bufsz = 0;
    cusparseXcsrsort_bufferSizeExt(h, m, n, nnz, d_off, d_ind, &bufsz);
    void* buf = nullptr;
    CK(cudaMalloc(&buf, bufsz));

    cusparseStatus_t st = cusparseXcsrsort(h, m, n, nnz, descr, d_off, d_ind, d_P, buf);
    cudaError_t e       = cudaDeviceSynchronize();
    printf("rep %d: cusparse_status=%d  sync=%s  bufsz=%zu\n", r, (int)st, cudaGetErrorString(e), bufsz);

    cudaFree(buf);
    cusparseDestroyMatDescr(descr);
    cusparseDestroy(h);
    if (e) {
      printf(">>> FAULTED at rep %d\n", r);
      return 1;
    }
  }
  printf("all %d reps OK\n", reps);
  return 0;
}
