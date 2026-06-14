// Minimal repro: does mccub's onesweep radix sort (cub::DeviceRadixSort::SortPairs,
// which dispatches to DeviceRadixSortOnesweepKernelLarge for large inputs) fault on
// C500 when its caller-provided temp storage is uninitialized/garbage?
//
// In cuopt, cusparseXcsrsort -> mcsparse -> mccub onesweep traps (Xnack/ATU) on the
// 2nd LP solve in a process, because rmm hands back the prior solve's freed bytes as
// the sort scratch. This isolates that: same SortPairs, with the temp buffer memset to
// `scratch_fill` (0 = zeroed like a fresh 1st solve; 0xff = garbage like a 2nd solve).
//
// Build (C500): source maca_env.sh; /opt/maca/tools/cu-bridge/tools/pre_make nvcc \
//                 docs/dev/mccub_onesweep_repro.cu -o /tmp/mccub_repro
// Run:   /tmp/mccub_repro 0      # zeroed scratch
//        /tmp/mccub_repro 255    # garbage scratch
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cub/cub.cuh>

__global__ void init_kernel(uint32_t* keys, uint32_t* vals, int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    keys[i] = 1103515245u * (uint32_t)i + 12345u;  // scrambled, so the sort does work
    vals[i] = (uint32_t)i;
  }
}

int main(int argc, char** argv)
{
  int scratch_fill = (argc > 1) ? atoi(argv[1]) : 0;
  int n            = (argc > 2) ? atoi(argv[2]) : 8000000;

  uint32_t *d_keys_in, *d_keys_out, *d_vals_in, *d_vals_out;
  cudaMalloc(&d_keys_in, (size_t)n * 4);
  cudaMalloc(&d_keys_out, (size_t)n * 4);
  cudaMalloc(&d_vals_in, (size_t)n * 4);
  cudaMalloc(&d_vals_out, (size_t)n * 4);
  init_kernel<<<(n + 255) / 256, 256>>>(d_keys_in, d_vals_in, n);
  cudaDeviceSynchronize();

  size_t temp_bytes = 0;
  cub::DeviceRadixSort::SortPairs(
    nullptr, temp_bytes, d_keys_in, d_keys_out, d_vals_in, d_vals_out, n);
  void* d_temp = nullptr;
  cudaMalloc(&d_temp, temp_bytes);
  cudaMemset(d_temp, scratch_fill, temp_bytes);  // <-- the variable under test
  cudaDeviceSynchronize();

  cub::DeviceRadixSort::SortPairs(
    d_temp, temp_bytes, d_keys_in, d_keys_out, d_vals_in, d_vals_out, n);
  cudaError_t e = cudaDeviceSynchronize();
  printf("n=%d scratch_fill=0x%02x temp_bytes=%zu  result=%s\n",
         n,
         (unsigned)scratch_fill & 0xff,
         temp_bytes,
         cudaGetErrorString(e));

  if (e == cudaSuccess) {
    uint32_t* h = (uint32_t*)malloc((size_t)n * 4);
    cudaMemcpy(h, d_keys_out, (size_t)n * 4, cudaMemcpyDeviceToHost);
    bool sorted = true;
    for (int i = 1; i < n; i++)
      if (h[i - 1] > h[i]) {
        sorted = false;
        break;
      }
    printf("  sorted_correct=%d\n", sorted);
    free(h);
  }
  return e != cudaSuccess;
}
