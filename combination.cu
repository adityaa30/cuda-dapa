#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#define int long long int

__device__ int Choose(int *fact, int n, int k) {
  return (n < k) ? 0 : (fact[n] / (fact[n - k] * fact[k]));
}

__global__ void PrintCombinations(int *fact, int n, int k, int *output) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int m = tid;
  int nCr = fact[n] / (fact[n - k] * fact[k]);

  if (m >= nCr) {
    // printf("[thread %lld] Abort.\n", tid);
    return;
  }

  int idx = 0;

  int a = n, b = k;
  int x = (Choose(fact, n, k) - 1) - m;
  for (int i = 0; i < k; ++i) {
    a = a - 1;

    while (Choose(fact, a, b) > x) {
      a = a - 1;
    }
    output[m * k + idx++] = n - 1 - a;
    x = x - Choose(fact, a, b);
    b = b - 1;
  }

  //   __syncthreads();
  //   for (int i = 0; i < nCr; ++i) {
  //     __syncthreads();
  //     if (i == tid) {
  //       printf("[m=%lld] ", m);
  //       for (int i = 0; i < idx; ++i) {
  //         printf("%lld ", output[i]);
  //       }
  //       printf("Done.\n");
  //     }
  //   }
}

int32_t main() {
  int n = 20, k = 10;
  //   scanf("%lld %lld", &n, &k);

  int *fact;
  fact = (int *)malloc(sizeof(int) * (n + 1));

  fact[0] = 1;
  for (int i = 1; i <= n; ++i) {
    fact[i] = fact[i - 1] * i;
  }

  printf("Factorial: ");
  for (int i = 0; i <= n; ++i) {
    printf("%lld ", fact[i]);
  }
  printf("\n");

  int nCk = fact[n] / (fact[n - k] * fact[k]);

  int *output = new int[nCk * k];
  int *d_fact, *d_output;
  cudaMalloc((void **)&d_fact, sizeof(int) * (n + 1));
  cudaMalloc((void **)&d_output, sizeof(int) * (nCk * k));
  cudaMemcpy(d_fact, fact, sizeof(int) * (n + 1), cudaMemcpyHostToDevice);
  cudaMemcpy(d_output, output, sizeof(int) * (nCk * k), cudaMemcpyHostToDevice);

  // Executing kernel
  int block_size = 256;
  int grid_size = ((nCk + block_size) / block_size);

  printf("Block size: %lld\n", block_size);
  printf("Grid size: %lld\n", grid_size);

  clock_t time_taken = clock();
  PrintCombinations<<<grid_size, block_size>>>(d_fact, n, k, d_output);
  cudaMemcpy(output, d_output, sizeof(int) * (nCk * k), cudaMemcpyDeviceToHost);

  time_taken = clock() - time_taken;

  printf("Time taken: %f ms\n", ((float)time_taken / CLOCKS_PER_SEC) * 1000);

  for (int i = 0; i < nCk; ++i) {
    printf("[Case #%lld] ", i);
    for (int j = 0; j < k; ++j) {
      printf("%lld ", output[i * k + j]);
    }
    printf("\n");
  }

  cudaFree(d_fact);
  free(fact);
}