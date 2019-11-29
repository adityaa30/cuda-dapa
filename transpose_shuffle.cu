#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void ShuffleTranspose(int *Mat, int n, int q) {
  // n = 2^q
  printf("[thread %d] Start.\n", threadIdx.x);
  for (int i = 1; i <= q; ++i) {
    int k = threadIdx.x;
    int val_k = Mat[k];

    __syncthreads();
    if(k >= 1 && k <= (n * n - 2)) {
      Mat[(2 * k) % (n * n - 1)] = val_k;
    }
  }
}

__host__ void PrintMatrix(int *Mat, int n, bool transpose) {
  if (transpose)
    printf("Transposed Matrix:\n");
  else
    printf("Original Matrix:\n");

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      printf("%d\t", Mat[i * n + j]);
    }
    printf("\n");
  }
}

int main() {
  int n = 32;
  int q = ceil(log2(n));
  printf("n=%d q=%d\n", n, q);

  int *Mat = new int[n * n];
  for (int i = 1; i <= n * n; ++i) {
    Mat[i - 1] = i;
  }

  int *d_Mat;
  cudaMalloc((void **)&d_Mat, sizeof(int) * (n * n));
  cudaMemcpy(d_Mat, Mat, sizeof(int) * (n * n), cudaMemcpyHostToDevice);

  dim3 block_size(n * n);
  dim3 grid_size(1);

  PrintMatrix(Mat, n, 0);

  clock_t time_taken = clock();
  ShuffleTranspose<<<grid_size, block_size>>>(d_Mat, n, q);
  cudaMemcpy(Mat, d_Mat, sizeof(int) * (n * n), cudaMemcpyDeviceToHost);
  printf("Transpose Done.\n");

  PrintMatrix(Mat, n, 1);

  time_taken = clock() - time_taken;
  printf("Time taken: %f ms", 1000 * ((float)time_taken/CLOCKS_PER_SEC));
  
  cudaFree(d_Mat);
  free(Mat);
}