#include <cuda.h>
#include <cuda_runtime.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

#define DEBUG false
#define NULL_DEFAULT -1

__device__ int x[] = {0, -1, 0, 1};
__device__ int y[] = {-1, 0, 1, 0};

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

/**
1. A(i, j) is used to store aij initially and aji when the algorithm terminates;
2. B(i, j) is used to store data received from P(i, j + 1) or P(i - 1, j), that
is, from its right or top neighbors; and
3. C(i, j) is used to store data received from P(i, j - 1) or P(i + 1, j), that
is, from its left or bottom neighbors.
*/

typedef struct {
  int a_km = NULL_DEFAULT, m = -1, k = -1;
  bool isNull = true;
} Data;

__device__ void CopyData(Data &dest, Data &src) {
  dest.a_km = src.a_km;
  dest.m = src.m;
  dest.k = src.k;
  dest.isNull = false;
}

__device__ void PrintData(Data &data) {
  printf("(%d, %d, %d %d)\t", data.a_km, data.m, data.k, data.isNull ? 1 : 0);
}

__device__ void MakeDataNull(Data &data) {
  data.isNull = true;
  data.m = -1;
  data.k = -1;
  data.a_km = NULL_DEFAULT;
}

__device__ void PrintDebugMatrix(int *A, Data *B, Data *C, int n, float step) {
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    printf("Debug matrix: Step #%1.1f\n", step);
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        int idx = i * n + j;
        int val_b = B[idx].isNull ? NULL_DEFAULT : B[idx].a_km;
        int val_c = C[idx].isNull ? NULL_DEFAULT : C[idx].a_km;
        printf("(%d, %d, %d)\t", A[idx], val_b, val_c);
      }
      printf("\n");
    }
  }
}

__device__ bool isValid(int x, int y, int n) {
  return (0 <= x && x < n && 0 <= y && y < n);
}

__global__ void MeshTranspose(int *A, Data *B, Data *C, int n) {
  // printf("[thread (%d, %d)] Start.\n", threadIdx.x, threadIdx.y);

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
	
	if(i >= n || j >= n) return;

	int i_j = i * n + j;

  MakeDataNull(B[i_j]);
  MakeDataNull(C[i_j]);

  // Step 1.1
  if (1 <= i && i < n && 0 <= j && j < i) {
    C[(i - 1) * n + j] = {A[i_j], j, i, false};
  }

  // Step 1.2
  if (0 <= i && i < n - 1 && i < j && j < n) {
    B[i * n + (j - 1)] = {A[i_j], j, i, false};
  }

  if (DEBUG)
    PrintDebugMatrix(A, B, C, n, 1.0f);

  for (int step = 0; step <= (2 * n + 3); ++step) {
    // printf("[thread (%d, %d)] Step %d.\n", threadIdx.x, threadIdx.y, step);

    // Declare local variables to implement message passing
    int temp_A = NULL_DEFAULT;
    bool got_A_from_B = false;
    Data *temp_B = new Data[4];
    Data *temp_C = new Data[4];

    // Step 2.1
    if (1 <= i && i < n && 0 <= j && j < i) {
      // (a_km, m, k) is received from P(i + 1, j)
      // send it to P(i - 1, j)
      CopyData(temp_C[1], C[i_j]);
      // (a_km, m, k) is received from P(i - 1, j)
      if (B[i_j].m == i && B[i_j].k == j) {
        // A(i, j) <- a_km {a_km has reached its destination}
        temp_A = B[i_j].a_km;
        got_A_from_B = true;
      } else {
        // Send (a_km, m, k) to P(i + 1, j)
        CopyData(temp_B[3], B[i_j]);
      }
    }

    // Step 2.2
    int i_i = i * n + i;
    if (0 <= i && i < n && i == j) {
      // (a_km, m, k) is received from P(i + 1, i)
      // Send it to P(i, i + 1)
      CopyData(temp_C[2], C[i_i]);

      // (a_km, m, k) is received from P(i, i + 1)
      // Send it to P(i + 1, i)
      CopyData(temp_B[3], B[i_i]);
    }

    // Step 2.3
    if (0 <= i && i < n - 1 && i < j && j < n) {
      // (a_km, m, k) is received from P(i, j + 1)
      // send it to P(i, j - 1)
      CopyData(temp_B[0], B[i_j]);
      // (a_km, m, k) is received from P(i, j - 1)
      if (C[i_j].m == i && C[i_j].k == j) {
        // A(i, j) <- a_km {a_km has reached its destination}
        temp_A = C[i_j].a_km;
        got_A_from_B = false;
      } else {
        // Send (a_km, m, k) to P(i, j + 1)
        CopyData(temp_C[2], C[i_j]);
      }
    }

		__syncthreads();
    // Copy the final state values now
    // Below section only deals with writing of data
    if (temp_A != NULL_DEFAULT) {
      A[i_j] = temp_A;
      if (got_A_from_B)
        MakeDataNull(B[i_j]);
      else
        MakeDataNull(C[i_j]);
    }
    for (int next = 0; next < 4; ++next) {
      int new_x = i + x[next];
      int new_y = j + y[next];
      int idx = new_x * n + new_y;
      if (isValid(new_x, new_y, n)) {
        if (temp_B[next].isNull == false) {
          CopyData(B[idx], temp_B[next]);
        }
        if (temp_C[next].isNull == false) {
          CopyData(C[idx], temp_C[next]);
        }
      }
    }

    // Now make null for the last row in C 2D-Array
    if (i == n - 1) {
      MakeDataNull(C[i_j]);
    }

    // Now make null for the last column in B 2D-Array
    if (j == n - 1) {
      MakeDataNull(B[i_j]);
    }

    if (DEBUG)
      PrintDebugMatrix(A, B, C, n, 2.0f + step / 10.0f);
  }
}

int main() {
  int n = 20;
  printf("n=%d\n", n);

  int *Mat = new int[n * n];
  int val = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      Mat[i * n + j] = ++val;
    }
  }

  int *d_A;
  Data *d_B, *d_C;
  cudaMalloc((void **)&d_A, sizeof(int) * (n * n));
  cudaMalloc((void **)&d_B, sizeof(Data) * (n * n));
  cudaMalloc((void **)&d_C, sizeof(Data) * (n * n));
  cudaMemcpy(d_A, Mat, sizeof(int) * (n * n), cudaMemcpyHostToDevice);

  dim3 block_size(32, 32);
  dim3 grid_size(1);

  if (DEBUG)
    PrintMatrix(Mat, n, 0);

  clock_t time_taken = clock();
  MeshTranspose<<<grid_size, block_size>>>(d_A, d_B, d_C, n);
  cudaMemcpy(Mat, d_A, sizeof(int) * (n * n), cudaMemcpyDeviceToHost);
  time_taken = clock() - time_taken;

  printf("Transpose Done.\n");

  if (DEBUG)
    PrintMatrix(Mat, n, 1);
  printf("Time taken: %f ms\n", 1000 * ((float)time_taken / CLOCKS_PER_SEC));

	// Testing if output is correct!
	bool ok = true;
	val = 0;
	for(int j = 0; j < n; ++j) {
		for(int i = 0; i < n; ++i) {
       if(Mat[i *n + j] != (++val)) {
				 ok = false;
			 }
		}
	}

	if(ok) printf("Test: OK\n");
	else {
		printf("Test: FAIL\n");
    PrintMatrix(Mat, n, 1);
	}

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  free(Mat);
}