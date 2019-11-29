#include <bits/stdc++.h>
using namespace std;

int main() {
  int n = 20;
  int q = ceil(log2(n));
  printf("n=%d q=%d\n", n, q);

  int *Mat = new int[n * n];
  for (int i = 1; i <= n * n; ++i) {
    Mat[i - 1] = i;
  }

  clock_t time_taken = clock();
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      swap(Mat[i * n + j], Mat[j * n + i]);
    }
  }
  printf("Transpose Done.\n");

  time_taken = clock() - time_taken;
  printf("Time taken: %f ms", 1000 * ((float)time_taken / CLOCKS_PER_SEC));
}