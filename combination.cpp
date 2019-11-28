#include <bits/stdc++.h>
#define int long long int
using namespace std;

int choose(int *fact, int n, int k) {
  return (n < k) ? 0 : (fact[n] / (fact[k] * fact[n - k]));
}

void combination(int *fact, int n, int k, int m) {
  int nCr = fact[n] / (fact[n - k] * fact[k]);

  if (m > nCr) {
    // printf("[thread: %d] Abort m=%d, nCr=%d\n", m, m, nCr);
    return;
  }

  int idx = 0;
  int a = n, b = k;

//   printf("[m=%d] ", m);
  int x = (choose(fact, n, k) - 1) - m;
  for (int i = 0; i < k; ++i) {
    a = a - 1;

    while (choose(fact, a, b) > x)
      a = a - 1;

    // printf("%d ", n - 1 - a);
    x = x - choose(fact, a, b);
    b = b - 1;
  }
//   printf("\n");

  return;
}

int32_t main() {
  int n = 20, r = 10;

  int *fact = new int[n + 1];
  fact[0] = 1;
  for (int i = 1; i <= n; ++i) {
    fact[i] = i * fact[i - 1];
  }

  int nCr = fact[n] / (fact[n - r] * fact[r]);
  clock_t time_req = clock();
  for (int i = 0; i < nCr; ++i) {
    combination(fact, n, r, i);
  }
  time_req = clock() - time_req;

  printf("Total time: %f ms", ((float)time_req / CLOCKS_PER_SEC) * 1000);

  return 0;
}