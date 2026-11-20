#include <omp.h>
#include <string.h>
#include "prefix_sum.h"

/* Parallel prefix sum using recursive approach. */
static void prefix_sums(long long *a, int start, int end) {
   if (start >= end) return;
   int mid = (start + end)/2;
   #pragma omp parallel num_threads(2)
   {
      #pragma omp single // Ensures ONLY ONE thread queues the tasks
      {
         #pragma omp task shared(a)
            prefix_sums(a, start, mid);
         #pragma omp task shared(a)
            prefix_sums(a, mid + 1, end);
      }
   } // Here,a thread waits. If it sees a queued task, it will
   // run it
   #pragma omp parallel for
   for (int k = mid + 1; k <= end; k++) {
      a[k] = a[k] + a[mid];
   }
}


void parallel_prefix_sum_recur(long long *input, long long *output, long n) {
   if (n <= 0){
      return;
   }
   memcpy(output, input, (size_t)n * sizeof(long long)); //copying input to output
   prefix_sums(output, 0, n - 1);
}