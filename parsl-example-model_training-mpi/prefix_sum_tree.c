#include <omp.h>
#include <math.h>
#include "prefix_sum.h"

//Parallel prefix sum using up sweep and down sweep approach
static void up_sweep(long long *array, long n) {
   long levels = (long)log2(n);

   for (long d = 0; d < levels; d++) {
      long step = 1L << (d + 1);
      long half_step = 1L << d;

      #pragma omp for schedule(static)
      for (long k = 0; k < n; k += step) {
         array[k + step - 1] = array[k + half_step - 1] + array[k + step - 1];
      }
   }
}

static void down_sweep(long long *array, long n) {
   long levels = (long)log2(n);

   for (long d = levels - 1; d >= 0; d--) {
      long step = 1L << (d + 1);
      long half_step = 1L << d;

      #pragma omp for schedule(static)
      for (long k = 0; k < n; k += step) {
         long long temp = array[k + half_step - 1];
         array[k + half_step - 1] = array[k + step - 1];
         array[k + step - 1] += temp;
      }
   }
}

static void prefix_sum(long long *array, long n){
   up_sweep(array, n);
   #pragma omp single
   {
      array[n - 1] = 0; // Set the last element to zero for down-sweep
   }
   down_sweep(array, n);
}


void parallel_prefix_sum_tree(long long *input, long long *output, long n) {
   if(n <= 0) {
      return;
   }
   // use 4 threads for parallel region
   #pragma omp parallel num_threads(4)
   {
      #pragma omp for schedule(static)
      for (long i = 0; i < n; i++) {
         output[i] = input[i];  // Copy input to output
      }

      prefix_sum(output, n); //implementing exclusive Blelloch scan

      #pragma omp for schedule(static)
      for (long i = 0; i < n; i++) {
         output[i] += input[i];  // converting exclusive to inclusive scan
      }
   }
}