#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "prefix_sum.h"

int main(int argc, char *argv[]){
   //checking the number of arguments and if it is less than 3, print error message and return
   if(argc != 3){
      fprintf(stderr, "Number of argument is insufficient: %s exponent name_of_output_file\n", argv[0]);
      return 1;
   }
   
   //checking if exponent is a valid integer
   int exp = atoi(argv[1]);

   //checking if exponent is less than 2, print error message and return
   if(exp < 2){
      fprintf(stderr, "Exponent should be greater than or equal to 2\n");
      return 1;
   }

   const char *output_file_name = argv[2];

   //opening the output file for writing the results
   FILE *output_file = fopen(output_file_name, "w");
   if(!output_file){
      fprintf(stderr, "Error opening output file: %s\n", output_file_name);
      return 1;                     
   }

   printf("Running the parallel prefix sum with exponent %d \n", exp);

   //looping through the exponents from 2 to input exponent and generating data of 2^x size
   for(long x = 2; x <= exp; x++){
      long n = 1L << x; 
      printf("Generating data (2^%ld)...\n", x);

      long long *input = malloc((size_t)n * sizeof(long long));
      long long *output_recursive = malloc((size_t)n * sizeof(long long));
      long long *output_tree = malloc((size_t)n * sizeof(long long));

      //checking if the memory allocation is succesfull or not
      if(!input || !output_tree || !output_recursive){
         printf("Memory allocation failed\n");
         free(input);
         free(output_tree);
         free(output_recursive);
         fclose(output_file);
         return 1;
      }

      //initializing the input array from 1 to n
      for(long i = 0; i < n; i++){
         input[i] = i + 1;
      }

      double start_time, end_time, recursive_time, tree_time;

      // Measure time for recursive parallel prefix sum
      start_time = omp_get_wtime();
      //call the recursive sum to find the prefix sum
      parallel_prefix_sum_recur(input, output_recursive, n);
      end_time = omp_get_wtime();
      recursive_time = end_time - start_time;
      printf("* parallel_prefix_sum_recur ... time is: %.4f seconds\n", recursive_time);

      // Measure time for tree based parallel prefix sum
      start_time = omp_get_wtime();
      //call the tree based sum to find prefix sum
      parallel_prefix_sum_tree(input, output_tree, n);
      end_time = omp_get_wtime();
      tree_time = end_time - start_time;
      printf("  parallel_prefix_sum_tree ... time is: %.4f seconds\n", tree_time);

      //writing the results to the output file
      fprintf(output_file, "%ld %.5f %.5f\n", n, recursive_time, tree_time);

      free(input);
      free(output_tree);
      free(output_recursive);
   }

   printf("Output file generated. Results written to %s\n", output_file_name);
   fclose(output_file);          
   return 0;
}