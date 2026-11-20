#ifndef PREFIX_SUM_H
#define PREFIX_SUM_H

void parallel_prefix_sum_tree(long long *input, long long *output, long n);
void parallel_prefix_sum_recur(long long *in, long long *out, long n);
#endif