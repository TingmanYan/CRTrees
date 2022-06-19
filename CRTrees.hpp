#ifndef _CRTREES_HPP_
#define _CRTREES_HPP_

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iomanip>
#include <iostream>
#include <string>
#include <thrust/execution_policy.h>
#include "util.h"

int
get_num_from_scan(const int*, const int*, const int);

class CRTrees
{
  protected:
    const int size;
    int m_N;

    const int m_block = 32 * 2;
    int m_grid;

    int* in_degree_d;
    int* nnc_d;
    int* nnn_d;
    int* mask_d;
    int* maskc_d;
    int* max_d;

    int* predicate_d;
    int* pos_scan_d;

    bool* jump_d;

  public:
    CRTrees(const int size)
      : size(size)
    {
        cudaMalloc(&in_degree_d, sizeof(int) * size);
        cudaMalloc(&nnc_d, sizeof(int) * size);
        cudaMalloc(&nnn_d, sizeof(int) * size);
        cudaMalloc(&mask_d, sizeof(int) * size);
        cudaMalloc(&maskc_d, sizeof(int) * size);
        cudaMalloc(&max_d, sizeof(int) * size);
        cudaMalloc(&predicate_d, sizeof(int) * size);
        cudaMalloc(&pos_scan_d, sizeof(int) * size);
        cudaMalloc(&jump_d, sizeof(bool));
    }
    ~CRTrees()
    {
        cudaFree(jump_d);
        cudaFree(pos_scan_d);
        cudaFree(predicate_d);
        cudaFree(max_d);
        cudaFree(maskc_d);
        cudaFree(mask_d);
        cudaFree(nnn_d);
        cudaFree(nnc_d);
        cudaFree(in_degree_d);
    }

    void get_clus(int*, int*, const int, int* mask_cycle_d = nullptr);
    int compact_clus_label(int*, const int);

  protected:
    bool get_in_degree(const int*, const int*, int*);
    void de_link_and_compute_mask_h(const int*, int*, int*);
    void pointer_jumping(int*&, const int*, const int*);
    void re_link_h(int*, int*, const int*, const int*);
    void get_cycle_mask(int*, const int*, int*);
    void pointer_jumping_max(int*&, int*&);

    void is_clus_label_h(const int*, int*);
    void remap_clus_label_h(int*, const int*);
    void traverse_cycle_root(int*&, int*);
};

#endif
