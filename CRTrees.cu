/*
 *  Find connected components in a direct 1-nn graph using parallel CRTrees labeling with O(N) memory and O(log(N)) steps for the worst case.
 *  Author: Tingman Yan (tmyann@outlook.com)
 */

#include "CRTrees.hpp"

// TODO: re-write this code using dynamic parallism
//       which may reduce the latency bewteen CPU and GPU

int
get_num_from_scan(const int* const predicate_d,
                  const int* const pos_scan_d,
                  const int size)
{
    int num_h = 0;
    cudaMemcpy(
      &num_h, &pos_scan_d[size - 1], sizeof(int), cudaMemcpyDeviceToHost);
    int is_last_h = 0;
    cudaMemcpy(
      &is_last_h, &predicate_d[size - 1], sizeof(int), cudaMemcpyDeviceToHost);
    num_h += is_last_h;
    return num_h;
}

__global__ void
compute_in_degree(const int* const nn_d,
                  const int* const mask_d,
                  int* const in_degree_d,
                  const int N)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= N || !mask_d[index])
        return;

    // since in-degree has an upper bound and is small (4/8 in grid graph)
    // atomicAdd would be effcient enough
    int nn = nn_d[index];
    atomicAdd(&in_degree_d[nn], 1);
}
__global__ void
check_larger_than_one(const int* const in_degree_d,
                      const int* const mask_d,
                      const int N,
                      bool* jump_d)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= N || !mask_d[index])
        return;

    if (in_degree_d[index] > 1)
        *jump_d = true;
}
// de-link and compute new mask
// mask out the nodes with in-degree larger than 1
__global__ void
de_link_and_compute_mask(const int* const in_degree_d,
                         int* const nnc_d,
                         int* const mask_d,
                         const int N)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= N)
        return;

    if (in_degree_d[index] > 1) {
        // make the masked node point to itself
        nnc_d[index] = index;
        mask_d[index] = 1;
    }
}
// TODO: one_step_jump consume most time (26%)
__global__ void
one_step_jump(int* const nnn_d,
              const int* const nnc_d,
              const int* const maskc_d,
              const int* const mask_d,
              const int N,
              bool* jump_d)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= N || !maskc_d[index])
        return;

    // pointer jumping
    int nn = nnc_d[index];
    int nnn = nnc_d[nn];
    nnn_d[index] = nnn;

    // if there exists node that jump into the marked node, go on jumping
    if (!mask_d[nn] && mask_d[nnn])
        *jump_d = true;
}
__global__ void
re_link(int* const nnc_d,
        const int* const nn_d,
        const int* const mask_d,
        const int* const maskc_d,
        const int N)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= N)
        return;

    if (mask_d[index]) {
        int nn = nn_d[index];
        nnc_d[index] = nn;
        if (!mask_d[nn])
            nnc_d[index] = nnc_d[nn];
    } else if (!maskc_d[index]) {
        int nn = nnc_d[index];
        if (maskc_d[nn])
            nnc_d[index] = nnc_d[nn];
    }
}
__global__ void
re_link_all(int* const nn_d,
            const int* const nnc_d,
            const int* const mask_d,
            const int N)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= N)
        return;

    int nn = nnc_d[index];
    if (mask_d[index]) {
        nn_d[index] = nn;
    } else if (mask_d[nn]) {
        int nnn = nnc_d[nn];
        nn_d[index] = nnn;
    }
}
__global__ void
is_clus_label(const int*, int*, const int);
__global__ void
one_step_jump_for_cycle_mask(int* nn_d,
                             int* nnn_d,
                             int* mask_d,
                             const int N,
                             bool* jump_d)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= N)
        return;

    int nn = nn_d[index];
    int nnn = nn_d[nn];
    nnn_d[index] = nnn;

    if (mask_d[nn] && !mask_d[nnn]) {
        mask_d[nnn] = 1;
        *jump_d = true;
    }
}
__global__ void
one_step_jump_max(int* nnc_d,
                  int* nnn_d,
                  int* max_d,
                  int* clus_d,
                  const int N,
                  bool* jump_d)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= N)
        return;

    int nn = nnc_d[index];
    int maxc = max_d[index];
    clus_d[index] = max(maxc, max_d[nn]);
    nnn_d[index] = nnc_d[nn];

    if (clus_d[index] != maxc)
        *jump_d = true;
}
void
CRTrees::traverse_cycle_root(int*& nn_d, int* const mask_d)
{
    int iter = 0;
    bool jump_h = true;
    while (jump_h) {
        cudaMemset(jump_d, 0, sizeof(bool));
        one_step_jump_for_cycle_mask<<<m_grid, m_block>>>(
          nn_d, nnn_d, mask_d, m_N, jump_d);
        cudaMemcpy(&jump_h, jump_d, sizeof(bool), cudaMemcpyDeviceToHost);
        // exchange pointer
        std::swap(nn_d, nnn_d);
        iter++;
    }
    // exchange back
    if (iter % 2)
        std::swap(nn_d, nnn_d);
}
bool
CRTrees::get_in_degree(const int* const nn_d,
                       const int* const mask_d,
                       int* const in_degree_d)
{
    cudaMemset(in_degree_d, 0, sizeof(int) * m_N);
    compute_in_degree<<<m_grid, m_block>>>(nn_d, mask_d, in_degree_d, m_N);

    // check if there exists a node with in-degree larger than one
    bool jump_h = false;
    cudaMemset(jump_d, 0, sizeof(bool));
    check_larger_than_one<<<m_grid, m_block>>>(
      in_degree_d, mask_d, m_N, jump_d);
    cudaMemcpy(&jump_h, jump_d, sizeof(bool), cudaMemcpyDeviceToHost);

    return jump_h;
}
void
CRTrees::de_link_and_compute_mask_h(const int* const in_degree_d,
                                    int* const nnc_d,
                                    int* const mask_d)
{
    cudaMemset(mask_d, 0, sizeof(int) * m_N);
    de_link_and_compute_mask<<<m_grid, m_block>>>(
      in_degree_d, nnc_d, mask_d, m_N);
}
// perform pointer jumping till converge
// the convergence is defined as: till no pointer jump to the marked nodes
void
CRTrees::pointer_jumping(int*& nnc_d,
                         const int* const maskc_d,
                         const int* const mask_d)
{
    int iter = 0;
    bool jump_h = true;
    while (jump_h) {
        cudaMemset(jump_d, 0, sizeof(bool));
        one_step_jump<<<m_grid, m_block>>>(
          nnn_d, nnc_d, maskc_d, mask_d, m_N, jump_d);
        cudaMemcpy(&jump_h, jump_d, sizeof(bool), cudaMemcpyDeviceToHost);
        // std::cout << "pointer jumping flag: " << jump_h << std::endl;
        std::swap(nnn_d, nnc_d);
        iter++;
    }
    // exchange back
    if (iter % 2)
        std::swap(nnn_d, nnc_d);
}
void
CRTrees::re_link_h(int* const nn_d,
                   int* const nnc_d,
                   const int* const mask_d,
                   const int* const maskc_d)
{
    re_link<<<m_grid, m_block>>>(nnc_d, nn_d, mask_d, maskc_d, m_N);
    re_link_all<<<m_grid, m_block>>>(nn_d, nnc_d, mask_d, m_N);
}
void
CRTrees::get_cycle_mask(int* nn_d, const int* const nnc_d, int* const mask_d)
{
    // first init mask_d, then traverse
    cudaMemset(mask_d, 0, sizeof(int) * m_N);
    is_clus_label<<<m_grid, m_block>>>(nnc_d, mask_d, m_N);

    //  now mask_d records seeds vertices in the cycle-roots
    //  the traverse starts from the seeds till stop
    //  only vertices in the cycle roots will be traversed by the structure of
    //  the crtrees
    traverse_cycle_root(nn_d, mask_d);
}
void
CRTrees::pointer_jumping_max(int*& nnc_d, int*& clus_d)
{
    cudaMemcpy(max_d, nnc_d, sizeof(int) * m_N, cudaMemcpyDeviceToDevice);

    int iter = 0;
    bool jump_h = true;
    while (jump_h) {
        cudaMemset(jump_d, 0, sizeof(bool));
        one_step_jump_max<<<m_grid, m_block>>>(
          nnc_d, nnn_d, max_d, clus_d, m_N, jump_d);
        cudaMemcpy(&jump_h, jump_d, sizeof(bool), cudaMemcpyDeviceToHost);
        // std::cout << "pointer jumping max flag: " << jump_h << std::endl;
        // ping-pang values
        std::swap(nnc_d, nnn_d);
        std::swap(max_d, clus_d);
        iter++;
    }
    // exchange back
    if (iter % 2) {
        std::swap(nnc_d, nnn_d);
        std::swap(max_d, clus_d);
    }
}

/* while there exist node with in-degree larger than 1
 * 1. make nodes with in-degree larger than 1 point to themselves, and record
 *    their original successors
 * 2. perform pointer jumping till converge
 *    the convergence is defined as: till no pointer jump to the marked nodes
 * 3. merge the clusters to a new list and keep the record of the map
 *    relation, i.e. shrink the cluster index
 * 4. re-linking using the recorded successors
 * After the above operation, the obtained graph consist of multiple disjoint
 * cycle sub-graphs.
 * Perform pointer jumping with maximum operation to label each cycle
 * sub-graph as the maximum index of its nodes.
 */

void
CRTrees::get_clus(int* nn_d, int* clus_d, const int N, int* mask_cycle_d)
{
    assert(N <= size);
    m_N = N;
    m_grid = (N + m_block - 1) / m_block;
    cudaMemset(mask_d, 1, sizeof(int) * m_N);
    cudaMemcpy(nnc_d, nn_d, sizeof(int) * m_N, cudaMemcpyDeviceToDevice);
    cudaMemcpy(clus_d, nn_d, sizeof(int) * m_N, cudaMemcpyDeviceToDevice);

    // while there is node with in-degree larger than 1
    while (get_in_degree(nnc_d, mask_d, in_degree_d)) {
        cudaMemcpy(
          maskc_d, mask_d, sizeof(int) * m_N, cudaMemcpyDeviceToDevice);
        de_link_and_compute_mask_h(in_degree_d, nnc_d, mask_d);

        pointer_jumping(nnc_d, maskc_d, mask_d);

        re_link_h(nn_d, nnc_d, mask_d, maskc_d);

        cudaMemcpy(nnc_d, nn_d, sizeof(int) * m_N, cudaMemcpyDeviceToDevice);
    }

    // keep nn_d unchanged
    cudaMemcpy(nn_d, clus_d, sizeof(int) * m_N, cudaMemcpyDeviceToDevice);
    // mask_cycle_d stores the vertices of the cycle-roots
    if (mask_cycle_d) {
        get_cycle_mask(nn_d, nnc_d, mask_cycle_d);
    }

    pointer_jumping_max(nnc_d, clus_d);
}

__global__ void
is_clus_label(const int* const clus_d, int* const predicate_d, const int N)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= N)
        return;

    int cc = clus_d[index];
    // write conflict has no affect to the result
    predicate_d[cc] = 1;
}
__global__ void
remap_clus_label(int* const clus_d, const int* const pos_scan_d, const int N)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= N)
        return;

    int cc = clus_d[index];
    clus_d[index] = pos_scan_d[cc];
}
void
CRTrees::is_clus_label_h(const int* const clus_d, int* const predicate_d)
{
    cudaMemset(predicate_d, 0, sizeof(int) * m_N);
    is_clus_label<<<m_grid, m_block>>>(clus_d, predicate_d, m_N);
}
void
CRTrees::remap_clus_label_h(int* const clus_d, const int* const pos_scan_d)
{
    remap_clus_label<<<m_grid, m_block>>>(clus_d, pos_scan_d, m_N);
}

int
CRTrees::compact_clus_label(int* const clus_d, const int N)
{
    assert(N <= size);
    m_N = N;
    m_grid = (N + m_block - 1) / m_block;

    is_clus_label_h(clus_d, predicate_d);

    thrust::exclusive_scan(
      thrust::device, predicate_d, predicate_d + m_N, pos_scan_d);

    remap_clus_label_h(clus_d, pos_scan_d);

    int num_clus_h = get_num_from_scan(predicate_d, pos_scan_d, m_N);

    return num_clus_h;
}
