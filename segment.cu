/*
 *  Hierarchical Image Segmentation by parallel CRTrees labeling.
 *  The step-complexity is O(log(N)) and the work-complexity is O(N) in average.
 *  The memory required is O(N).
 *  Author: Tingman Yan (tmyann@outlook.com)
 */

#include "segment.hpp"
#include <cfloat>

/*
 * distance function for pixels
 */
__forceinline__ __device__ float
distL2(const float4& s, const float4& n)
{
    float dist_3d = (s.x - n.x) * (s.x - n.x) + (s.y - n.y) * (s.y - n.y) +
                    (s.z - n.z) * (s.z - n.z);

    float h_x = (s.x - n.x) * 1.f;
    dist_3d += h_x > 0.f ? h_x : 0.f;
    return dist_3d;
}
/*
 * distance function for superpixels
 * per-channel calculation, can be reduced by atomicAdd
 */
__forceinline__ __device__ float
distL2(const float& s, const float& n, const int& c)
{
    float dist = (s - n) * (s - n);
    if (c == 0) {
        float h = (s - n) * 1.f;
        dist += h > 0.f ? h : 0.f;
    }
    return dist;
}
/*
 * help funciton
 * find the nearest neighbor and coresponding distance for a pixel
 */
__forceinline__ __device__ void
find_minimum(int& minimum_id,
             float& minimum,
             const float& dist,
             const int& index)
{
    if (dist < minimum) {
        minimum = dist;
        minimum_id = index;
    }
}
/*
 * search the nearest neighbor in the image grid
 * can initialize with 4-neighbor or 8-neighbor grids
 * 8-neighbor gives higher BR, but worser UE performance
 */
__global__ void
compute_1nn_grid_k(const float4* const img_f4_d,
                   int* const nn_d,
                   int2* const bd_d,
                   float* const dist_d,
                   float* const dist_min_d,
                   const int num_nb,
                   const int width,
                   const int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = y * width + x;
    float4 val = img_f4_d[index];

    float minimum = FLT_MAX;
    int minimum_id = 0;

    if (x >= width || y >= height)
        return;

    // unroll the loop runs faster, although this is redundancy
    if (y > 0) {
        int index_u = index - width;
        float4 val_u = img_f4_d[index_u];
        float dist = distL2(val, val_u);
        find_minimum(minimum_id, minimum, dist, index_u);
        bd_d[num_nb * index].y = index_u;
        if (dist_d)
            dist_d[num_nb * index] = dist;
    } else
        bd_d[num_nb * index].y = index;
    __syncthreads();

    if (y < height - 1) {
        int index_d = index + width;
        float4 val_d = img_f4_d[index_d];
        float dist = distL2(val, val_d);
        find_minimum(minimum_id, minimum, dist, index_d);
        bd_d[num_nb * index + 1].y = index_d;
        if (dist_d)
            dist_d[num_nb * index + 1] = dist;
    } else
        bd_d[num_nb * index + 1].y = index;
    __syncthreads();

    if (x > 0) {
        int index_l = index - 1;
        float4 val_l = img_f4_d[index_l];
        float dist = distL2(val, val_l);
        find_minimum(minimum_id, minimum, dist, index_l);
        bd_d[num_nb * index + 2].y = index_l;
        if (dist_d)
            dist_d[num_nb * index + 2] = dist;
    } else
        bd_d[num_nb * index + 2].y = index;
    __syncthreads();

    if (x < width - 1) {
        int index_r = index + 1;
        float4 val_r = img_f4_d[index_r];
        float dist = distL2(val, val_r);
        find_minimum(minimum_id, minimum, dist, index_r);
        bd_d[num_nb * index + 3].y = index_r;
        if (dist_d)
            dist_d[num_nb * index + 3] = dist;
    } else
        bd_d[num_nb * index + 3].y = index;
    __syncthreads();

    if (num_nb == 8) {
        if (y > 0 && x > 0) {
            int index_ul = index - width - 1;
            float4 val_ul = img_f4_d[index_ul];
            float dist = distL2(val, val_ul);
            find_minimum(minimum_id, minimum, dist, index_ul);
            bd_d[num_nb * index + 4].y = index_ul;
            if (dist_d)
                dist_d[num_nb * index + 4] = dist;
        } else
            bd_d[num_nb * index + 4].y = index;
        __syncthreads();

        if (y > 0 && x < width - 1) {
            int index_ur = index - width + 1;
            float4 val_ur = img_f4_d[index_ur];
            float dist = distL2(val, val_ur);
            find_minimum(minimum_id, minimum, dist, index_ur);
            bd_d[num_nb * index + 5].y = index_ur;
            if (dist_d)
                dist_d[num_nb * index + 5] = dist;
        } else
            bd_d[num_nb * index + 5].y = index;
        __syncthreads();

        if (y < height - 1 && x < width - 1) {
            int index_dr = index + width + 1;
            float4 val_dr = img_f4_d[index_dr];
            float dist = distL2(val, val_dr);
            find_minimum(minimum_id, minimum, dist, index_dr);
            bd_d[num_nb * index + 6].y = index_dr;
            if (dist_d)
                dist_d[num_nb * index + 6] = dist;
        } else
            bd_d[num_nb * index + 6].y = index;
        __syncthreads();

        if (y < height - 1 && x > 0) {
            int index_dl = index + width - 1;
            float4 val_dl = img_f4_d[index_dl];
            float dist = distL2(val, val_dl);
            find_minimum(minimum_id, minimum, dist, index_dl);
            bd_d[num_nb * index + 7].y = index_dl;
            if (dist_d)
                dist_d[num_nb * index + 7] = dist;
        } else
            bd_d[num_nb * index + 7].y = index;
        __syncthreads();
    }

    nn_d[index] = minimum_id;
    dist_min_d[index] = minimum;

    for (int i = 0; i < num_nb; ++i)
        bd_d[num_nb * index + i].x = index;
}

/*
 * used to initialize the label of image pixels
 */
__global__ void
set_to_identity(int* const i1_d, const int N)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= N)
        return;

    i1_d[index] = index;
}

/*
 * simply draw the boundaries
 */
__global__ void
draw_boundary(uchar3* const seg_d,
              const int* const clus_d,
              const int width,
              const int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    int index = y * width + x;
    int clus_c = clus_d[index];

    if (x > 0) { // for boundary point to left
        int clus_left = clus_d[index - 1];
        if (clus_left != clus_c)
            seg_d[index] = make_uchar3(0, 0, 0);
    }

    if (y > 0) { // for boundary point to up
        int clus_up = clus_d[index - width];
        if (clus_up != clus_c)
            seg_d[index] = make_uchar3(0, 0, 0);
    }

    /* if (bold)
    if (x < width - 1) { // for boundary point to right
        int clus_right = clus_d[index + 1];
        if (clus_right != clus_c)
            seg_d[index] = make_uchar3(0, 0, 0);
    }

    if (y < height - 1) { // for boundary point to down
        int clus_down = clus_d[index + width];
        if (clus_down != clus_c)
            seg_d[index] = make_uchar3(0, 0, 0);
    }
    */
}
/*
 * visualize which segmentations are the cycle-roots of the CRTress
 */
__global__ void
draw_cycle_root(uchar3* const seg_d,
                int* const mask_d,
                const int width,
                const int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    int index = y * width + x;

    if (mask_d[index]) {
        uchar3 c = seg_d[index];
        seg_d[index] = make_uchar3(c.x, 0, 0);
    }
}
/*
 * draw segmentation labels in the uchar3 format
 */
__global__ void
draw_labels_uchar3(uchar3* const seg_d,
                   const int* const clus_d,
                   const int width,
                   const int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    int index = y * width + x;
    int clus_c = clus_d[index] + 1; // make index start from 1
    // code int to uchar3
    uchar clus_x = (uchar)(clus_c & 0xFF);
    uchar clus_y = (uchar)(clus_c >> 8 & 0xFF);
    uchar clus_z = (uchar)(clus_c >> 16 & 0xFF);

    // BGR order
    seg_d[index] = make_uchar3(clus_x, clus_y, clus_z);
}
/*
 * draw segmentation label in int format
 */
__global__ void
draw_labels_int(int* const seg_d,
                const int* const clus_d,
                const int width,
                const int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    int index = y * width + x;
    int clus_c = clus_d[index] + 1; // make index start from 1

    seg_d[index] = clus_c;
}

/*
 * after one iteration of clustering, map the index of boundaries to
 * the new cluster indices
 */
__global__ void
map_boundary(const int* const clus_d, int2* const bd_d, const int num_bd)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_bd)
        return;

    int2 bd = bd_d[index];
    bd.x = clus_d[bd.x];
    bd.y = clus_d[bd.y];
    bd_d[index] = bd;
}
/*
 * a predicate for removing duplicated boundaries
 * boundaries with the same indices or be the same with the previous boundary
 * are duplicated
 */
__global__ void
is_diff_bd(const int2* const bd_d, int* const predicate_d, const int num_bd)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_bd)
        return;

    int2 bd = bd_d[index];
    if ((index == num_bd - 1 || bd_d[index + 1] != bd) && (bd.x != bd.y)) {
        predicate_d[index] = 1;
    } else
        predicate_d[index] = 0;
}
/*
 * atomic reduce the distance associated with boundaries
 */
__global__ void
reduce_dist(const int2* const bd_d,
            int* const pos_scan_d,
            const float* const dist_d,
            float* const dist_rd_d,
            const int num_bd,
            const Linkage link)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_bd)
        return;

    int2 bd = bd_d[index];
    if (bd.x == bd.y) {
        return;
    }

    int t_id = pos_scan_d[index];

    float dist = dist_d[index];
    if (link == MinLink)
        atomicMinFloat(&dist_rd_d[t_id], dist);
    else if (link == MaxLink)
        atomicMaxFloat(&dist_rd_d[t_id], dist);
    else { // Sum link or Mean link
        atomicAdd(&dist_rd_d[t_id], dist);
    }
}

/*
 * get the start index of boundaries
 */
__global__ void
cast_boundary_s(const int2* const bd_d, int* const bds_d, const int num_bd)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_bd)
        return;

    bds_d[index] = bd_d[index].x;
}
/*
 * find the corresponding index of the minimum distance
 */
__global__ void
min_value_to_label(const int2* const bd_d,
                   const float* const dist_d,
                   float* const min_dist_d,
                   int* nn_d,
                   const int num_bd)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_bd)
        return;

    int2 bd = bd_d[index];
    float dist = dist_d[index];
    float min_dist = min_dist_d[bd.x];

    if (min_dist == dist) {
        // nn_d[bd.x] = bd.y;
        // if there are multiple same minimum, select the one with the maximum
        // index to avoid inconsistent results during multiple runs
        atomicMax(&nn_d[bd.x], bd.y);
    }
}

/*
 * set to a constant 'zero' value
 */
template<typename T>
__global__ void
set_to_zero(T* const t1_d, const int N, const T zero)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= N)
        return;

    t1_d[index] = zero;
}
/*
 * atomic reduce to compute maximum distance
 */
__global__ void
compute_dist_max_atomic(const int* const clus_d,
                        const float* const dist_min_d,
                        float* dist_max_d,
                        const int num_clus_p)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_clus_p)
        return;

    int cc = clus_d[index];
    float dist = dist_min_d[index];
    atomicMaxFloat(&dist_max_d[cc], dist);
}

/*
 * atomic reduce to compute minimum distance
 */
__global__ void
compute_dist_min_atomic(const int* const bds_d,
                        const float* const dist_d,
                        float* dist_min_d,
                        const int num_bd)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_bd)
        return;

    int bds = bds_d[index];
    float dist = dist_d[index];
    atomicMinFloat(&dist_min_d[bds], dist);
}

/*
 * atomicAdd reduce along the x axis per channel
 */
__global__ void
reduce_rows_atomic(const float* const mean_d,
                   float* const mean_rd_d,
                   const int* const clus_d,
                   const int length,
                   const int length_rd,
                   const int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= length || y >= channels)
        return;

    int cc = clus_d[x];
    int index = y * length + x;
    int index_rd = y * length_rd + cc;
    atomicAdd(&mean_rd_d[index_rd], mean_d[index]);
}

/*
 * sum to mean
 */
__global__ void
sum_to_mean(float* const mean_d, const int length, const int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= length || y >= channels - 1)
        return;

    int index = y * length + x;
    int index_w = (channels - 1) * length + x;
    float w = mean_d[index_w];
    mean_d[index] /= w;
}

/*
 * compute the distance of superpixels for Centroid and Ward Linkage
 */
__global__ void
compute_dist_mean_k(const float* const mean_d,
                    const int2* const bd_d,
                    float* const dist_d,
                    const int num_bd,
                    const int num_clus,
                    const int channels,
                    const bool is_compact)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_bd)
        return;

    int2 bd = bd_d[index];
    int s = bd.x, t = bd.y;

    // This is a BUG, have found no reason
    /*float dist = 0;*/
    /*for (int i = 0; i < channels - 1; ++i) {*/
    /*int base_id = i * num_clus;*/
    /*dist += distL2(mean_d[base_id + s], mean_d[base_id + t], i);*/
    /*}*/
    // This gives correct result as the atomic version
    float dist = distL2(mean_d[s], mean_d[t], 0) +
                 distL2(mean_d[s + num_clus], mean_d[t + num_clus], 1) +
                 distL2(mean_d[s + num_clus * 2], mean_d[t + num_clus * 2], 2);

    // TODO: can combine color and pos dist to achieve compactness
    if (is_compact)
        ;

    dist_d[index] = dist;
}

/*
 * compute the distance of superpixels for Centroid and Ward Linkage
 * atomic version
 */
__global__ void
compute_dist_mean_atomic(const float* const mean_d,
                         const int2* const bd_d,
                         float* const dist_d,
                         const int num_bd,
                         const int num_clus,
                         const int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= num_bd || y >= channels - 1)
        return;

    int2 bd = bd_d[x];
    int s = bd.x, t = bd.y;
    int base_id = y * num_clus;

    float dist = distL2(mean_d[base_id + s], mean_d[base_id + t], y);
    atomicAdd(&dist_d[x], dist);
}

/*
 * Centroid to Ward Linkage conversion
 */
__global__ void
ward_weight(const float* const mean_d,
            const int2* const bd_d,
            float* const dist_d,
            const int num_bd,
            const int num_clus,
            const int channels)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_bd)
        return;

    int2 bd = bd_d[index];
    int s = bd.x, t = bd.y;
    int base_id = (channels - 1) * num_clus;

    float s_w = mean_d[base_id + s];
    float t_w = mean_d[base_id + t];
    dist_d[index] = s_w * t_w / (s_w + t_w) * dist_d[index];
}
/*
 * pernalize the distance if it is larger than the target's
 * maximum inner-cluster distance
 */
__global__ void
pernalize_dist_k(const int2* const bd_d,
                 const float* const dist_max_d,
                 float* const dist_d,
                 const int num_bd)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_bd)
        return;

    // int s = bd_d[index].x;
    int t = bd_d[index].y;
    // This gives higher boundary recall, but lower under segmentation error
    // dist_d[index] += abs(dist_max_d[s] - dist_max_d[t]);
    float delta_d = dist_d[index] - dist_max_d[t];
    if (delta_d > 0) {
        dist_d[index] += delta_d;
    }

    // This makes the distance symetry, such that the cycle-root has only two
    // vertices
    // int s = bd_d[index].x;
    // int t = bd_d[index].y;
    // float delta_d_s = dist_d[index] - dist_max_d[s];
    // float delta_d_t = dist_d[index] - dist_max_d[t];
    // if (delta_d_s > 0) {
    //   dist_d[index] += delta_d_s;
    // }
    // if (delta_d_t > 0) {
    //   dist_d[index] += delta_d_t;
    // }
}

/*
 * update the label of image pixels
 */
__global__ void
update_image_label_k(int* img_clus_d,
                     int* const clus_d,
                     int* img_mask_d,
                     int* const mask_d,
                     const int im_size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= im_size)
        return;

    int label = img_clus_d[index];
    img_clus_d[index] = clus_d[label];
    if (img_mask_d)
        img_mask_d[index] = mask_d[label];
}

/*
 * pos_scan_d stores the number of child vertices
 * dist_d stores the summation of inner-cluster distance
 */
__global__ void
reduce_dist_pos_atomic(const float* const dist_min_d,
                       const int* const clus_d,
                       float* const dist_d,
                       int* const pos_scan_d,
                       const int num_clus_p)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_clus_p)
        return;

    int cc = clus_d[index];
    float dist_min = dist_min_d[index];
    atomicAdd(&pos_scan_d[cc], 1);
    atomicAdd(&dist_d[cc], dist_min);
}

/*
 * bound the label and number of child vertices of a superpixel
 * for the latter sorting
 */
__global__ void
set_label_pos(int2* const bd_scan_d, int* pos_scan_d, const int num_clus)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_clus)
        return;

    int2 bd_scan;
    bd_scan.x = index;
    bd_scan.y = pos_scan_d[index];
    bd_scan_d[index] = bd_scan;
}

/*
 * unpack the sorted label and pos
 */
__global__ void
unpack_label_pos(int2* const bd_scan_d,
                 int* const bds_d,
                 int* const pos_scan_d,
                 const int num_clus)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_clus)
        return;

    int2 bd_scan = bd_scan_d[index];
    bds_d[index] = bd_scan.x;
    pos_scan_d[index] = bd_scan.y;
}
/*
 * determine if a superpixel shall be split to its child vertices
 */
__global__ void
is_break_cluster(int* const predicate_d,
                 int* const pos_scan_d,
                 const int target_clus,
                 const int num_clus)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_clus)
        return;

    int expect_num = pos_scan_d[index] + num_clus - (index + 1);
    int expect_num_p = 0;
    if (index > 0)
        expect_num_p = pos_scan_d[index - 1] + num_clus - index;

    if (expect_num >= target_clus) {
        predicate_d[index] = 0;
        if (expect_num_p < target_clus) {
            predicate_d[index] = pos_scan_d[index];
            pos_scan_d[num_clus - 1] = index;
        }
    } else
        predicate_d[index] = pos_scan_d[index];
}
/*
 * help function
 */
__global__ void
pre_clus_label(int* const predicate_d,
               const int breaked_id_h,
               const int num_breaked_clus,
               const int num_clus)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_clus)
        return;

    if (predicate_d[index] == 0)
        predicate_d[index] = index - breaked_id_h + num_breaked_clus - 1;
    else
        predicate_d[index] -= 1;
}

/*
 * determine the new label of clusters
 */
__global__ void
reset_clus_label(int* const clus_d,
                 int* const pos_scan_d,
                 const int num_clus_p,
                 const int num_breaked_clus,
                 const int num_breaked_exact)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_clus_p)
        return;

    int cc = clus_d[index];
    int pre_cc = pos_scan_d[cc];
    if (pre_cc < num_breaked_clus) {
        // the atomic return old, instead of old - value
        // clus_d[index] = atomicSub(&pos_scan_d[cc], 1);
        int new_cc = atomicSub(&pos_scan_d[cc], 1);
        if (new_cc >= num_breaked_exact)
            clus_d[index] = num_breaked_exact;
        else
            clus_d[index] = new_cc;
    } else {
        clus_d[index] = pre_cc + num_breaked_exact - num_breaked_clus + 1;
    }
}

__global__ void
img_struct_to_array(const float4* const, float* const, const int, const int);

__global__ void
img_uchar3_to_float4(const uchar3*, float4*, const int, const int);

__global__ void
img_float4_to_uchar3(const float4*, uchar3*, const int, const int);

__global__ void
img_uchar3_to_float1(const uchar3*, float*, const int, const int);

__global__ void
img_float4_to_float1(const float4*, float*, const int, const int, const int);

__global__ void
img_float1_to_float4(const float*, float4*, const int, const int, const int);

__global__ void
img_float1_to_uchar3(const float*, uchar3*, const int, const int);

__global__ void
img_BGR_to_LAB(float4*, const int, const int);

__global__ void
img_LAB_to_BGR(float4*, const int, const int);

/*********************************************************************************/
// code for host
void
img_gauss_blur(float4*&, float4*&, float*, const double, const int, const int);

void
img_sobel_grad(float4*&, float4*&, const int, const int);

// The image opearations can also implemented by OpenCV GPU
// which would be much simpler
/*
 * copy image from host to device
 */
void
SegHAC::img_CPU_to_GPU(const cv::Mat& src,
                       float4*& img_f4_d,
                       const double sigma)
{
    cudaMemcpy(
      img_u3_d, src.data, sizeof(uchar3) * im_size, cudaMemcpyHostToDevice);

    img_uchar3_to_float4<<<img_grids, m_blocks>>>(
      img_u3_d, img_f4_d, width, height);

    if (sigma > FLT_MIN)
        img_gauss_blur(img_f4_d, buf_f4_d, filter_d, sigma, width, height);

    img_BGR_to_LAB<<<img_grids, m_blocks>>>(img_f4_d, width, height);
}

/*
 * copy image from device to host
 */
void
SegHAC::img_GPU_to_CPU(const float* const img_f1_d, cv::Mat& dst_img)
{
    img_float1_to_float4<<<img_grids, m_blocks>>>(
      img_f1_d, img_f4_d, width, height, 0);

    img_LAB_to_BGR<<<img_grids, m_blocks>>>(img_f4_d, width, height);

    img_float4_to_uchar3<<<img_grids, m_blocks>>>(
      img_f4_d, img_u3_d, width, height);

    cudaMemcpy(
      dst_img.data, img_u3_d, sizeof(uchar3) * im_size, cudaMemcpyDeviceToHost);
}

/*
 * compute the nearest neighbor of pixels in the
 * image grid
 */
void
SegHAC::compute_1nn_grid(const float4* const img_f4_d,
                         int* const nn_d,
                         int2* const bd_d,
                         float* const dist_d,
                         float* const dist_min_d)
{
    compute_1nn_grid_k<<<img_grids, m_blocks>>>(
      img_f4_d, nn_d, bd_d, dist_rd_d, dist_min_d, num_nb, width, height);

    if (dist_rd_d)
        cudaMemcpy(
          dist_d, dist_rd_d, sizeof(float) * num_bd, cudaMemcpyDeviceToDevice);
}

/*
 * initialize the mean vector
 */
void
SegHAC::init_data_mean(const float4* const img_f4_d, float* const mean_d)
{
    img_struct_to_array<<<img_grids, m_blocks>>>(
      img_f4_d, mean_d, width, height);
}

/*
 * initialize the image label
 */
void
SegHAC::initialize_image_label(int* const img_clus_d, const int im_size)
{
    set_to_identity<<<img_grid, m_block>>>(img_clus_d, im_size);

    num_bd = im_size * num_nb;
    num_clus_isp.clear();
    num_bd_isp.clear();
}

void
SegHAC::update_image_label(int* img_clus_d,
                           int* const clus_d,
                           int* img_mask_d,
                           int* const mask_d)
{
    if (img_mask_d)
        cudaMemset(img_mask_d, 0, sizeof(int) * im_size);
    update_image_label_k<<<img_grid, m_block>>>(
      img_clus_d, clus_d, img_mask_d, mask_d, im_size);
}

void draw_1nn_graph(cv::Mat& seg_h, const int* const nn_d, const int* const img_clus_d,
const int num_clus, const int width, const int height){
    int* nn_h = new int[num_clus];
    int* img_clus_h = new int[width * height];
    float* pos_h = new float[num_clus * 3];
    cudaMemcpy(nn_h, nn_d, sizeof(int)*num_clus, cudaMemcpyDeviceToHost);
    cudaMemcpy(img_clus_h, img_clus_d, sizeof(int)*width*height, cudaMemcpyDeviceToHost);
    for (int i=0;i<num_clus * 3;++i) pos_h[i] = 0;
    for (int i=0;i<width * height;++i) {
        int c = img_clus_h[i];
        pos_h[c] += i % width;
        pos_h[num_clus + c] += i / width;
        pos_h[num_clus * 2 + c] += 1;
    }
    for (int i=0;i<num_clus;++i) {
        pos_h[i] /= pos_h[num_clus * 2 +i];
        pos_h[num_clus + i] /= pos_h[num_clus * 2 +i];
    }

    for (int i=0;i<num_clus;++i) {
        int nn = nn_h[i];
        float s_x = pos_h[i];
        float s_y = pos_h[num_clus + i];
        float t_x = pos_h[nn];
        float t_y = pos_h[num_clus + nn];
        cv::Point2f s_p(s_x, s_y);
        cv::Point2f t_p(t_x, t_y);
        cv::Point2f c_p = s_p + (t_p-s_p)/ 3 * 2;
        cv::circle(seg_h, s_p, 2, cv::Scalar(0,0,0), 2, cv::LINE_AA);
        cv::arrowedLine(seg_h, s_p, c_p, cv::Scalar(0,0,0), 2, cv::LINE_AA,0, 0.10);
        cv::line(seg_h, c_p, t_p, cv::Scalar(0,0,0), 2, cv::LINE_AA);
    }


    delete[] pos_h;
    delete[] img_clus_h;
    delete[] nn_h;
}

cv::Mat
SegHAC::draw_segmentation(int* img_clus_d, int* img_mask_d)
{
    if (m_seg_format == BoundaryFormat) {
        cudaMemcpy(seg_u3_d,
                   img_u3_d,
                   sizeof(uchar3) * im_size,
                   cudaMemcpyDeviceToDevice);
        draw_boundary<<<img_grids, m_blocks>>>(
          seg_u3_d, img_clus_d, width, height);
        if (img_mask_d)
            draw_cycle_root<<<img_grids, m_blocks>>>(
              seg_u3_d, img_mask_d, width, height);
    } else if (m_seg_format == LabelUchar3Format) {
        cudaMemset(seg_u3_d, 0, sizeof(uchar3) * im_size);
        draw_labels_uchar3<<<img_grids, m_blocks>>>(
          seg_u3_d, img_clus_d, width, height);
    } else if (m_seg_format == LabelIntFormat) {
        cudaMemset(seg_i1_d, 0, sizeof(uchar3) * im_size);
        draw_labels_int<<<img_grids, m_blocks>>>(
          seg_i1_d, img_clus_d, width, height);
    }

    cv::Mat seg_h;
    if (m_seg_format == BoundaryFormat || m_seg_format == LabelUchar3Format) {
        seg_h = cv::Mat(height, width, CV_8UC3);
        cudaMemcpy(seg_h.data,
                   seg_u3_d,
                   sizeof(uchar3) * im_size,
                   cudaMemcpyDeviceToHost);
    } else if (m_seg_format == LabelIntFormat) {
        seg_h = cv::Mat(height, width, CV_32SC1);
        cudaMemcpy(
          seg_h.data, seg_i1_d, sizeof(int) * im_size, cudaMemcpyDeviceToHost);
    }

    return seg_h;
}

void
SegHAC::compute_dist_pos(const float* const dist_min_d,
                         const int* const clus_d,
                         float* const dist_d,
                         int* const pos_scan_d,
                         const int num_clus_p,
                         const int num_clus)
{
    int grid_p = (num_clus_p + m_block - 1) / m_block;
    cudaMemset(pos_scan_d, 0, sizeof(int) * num_clus);
    cudaMemset(dist_d, 0, sizeof(float) * num_clus);
    reduce_dist_pos_atomic<<<grid_p, m_block>>>(
      dist_min_d, clus_d, dist_d, pos_scan_d, num_clus_p);
}

void
SegHAC::compute_scaned_pos(float* const dist_d,
                           int* const pos_scan_d,
                           int* const bds_d,
                           const int num_clus)
{
    int grid = (num_clus + m_block - 1) / m_block;
    // use bd_scan_d as buffer here
    set_label_pos<<<grid, m_block>>>(bd_scan_d, pos_scan_d, num_clus);

    // sort in descend order
    thrust::sort_by_key(thrust::device,
                        dist_d,
                        dist_d + num_clus,
                        bd_scan_d,
                        thrust::greater<float>());

    unpack_label_pos<<<grid, m_block>>>(bd_scan_d, bds_d, pos_scan_d, num_clus);

    // inplace inclusive scan
    thrust::inclusive_scan(
      thrust::device, pos_scan_d, pos_scan_d + num_clus, pos_scan_d);
}

std::pair<int, int>
SegHAC::set_pre_clus_label(int* const pos_scan_d,
                           const int* const bds_d,
                           const int target_clus,
                           const int num_clus)
{
    int grid = (num_clus + m_block - 1) / m_block;
    is_break_cluster<<<grid, m_block>>>(
      predicate_d, pos_scan_d, target_clus, num_clus);

    int breaked_id_h = 0;
    cudaMemcpy(&breaked_id_h,
               &pos_scan_d[num_clus - 1],
               sizeof(int),
               cudaMemcpyDeviceToHost);
    int num_breaked_clus = 0;
    cudaMemcpy(&num_breaked_clus,
               &predicate_d[breaked_id_h],
               sizeof(int),
               cudaMemcpyDeviceToHost);

    pre_clus_label<<<grid, m_block>>>(
      predicate_d, breaked_id_h, num_breaked_clus, num_clus);

    thrust::scatter(
      thrust::device, predicate_d, predicate_d + num_clus, bds_d, pos_scan_d);

    int num_breaked_exact = m_target_clus - num_clus + breaked_id_h;
    return std::make_pair(num_breaked_clus, num_breaked_exact);
}

/*
 * Generate given number of superpixels
 * combaine the superpixels of the current hierarchy and the previous hierarchy
 */
int
SegHAC::get_target_clus(const float* const dist_min_d,
                        int* const clus_d,
                        const int num_clus_p,
                        const int num_clus)
{
    if (num_clus == m_target_clus)
        return num_clus;
    // now num_clus_p > m_target_clus && num_clus < m_target_clus
    compute_dist_pos(
      dist_min_d, clus_d, dist_d, pos_scan_d, num_clus_p, num_clus);

    compute_scaned_pos(dist_d, pos_scan_d, bds_d, num_clus);

    std::pair<int, int> num_breaked =
      set_pre_clus_label(pos_scan_d, bds_d, m_target_clus, num_clus);

    int grid_p = (num_clus_p + m_block - 1) / m_block;
    reset_clus_label<<<grid_p, m_block>>>(
      clus_d, pos_scan_d, num_clus_p, num_breaked.first, num_breaked.second);

    // TODO: this can be omitted, it is used to check the correctness
    int num_clus_new = crtrees->compact_clus_label(clus_d, num_clus_p);
    assert(num_clus_new == m_target_clus);
    return num_clus_new;
}

/*
 * Map and reduce the boundary and the corresponding distance
 * After clustering, the start and end index of a boundary shall be
 * mapped into the new cluster indices. Then the redundant boundaries shall
 * be removed.
 * The dist_d has a one-to-one mapping with the bd_d
 */
int
SegHAC::reduce_boundary_dist(int2*& bd_d,
                             int2*& bd_rd_d,
                             float*& dist_d,
                             float*& dist_rd_d,
                             const int* const clus_d,
                             const int num_bd)
{
    int grid = (num_bd + m_block - 1) / m_block;
    map_boundary<<<grid, m_block>>>(clus_d, bd_d, num_bd);
    if (!use_mean) {
        thrust::sort_by_key(thrust::device, bd_d, bd_d + num_bd, dist_d);
    } else
        thrust::sort(thrust::device, bd_d, bd_d + num_bd);

    is_diff_bd<<<grid, m_block>>>(bd_d, predicate_d, num_bd);
    thrust::exclusive_scan(
      thrust::device, predicate_d, predicate_d + num_bd, pos_scan_d);
    int num_bd_new = get_num_from_scan(predicate_d, pos_scan_d, num_bd);
    // std::cout << num_bd << ", " << num_bd_new << std::endl;
    int grid_new = (num_bd_new + m_block - 1) / m_block;
    if (!use_mean) {
        set_to_zero<<<grid_new, m_block>>>(
          dist_rd_d, num_bd_new, m_link == MinLink ? FLT_MAX : 0.f);
        reduce_dist<<<grid, m_block>>>(
          bd_d, pos_scan_d, dist_d, dist_rd_d, num_bd, m_link);
        std::swap(dist_d, dist_rd_d);
    }

    thrust::scatter_if(
      thrust::device, bd_d, bd_d + num_bd, pos_scan_d, predicate_d, bd_rd_d);
    // ping_pang pointer
    std::swap(bd_d, bd_rd_d);

    return num_bd_new;
}

/*
 * Save the segmentation results
 */
bool
SegHAC::save_output(std::vector<cv::Mat>* segs,
                    const int num_clus_p,
                    const int num_clus,
                    const int num_bd)
{
    update_image_label(img_clus_d, clus_d, img_mask_d, mask_d);
    if (m_seg_format != EmptyFormat &&
        (m_target_clus == 1 || m_target_clus == num_clus)) {
        segs->emplace_back(draw_segmentation(img_clus_d, img_mask_d));
        std::cout << "number of CRTrees: " << num_clus << std::endl;
    }
    m_level ++;
    // determine stop
    return (m_target_clus == num_clus);
}

/*
 * Compute the maximum inner-cluster distance
 */
void
SegHAC::compute_dist_inner_max(const int* const clus_d,
                               const float* const dist_min_d,
                               float* const dist_max_d,
                               const int num_clus_p,
                               const int num_clus)
{
    int grid_p = (num_clus_p + m_block - 1) / m_block;
    // set to 'zero', since will use atomic
    // for distance value, the 'zero' is 0 here
    cudaMemset(dist_max_d, 0, sizeof(float) * num_clus);
    compute_dist_max_atomic<<<grid_p, m_block>>>(
      clus_d, dist_min_d, dist_max_d, num_clus_p);
}

// TODO: atomicAdd of float will introduce rounding error and make
// the result non-determinent
// - can use the previous version compute_data_mean to resolve this,
// but its runtime is longer
void
SegHAC::reduce_data_mean(float* const mean_d,
                         float* const mean_rd_d,
                         const int* const clus_d,
                         const int num_clus_p,
                         const int num_clus,
                         const int channels)
{
    dim3 grids((num_clus + m_blocks.x - 1) / m_blocks.x,
               (channels + m_blocks.y - 1) / m_blocks.y);
    dim3 grids_p((num_clus_p + m_blocks.x - 1) / m_blocks.x,
                 (channels + m_blocks.y - 1) / m_blocks.y);

    cudaMemset(mean_rd_d, 0, sizeof(float) * num_clus * channels);
    reduce_rows_atomic<<<grids_p, m_blocks>>>(
      mean_d, mean_rd_d, clus_d, num_clus_p, num_clus, channels);

    cudaMemcpy(mean_d,
               mean_rd_d,
               sizeof(float) * num_clus * channels,
               cudaMemcpyDeviceToDevice);

    // normalize
    sum_to_mean<<<grids, m_blocks>>>(mean_rd_d, num_clus, channels);
}

/*
 * Compute distance between superpixels represented by image means
 */
void
SegHAC::compute_dist_mean(float* const mean_d,
                          const int2* const bd_d,
                          float* const dist_d,
                          const int num_bd,
                          const int num_clus)
{
    int grid = (num_bd + m_block - 1) / m_block;

    dim3 grids((num_bd + m_blocks.x - 1) / m_blocks.x,
               (mean_channels + m_blocks.y - 1) / m_blocks.y);

    // cudaMemset(dist_d, 0, sizeof(float) * num_bd);
    // compute_dist_mean_atomic<<<grids, m_blocks>>>(
    //   mean_d, bd_d, dist_d, num_bd, num_clus, mean_channels);

    cudaMemset(dist_d, 0, sizeof(float) * num_bd);
    compute_dist_mean_k<<<grid, m_block>>>(
      mean_d, bd_d, dist_d, num_bd, num_clus, mean_channels, m_compact);

    if (m_link == WardLink) {
        ward_weight<<<grid, m_block>>>(
          mean_d, bd_d, dist_d, num_bd, num_clus, mean_channels);
    }
}

/*
 * pernalization of distance
 */
void
SegHAC::pernalize_dist(const int2* const bd_d,
                       const float* const dist_max_d,
                       float* const dist_d,
                       const int num_bd)
{
    int grid = (num_bd + m_block - 1) / m_block;
    pernalize_dist_k<<<grid, m_block>>>(bd_d, dist_max_d, dist_d, num_bd);
}

int
SegHAC::compute_dist_reduce(const int2* const bd_d,
                            const float* const dist_d,
                            int2* const bd_rd_d,
                            float* const dist_rd_d,
                            const int num_bd)
{
    thrust::equal_to<int2> equal_pred;
    thrust::maximum<float> reduce_op;
    auto new_end = thrust::reduce_by_key(thrust::device,
                                         bd_d,
                                         bd_d + num_bd,
                                         dist_d,
                                         bd_rd_d,
                                         dist_rd_d,
                                         equal_pred,
                                         reduce_op);
    int num_bd_rd = (int)(new_end.second - dist_rd_d);
    return num_bd_rd;
}

/*
 * Compute the nearest neighbor indices of superpixels
 */
void
SegHAC::compute_1nn(const int2* const bd_d,
                    const float* const dist_d,
                    int* const nn_d,
                    const int num_bd,
                    const int num_clus)
{
    int grid = (num_bd + m_block - 1) / m_block;
    cast_boundary_s<<<grid, m_block>>>(bd_d, bds_d, num_bd);

    // There is a previous thrust::reduce_by_key version to ensure
    // correctness Set to 'zero', since will use atomic for distance value,
    // the 'zero' is FLT_MAX here
    set_to_zero<<<(num_clus + m_block - 1) / m_block, m_block>>>(
      dist_min_d, num_clus, FLT_MAX);
    compute_dist_min_atomic<<<grid, m_block>>>(
      bds_d, dist_d, dist_min_d, num_bd);
    // Set nn_d to 'zero' since atomicMax is used in min_value_to_label
    cudaMemset(nn_d, 0, sizeof(int) * num_clus);
    min_value_to_label<<<grid, m_block>>>(
      bd_d, dist_d, dist_min_d, nn_d, num_bd);
}

/*
 * This function is to validate when the distance is not symmetry,
 * the cycle-root may have larger than 2 vertices
 */
void
check_cycle_num(const int* const mask_d,
                const int* const clus_d,
                const int num_clus_p,
                const int num_clus)
{
    int* mask_h = new int[num_clus_p];
    int* clus_h = new int[num_clus_p];
    int* mask_num = new int[num_clus]{ 0 };
    cudaMemcpy(
      mask_h, mask_d, sizeof(int) * num_clus_p, cudaMemcpyDeviceToHost);
    cudaMemcpy(
      clus_h, clus_d, sizeof(int) * num_clus_p, cudaMemcpyDeviceToHost);

    for (int i = 0; i < num_clus_p; ++i) {
        if (mask_h[i])
            mask_num[clus_h[i]]++;
    }

    bool has_larger_than_two = false;
    for (int i = 0; i < num_clus; ++i)
        if (mask_num[i] > 2)
            has_larger_than_two = true;

    if (has_larger_than_two) {
        std::cout << "cycle num larger than 2: " << std::endl;
        for (int i = 0; i < num_clus; ++i)
            if (mask_num[i] > 2)
                std::cout << mask_num[i] << " ";
        std::cout << std::endl;
    }

    delete[] mask_num;
    delete[] clus_h;
    delete[] mask_h;
}

/*
 * The main funtion for image segmentation using CRTrees-Clustering
 * num_clus = 1 : generate a hierarchy of image segmentations till there is
 *                only 1 cluster
 * num_clus > 1 : generate given number of superpixles
 */
void
SegHAC::run_ms(std::vector<cv::Mat>* segs)
{
    compute_1nn_grid(img_f4_d, nn_d, bd_d, dist_d, dist_min_d);

    if (use_mean)
        init_data_mean(img_f4_d, mean_d);

    int num_clus = im_size;
    initialize_image_label(img_clus_d, im_size);

    while (num_clus > 1) {
        auto c0 = std::chrono::steady_clock::now();
        crtrees->get_clus(nn_d, clus_d, num_clus, mask_d);
        auto c1 = std::chrono::steady_clock::now();
        auto delta_c = c1 - c0;
        std::cout << "Time for CRTrees Labeling to label " << num_clus << " vertices: " << std::chrono::duration_cast<std::chrono::microseconds>(delta_c).count() /1e3 << " ms" << std::endl;

        int num_clus_p = num_clus;
        num_clus = crtrees->compact_clus_label(clus_d, num_clus);

        if (show_cycle)
            check_cycle_num(mask_d, clus_d, num_clus_p, num_clus);

        if (num_clus <= m_target_clus) {
            num_clus =
              get_target_clus(dist_min_d, clus_d, num_clus_p, num_clus);
        }

        if (save_output(segs, num_clus_p, num_clus, num_bd))
            break;

        num_bd = reduce_boundary_dist(
          bd_d, bd_rd_d, dist_d, dist_rd_d, clus_d, num_bd);

        compute_dist_inner_max(
          clus_d, dist_min_d, dist_max_d, num_clus_p, num_clus);

        if (use_mean) {
            reduce_data_mean(
              mean_d, mean_rd_d, clus_d, num_clus_p, num_clus, mean_channels);
            compute_dist_mean(mean_rd_d, bd_d, dist_d, num_bd, num_clus);
        }
        if (is_pernalize_dist)
            pernalize_dist(bd_d, dist_max_d, dist_d, num_bd);
        compute_1nn(bd_d, dist_d, nn_d, num_bd, num_clus);

        // change the -1 to a desired level to draw the 1-NN graph
        if (m_seg_format == BoundaryFormat && (m_level == -1)) {
            draw_1nn_graph(segs->at(m_level-1), nn_d, img_clus_d, num_clus, width, height);
        }
    }
}

// TODO: explore multiple streams

// TODO: utilize CPU when num_clus is small (e.g. < 1000)
//       GPU works well when num_clus is large (scale well)
//       but for small num_clus, it does not scale well and consume much
//       time
