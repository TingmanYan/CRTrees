#ifndef _SEGMENT_HPP_
#define _SEGMENT_HPP_

#include "CRTrees.hpp"
#include "util.h"
#include <opencv2/opencv.hpp>

enum Linkage
{
    // Min and Max Links are more local
    MinLink,
    MaxLink,
    // Centorid and Ward Links exploit region information
    CentoridLink,
    WardLink
};

enum SegFormat
{
    LabelUchar3Format, // Per-pixel label, index starts from 1
    LabelIntFormat,
    BoundaryFormat,     // Boundary overlapped on the input color image
    LabelDeviceIntFormat, // Label on GPU
    EmptyFormat
};

class SegHAC
{
  protected:
    void img_CPU_to_GPU(const cv::Mat&, float4*&, const double);
    void img_GPU_to_CPU(const float*, cv::Mat&);

    void compute_1nn_grid(const float4*, int*, int2*, float*, float*);
    void init_data_mean(const float4*, float*);
    void initialize_image_label(int*, const int);
    int get_target_clus(const float*, int*, const int, const int);
    int reduce_boundary_dist(int2*&,
                             int2*&,
                             float*&,
                             float*&,
                             const int*,
                             const int);
    bool save_output(std::vector<cv::Mat>*, const int, const int, const int);
    void compute_dist_inner_max(const int*,
                                const float*,
                                float*,
                                const int,
                                const int);
    void compute_1nn(const int2*, const float*, int*, const int, const int);
    int compute_dist_reduce(const int2*,
                            const float*,
                            int2*,
                            float*,
                            const int);
    void reduce_data_mean(float*,
                          float*,
                          const int*,
                          const int,
                          const int,
                          const int);
    void compute_dist_mean(float*, const int2*, float*, const int, const int);
    void pernalize_dist(const int2*, const float*, float*, const int);

  private:
    void update_image_label(int*, int*, int*, int*);
    cv::Mat draw_segmentation(int*, int*);

    void compute_dist_pos(const float* const,
                          const int* const,
                          float* const,
                          int* const,
                          const int,
                          const int);
    void compute_scaned_pos(float* const, int* const, int* const, const int);
    std::pair<int, int> set_pre_clus_label(int* const,
                                           const int* const,
                                           const int,
                                           const int);

  public:
    void run_ms(std::vector<cv::Mat>* segs = nullptr);

    /*
     * update images of video streams
     */
    void set_frame(const cv::Mat& frame)
    {
        img_CPU_to_GPU(frame, img_f4_d, m_sigma);
    }
    /*
     * update images for image sequence
     */
    void set_seq_img(const cv::Mat& img)
    {
        if (img.cols != width) {
            assert(img.cols == height && img.rows == width);
            width = img.cols, height = img.rows;
            // grids also changed
            img_grids = dim3((width + m_blocks.x - 1) / m_blocks.x,
                             (height + m_blocks.y - 1) / m_blocks.y);
            img_grid = (im_size + m_block - 1) / m_block;
        }
        img_CPU_to_GPU(img, img_f4_d, m_sigma);
    }

  public:
    // interfaces
    int g_width() const { return this->width; }
    int g_height() const { return this->height; }
    std::vector<int> g_num_clus_isp() const { return this->num_clus_isp; }
    std::vector<int> g_num_bd_isp() const { return this->num_bd_isp; }

  protected:
    // stores the start and end vertex indices of a boundary 
    // the '_d' means 'device'
    int2* bd_d = nullptr;

    // stores color, position, and num pixels
    float* mean_d = nullptr;

    std::vector<int> num_clus_isp;
    std::vector<int> num_bd_isp;

  private:
    CRTrees* crtrees = nullptr;

    const Linkage m_link;
    const int num_nb = 8;                // or 4: up, down, left, and right
    const double m_sigma = 0;
    const SegFormat m_seg_format;
    const int m_target_clus;
    const bool show_cycle;

    const bool m_compact = false;

    uchar3* img_u3_d = nullptr;
    float4* img_f4_d = nullptr;
    float4* buf_f4_d = nullptr;

    uchar3* seg_u3_d = nullptr;
    int* seg_i1_d = nullptr;

    // stores distance between neighboring pixels or superpixels
    float* dist_d = nullptr;

    // reduced bd_d
    int2* bd_rd_d = nullptr;
    float* dist_rd_d = nullptr;

    int* nn_d = nullptr;
    int* clus_d = nullptr;
    int* img_clus_d = nullptr;

    float* mean_rd_d = nullptr;

    // help variables for scan operation
    int* predicate_d = nullptr;
    int* pos_scan_d = nullptr;

    int2* bd_scan_d = nullptr;

    int* bds_d = nullptr;
    float* dist_min_d = nullptr;
    float* dist_max_d = nullptr;

    int* mask_d = nullptr;
    int* img_mask_d = nullptr;

    float* filter_d = nullptr;

    const int mean_channels = 3 + 2 + 1; // color, position, and num-pixels
    const int max_filter_width = 15;

    int max_isp_levels;
    int width;
    int height;
    const int im_size;
    int num_bd;
    int m_level=0;

    const dim3 m_blocks = dim3(32, 2);
    const int m_block = 32 * 2;

    dim3 img_grids;
    int img_grid;

    const bool is_pernalize_dist = true;
    const bool use_mean;

  public:
    /*
     * allocate all device variables at once
     * this is helpful when processing video streams
     */
    SegHAC(const cv::Mat& src,
           const Linkage link,
           const int num_nb,
           const double sigma,
           const SegFormat seg_format,
           const int target_clus,
           const bool show_cycle)
      : width(src.cols)
      , height(src.rows)
      , im_size(width * height)
      , img_grids((width + m_blocks.x - 1) / m_blocks.x,
                  (height + m_blocks.y - 1) / m_blocks.y)
      , img_grid((im_size + m_block - 1) / m_block)
      , num_bd(im_size * num_nb)
      , m_link(link)
      , use_mean(link == CentoridLink || link == WardLink)
      , num_nb(num_nb)
      , m_sigma(sigma)
      , m_seg_format(seg_format)
      , m_target_clus(target_clus)
      , show_cycle(show_cycle)
    {
        crtrees = new CRTrees(im_size);
        // TODO: Try to allocate multiple arrays in parallel
        cudaMalloc(&img_u3_d, sizeof(uchar3) * im_size);
        cudaMalloc(&img_f4_d, sizeof(float4) * im_size);
        cudaMalloc(&buf_f4_d, sizeof(float4) * im_size);
        cudaMalloc(&bd_d, sizeof(int2) * num_bd);
        cudaMalloc(&bd_rd_d, sizeof(int2) * num_bd);
        cudaMalloc(&nn_d, sizeof(int) * im_size);
        cudaMalloc(&clus_d, sizeof(int) * im_size);
        cudaMalloc(&img_clus_d, sizeof(int) * im_size);
        cudaMalloc(&dist_d, sizeof(float) * num_bd);

        if (use_mean) {
            cudaMalloc(&mean_d, sizeof(float) * im_size * mean_channels);
            cudaMalloc(&mean_rd_d, sizeof(float) * im_size * mean_channels);
        } else {
            cudaMalloc(&dist_rd_d, sizeof(float) * num_bd);
        }

        if (m_seg_format == BoundaryFormat || m_seg_format == LabelUchar3Format)
            cudaMalloc(&seg_u3_d, sizeof(uchar3) * im_size);
        else if (m_seg_format == LabelIntFormat)
            cudaMalloc(&seg_i1_d, sizeof(int) * im_size);

        if (show_cycle) {
            cudaMalloc(&mask_d, sizeof(int) * im_size);
            cudaMalloc(&img_mask_d, sizeof(int) * im_size);
        }

        cudaMalloc(&predicate_d, sizeof(int) * num_bd);
        cudaMalloc(&pos_scan_d, sizeof(int) * num_bd);

        cudaMalloc(&bd_scan_d, sizeof(int2) * num_bd);
        cudaMalloc(&bds_d, sizeof(int) * num_bd);

        cudaMalloc(&dist_min_d, sizeof(float) * im_size);
        cudaMalloc(&dist_max_d, sizeof(float) * im_size);

        assert(static_cast<int>(round(3 * sigma)) * 2 + 1 <= max_filter_width);
        // 2 * [3sigma] + 1 <= 15, sigma < 2.33
        cudaMalloc(&filter_d,
                   sizeof(float) * max_filter_width * max_filter_width);

        img_CPU_to_GPU(src, img_f4_d, m_sigma);
    }

    ~SegHAC()
    {
        cudaFree(filter_d);
        cudaFree(dist_max_d);
        cudaFree(dist_min_d);
        cudaFree(bds_d);
        cudaFree(bd_scan_d);
        cudaFree(pos_scan_d);
        cudaFree(predicate_d);
        cudaFree(seg_i1_d);
        cudaFree(seg_u3_d);
        cudaFree(mean_rd_d);
        cudaFree(mean_d);
        cudaFree(bd_rd_d);
        cudaFree(dist_d);
        cudaFree(img_mask_d);
        cudaFree(mask_d);
        cudaFree(dist_rd_d);
        cudaFree(img_clus_d);
        cudaFree(clus_d);
        cudaFree(nn_d);
        cudaFree(bd_d);
        cudaFree(buf_f4_d);
        cudaFree(img_f4_d);
        cudaFree(img_u3_d);
        delete crtrees;
    }
};

#endif
