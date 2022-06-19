#include "util.h"
#include <cmath>

#define uchar unsigned char

__device__ __forceinline__ float4
BGR_to_LAB(const float4& in)
{
    double b, g, r;
    double x, y, z;

    // assume the input ranges from [0, 255]
    b = in.x / 255.0;
    g = in.y / 255.0;
    r = in.z / 255.0;

    r = (r > 0.04045) ? pow((r + 0.055) / 1.055, 2.4) : r / 12.92;
    g = (g > 0.04045) ? pow((g + 0.055) / 1.055, 2.4) : g / 12.92;
    b = (b > 0.04045) ? pow((b + 0.055) / 1.055, 2.4) : b / 12.92;

    x = (r * 0.412453 + g * 0.357580 + b * 0.180423) / 0.950456;
    y = (r * 0.212671 + g * 0.715160 + b * 0.072169) / 1.0;
    z = (r * 0.019334 + g * 0.119193 + b * 0.950227) / 1.088754;

    double epsilon = 216 / 24389.0;
    double k = 841 / 108.0;
    double kb = 16 / 116.0;
    double alpha = 1 / 3.0;

    x = (x > epsilon) ? pow(x, alpha) : (k * x) + kb;
    y = (y > epsilon) ? pow(y, alpha) : (k * y) + kb;
    z = (z > epsilon) ? pow(z, alpha) : (k * z) + kb;

    float4 out;

    out.x = (116 * y) - 16;
    out.y = 500 * (x - y);
    out.z = 200 * (y - z);
    out.w = in.w;

    return out;
}

__device__ __forceinline__ float4
LAB_to_BGR(const float4& in)
{
    double b, g, r;
    double x, y, z;

    y = (in.x + 16.0) / 116.0;
    x = in.y / 500.0 + y;
    z = y - in.z / 200.0;

    x =
      0.95047 * ((x * x * x > 0.008856) ? x * x * x : (x - 16 / 116.0) / 7.787);
    y =
      1.00000 * ((y * y * y > 0.008856) ? y * y * y : (y - 16 / 116.0) / 7.787);
    z =
      1.08883 * ((z * z * z > 0.008856) ? z * z * z : (z - 16 / 116.0) / 7.787);

    r = x * 3.2406 + y * -1.5372 + z * -0.4986;
    g = x * -0.9689 + y * 1.8758 + z * 0.0415;
    b = x * 0.0557 + y * -0.2040 + z * 1.0570;

    r = (r > 0.0031308) ? (1.055 * pow(r, 0.4166) - 0.055) : 12.92 * r;
    g = (g > 0.0031308) ? (1.055 * pow(g, 0.4166) - 0.055) : 12.92 * g;
    b = (b > 0.0031308) ? (1.055 * pow(b, 0.4166) - 0.055) : 12.92 * b;

    float4 out;

    out.x = max(0., min(1., b)) * 255;
    out.y = max(0., min(1., g)) * 255;
    out.z = max(0., min(1., r)) * 255;
    out.w = in.w;

    return out;
}

__device__ __forceinline__ float
sobel_kernel(const float& pix00,
             const float& pix01,
             const float& pix02,
             const float& pix10,
             const float& pix12,
             const float& pix20,
             const float& pix21,
             const float& pix22)
{
    float h_grad = pix02 + 2 * pix12 + pix22 - pix00 - 2 * pix10 - pix20;
    float v_grad = pix00 + 2 * pix01 + pix02 - pix20 - 2 * pix21 - pix22;
    return fabsf(h_grad) + fabsf(v_grad);
}

__global__ void
img_struct_to_array(const float4* const img_f4_d,
                    float* const array_d,
                    const int width,
                    const int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    int im_size = width * height;
    int index = y * width + x;
    float4 val = img_f4_d[index];

    array_d[index] = val.x;
    array_d[index + im_size] = val.y;
    array_d[index + im_size * 2] = val.z;

    array_d[index + im_size * 3] = x;
    array_d[index + im_size * 4] = y;
    array_d[index + im_size * 5] = val.w;
}

__global__ void
img_BGR_to_grey(const uchar3* const img_u3_d,
                float* const img_f1_d,
                const int width,
                const int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    int index = y * width + x;
    uchar3 data = img_u3_d[index];
    img_f1_d[index] = data.x * 0.114 + data.y * 0.587 + data.z * 0.299;
}

__global__ void
img_float1_to_uchar1(const float* const img_f1_d,
                     uint8_t* const img_u1_d,
                     const int width,
                     const int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    int index = y * width + x;
    float data = img_f1_d[index];
    img_u1_d[index] = data < 255.f ? (uint8_t)data : 255;
}

__global__ void
img_uchar3_to_float4(const uchar3* const img_u3_d,
                     float4* const img_f4_d,
                     const int width,
                     const int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    int index = y * width + x;

    uchar3 data = img_u3_d[index];
    img_f4_d[index] = make_float4(data.x, data.y, data.z, 1);
}

__global__ void
img_float4_to_uchar3(const float4* const img_f4_d,
                     uchar3* const img_u3_d,
                     const int width,
                     const int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    int index = y * width + x;
    float4 val = img_f4_d[index];
    int val_x = val.x;
    int val_y = val.y;
    int val_z = val.z;
    val_x = val_x >= 0 ? val_x : 0;
    val_y = val_y >= 0 ? val_y : 0;
    val_z = val_z >= 0 ? val_z : 0;
    val_x = val_x <= 255 ? val_x : 255;
    val_y = val_y <= 255 ? val_y : 255;
    val_z = val_z <= 255 ? val_z : 255;

    uchar3 data = make_uchar3(val_x, val_y, val_z);
    img_u3_d[index] = data;
}

__global__ void
img_float4_to_float1(const float4* const img_f4_d,
                     float* const img_f1_d,
                     const int width,
                     const int height,
                     const int channel)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    int index = y * width + x;
    float4 val = img_f4_d[index];
    switch (channel) {
        case 0:
            img_f1_d[index] = val.x;
            break;
        case 1:
            img_f1_d[index] = val.y;
            break;
        case 2:
            img_f1_d[index] = val.z;
            break;
        case 3:
            img_f1_d[index] = val.w;
            break;
        default:
            break;
    }
}

__global__ void
img_float1_to_float4(const float* const img_f1_d,
                     float4* const img_f4_d,
                     const int width,
                     const int height,
                     const int channel)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    int index = y * width + x;
    float val = img_f1_d[index];
    switch (channel) {
        case 0:
            img_f4_d[index].x = val;
            break;
        case 1:
            img_f4_d[index].y = val;
            break;
        case 2:
            img_f4_d[index].z = val;
            break;
        case 3:
            img_f4_d[index].w = val;
            break;
        default:
            break;
    }
}

__global__ void
img_uchar3_to_float1(const uchar3* const img_u3_d,
                     float* const img_f1_d,
                     const int width,
                     const int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    int index = y * width + x;

    uchar3 data = img_u3_d[index];

    int im_size = width * height;
    img_f1_d[index] = data.x;
    img_f1_d[index + im_size] = data.y;
    img_f1_d[index + im_size * 2] = data.z;
    img_f1_d[index + im_size * 3] = 1;
}

__global__ void
img_uchar_to_float(const uchar* const img_u1_d,
                     float* const img_f1_d,
                     const int width,
                     const int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    int index = y * width + x;

    img_f1_d[index] = img_u1_d[index];
}

__global__ void
img_float1_to_uchar3(const float* const img_f1_d,
                     uchar3* const img_u3_d,
                     const int width,
                     const int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    int index = y * width + x;
    int im_size = width * height;

    int val_x = __float2int_rd(img_f1_d[index]);
    int val_y = __float2int_rd(img_f1_d[index + im_size]);
    int val_z = __float2int_rd(img_f1_d[index + im_size * 2]);
    val_x = val_x >= 0 ? val_x : 0;
    val_y = val_y >= 0 ? val_y : 0;
    val_z = val_z >= 0 ? val_z : 0;
    val_x = val_x <= 255 ? val_x : 255;
    val_y = val_y <= 255 ? val_y : 255;
    val_z = val_z <= 255 ? val_z : 255;

    uchar3 data = make_uchar3(val_x, val_y, val_z);
    img_u3_d[index] = data;
}

__global__ void
img_BGR_to_LAB(float4* const img_d, const int width, const int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    int index = y * width + x;
    img_d[index] = BGR_to_LAB(img_d[index]);
}

__global__ void
img_LAB_to_BGR(float4* const img_d, const int width, const int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    int index = y * width + x;
    img_d[index] = LAB_to_BGR(img_d[index]);
}

__global__ void
sobel_grad(const float4* const img_d,
           float4* const out_d,
           const int S,
           const int block_w,
           const int block_h,
           const int width,
           const int height)
{
    extern __shared__ float sobel_smem[];
    // copy data into shared memory
    int x = blockIdx.x * block_w + threadIdx.x - S;
    int y = blockIdx.y * block_h + threadIdx.y - S;
    x = min(max(0, x), width - 1);
    y = min(max(0, y), height - 1);

    int index = y * width + x;
    int tile_index = threadIdx.y * blockDim.x + threadIdx.x;
    float4 pix = img_d[index];
    // BGR to grey
    sobel_smem[tile_index] = pix.x * 0.114 + pix.y * 0.587 + pix.z * 0.299;
    __syncthreads();

    if (threadIdx.x >= S && threadIdx.x < blockDim.x - S && threadIdx.y >= S &&
        threadIdx.y < blockDim.y - S) {
        float pix00 = sobel_smem[tile_index - blockDim.x - 1];
        float pix01 = sobel_smem[tile_index - blockDim.x];
        float pix02 = sobel_smem[tile_index - blockDim.x + 1];
        float pix10 = sobel_smem[tile_index - 1];
        float pix12 = sobel_smem[tile_index + 1];
        float pix20 = sobel_smem[tile_index + blockDim.x - 1];
        float pix21 = sobel_smem[tile_index + blockDim.x];
        float pix22 = sobel_smem[tile_index + blockDim.x + 1];
        float4 out;
        out.w =
          sobel_kernel(pix00, pix01, pix02, pix10, pix12, pix20, pix21, pix22);
        out.x = pix.x;
        out.y = pix.y;
        out.z = pix.z;
        out_d[index] = out;
    }
}

__global__ void
gauss_blur(const float4* const img_d,
           float4* const out_d,
           const float* const filter_d,
           const int S,
           const int block_w,
           const int block_h,
           const int width,
           const int height)
{
    const int filter_step = 2 * S + 1;
    extern __shared__ float4 gauss_smem[];
    // copy filter weights into shared memory
    float4* filter_s = &gauss_smem[blockDim.x * blockDim.y];
    if (threadIdx.x < filter_step && threadIdx.y < filter_step) {
        int filter_id = threadIdx.y * filter_step + threadIdx.x;
        filter_s[filter_id].w = filter_d[filter_id];
    }
    __syncthreads();
    // copy data into shared memory
    int x = blockIdx.x * block_w + threadIdx.x - S;
    int y = blockIdx.y * block_h + threadIdx.y - S;
    x = min(max(0, x), width - 1);
    y = min(max(0, y), height - 1);

    int index = y * width + x;
    int tile_index = threadIdx.y * blockDim.x + threadIdx.x;
    gauss_smem[tile_index] = img_d[index];
    __syncthreads();

    if (threadIdx.x >= S && threadIdx.x < blockDim.x - S && threadIdx.y >= S &&
        threadIdx.y < blockDim.y - S) {
        float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int r = -S; r <= S; ++r)
            for (int c = -S; c <= S; ++c) {
                sum += gauss_smem[tile_index + r * blockDim.x + c] *
                       filter_s[(r + S) * filter_step + c + S].w;
            }
        out_d[index] = sum;
    }
}

void
compute_gauss_kernel(float* const filter_d, const double sigma, const int S)
{
    const int filter_size = (2 * S + 1) * (2 * S + 1);
    float* filter_f32_h = new float[filter_size];
    double* filter_f64_h = new double[filter_size];

    double sum = 0;
    for (int r = -S; r <= S; ++r)
        for (int c = -S; c <= S; ++c) {
            double val = exp(-(double)(r * r + c * c) / (2 * sigma * sigma));
            filter_f64_h[(r + S) * (2 * S + 1) + c + S] = val;
            sum += val;
        }
    sum = 1. / sum;
    // normalize
    for (int r = -S; r <= S; ++r)
        for (int c = -S; c <= S; ++c)
            filter_f64_h[(r + S) * (2 * S + 1) + c + S] *= sum;

    for (int i = 0; i < filter_size; i++)
        filter_f32_h[i] = static_cast<float>(filter_f64_h[i]);

    cudaMemcpy(filter_d,
               filter_f32_h,
               sizeof(float) * filter_size,
               cudaMemcpyHostToDevice);
    delete[] filter_f64_h;
    delete[] filter_f32_h;
}

void
img_gauss_blur(float4*& img_d,
               float4*& buf_d,
               float* const filter_d,
               const double sigma,
               const int width,
               const int height)
{
    const int S = static_cast<int>(round(3 * sigma));
    compute_gauss_kernel(filter_d, sigma, S);

    const int block_w = 16;
    const int block_h = 16;
    const int tile_w = block_w + 2 * S;
    const int tile_h = block_h + 2 * S;

    const dim3 grids((width + block_w - 1) / block_w,
                     (height + block_h - 1) / block_h);
    const dim3 blocks(tile_w, tile_h);
    const size_t shared_size =
      (tile_w * tile_h + (2 * S + 1) * (2 * S + 1)) * sizeof(float4);
    gauss_blur<<<grids, blocks, shared_size>>>(
      img_d, buf_d, filter_d, S, block_w, block_h, width, height);
    std::swap(img_d, buf_d);
}

void
img_sobel_grad(float4*& img_d,
               float4*& buf_d,
               const int width,
               const int height)
{
    // For sobel 3 x 3, S = 1
    const int S = 1;
    const int block_w = 16;
    const int block_h = 16;
    const int tile_w = block_w + 2 * S;
    const int tile_h = block_h + 2 * S;
    const dim3 grids((width + block_w - 1) / block_w,
                     (height + block_h - 1) / block_h);
    const dim3 blocks(tile_w, tile_h);
    const size_t shared_size = (tile_w * tile_h) * sizeof(float);
    sobel_grad<<<grids, blocks, shared_size>>>(
      img_d, buf_d, S, block_w, block_h, width, height);
    std::swap(img_d, buf_d);
}
