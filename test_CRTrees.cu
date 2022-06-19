#include "CRTrees.hpp"
#include "util.h"
#include <experimental/random>
#include <fstream>

int
main(int argc, char** argv)
{
    if (argc != 3) {
        std::cerr << "usage: " << argv[0] << " N trails\n";
        exit(1);
    }
    const int N = atoi(argv[1]);
    const int trails = atoi(argv[2]);
    int* nn_h = new int[N];

    int *nn_d, *clusters_d, *mask_cycle_d;
    checkCudaErrors(cudaMalloc(&nn_d, sizeof(int) * N));
    checkCudaErrors(cudaMalloc(&clusters_d, sizeof(int) * N));
    checkCudaErrors(cudaMalloc(&mask_cycle_d, sizeof(int) * N));

    CRTrees* crtrees = new CRTrees(N);

    auto c0 = std::chrono::steady_clock::now();
    auto delta_c = c0 - c0;
    for (int i=0; i<trails; ++i) {

        for (int j=0; j<N; ++j) {
            int rn = std::experimental::randint(0, N-2);
            if (rn >= j)
                rn++;
            nn_h[j] = rn;
        }
        // std::ofstream out_rand;
        // out_rand.open("rand_1nn.mtx",std::ios::trunc);
        // out_rand << "%%MatrixMarket matrix coordinate pattern general" << std::endl;
        // out_rand << N << " " << N << " "<< N * 2 << " " << std::endl;
        // for (int j=0;j<N; ++j) {
            // out_rand << nn_h[j] + 1<< " " << j + 1 << std::endl;
            // out_rand << j + 1<< " " << nn_h[j] + 1 << std::endl;
        // }
        // out_rand.close();
        checkCudaErrors(
          cudaMemcpy(nn_d, nn_h, sizeof(int) * N, cudaMemcpyHostToDevice));

        c0 = std::chrono::steady_clock::now();
        crtrees->get_clus(nn_d, clusters_d, N, nullptr);
        cudaDeviceSynchronize();
        auto c1 = std::chrono::steady_clock::now();
        delta_c += c1 - c0;
    }
    int num_clus = crtrees->compact_clus_label(clusters_d, N);
    std::cout << "Time for CRTrees labeling to label " << N << " vertices: "
              << std::chrono::duration_cast<std::chrono::microseconds>(delta_c).count() / 1e3 / trails
              << " ms" << std::endl;
    std::cout << "number of clusters: " << num_clus << std::endl;

    int* clusters_h = new int[N];
    checkCudaErrors(cudaMemcpy(
      clusters_h, clusters_d, sizeof(int) * N, cudaMemcpyDeviceToHost));

    if (N <= 1024) {
        std::cout << "output labels" << std::endl;
        for (int i = 0; i < N; ++i) {
            std::cout << clusters_h[i] + 1 << " ";
        }
        std::cout << std::endl;

        int* centers = new int[N]{ 0 };
        for (int i = 0; i < N; ++i) {
            centers[clusters_h[i]] = 1;
        }

        for (int i = 0; i < N; ++i) {
            if (centers[i]) {
                std::cout << "possiable CC root: " << i + 1;
                if (i == clusters_h[i])
                    std::cout << ",  confirmed" << std::endl;
                else
                    std::cout << ",  not confirmed" << std::endl;
            }
        }
        delete[] centers;

        int* mask_cycle_h = new int[N];
        checkCudaErrors(cudaMemcpy(
          mask_cycle_h, mask_cycle_d, sizeof(int) * N, cudaMemcpyDeviceToHost));

        std::cout << "cycle roots" << std::endl;
        for (int i = 0; i < N; ++i)
            if (mask_cycle_h[i])
                std::cout << i + 1 << " ";
        std::cout << std::endl;
        delete[] mask_cycle_h;
    }

    checkCudaErrors(cudaFree(mask_cycle_d));
    checkCudaErrors(cudaFree(clusters_d));
    checkCudaErrors(cudaFree(nn_d));

    delete[] clusters_h;
    delete[] nn_h;
    delete crtrees;

    return 0;
}
