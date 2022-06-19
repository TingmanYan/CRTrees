#include "segment.hpp"

int
main(int argc, char** argv)
{
    if (argc != 8) {
        std::cerr
          << "usage: perform superpixel segmentation or HIS for a color image " << argv[0]
          << " img linkage num_neighbor sigma seg_format target_clus out_name\n"
             "img: the input image\n"
             "linkage: 0 - MinLink, 1 - MaxLink, 2 - CentoridLink, 3 - WardLink\n"
             "num_neighbor: 4 or 8, the number of neighbors for the initial pixel\n"
             "sigma: for the preprocess Gaussian filtering, default is 0.0 (not used)\n"
             "seg_format: 0 - per-pixel Uchar3 label, 1 - per-pixel Int label, "
             "2 - Boundary overlapped on the color image, 3 - Int label on GPU, "
             "4 - Empty label\n"
             "target_clus: > 1 for superpixel, = 1 for HIS\n"
             "out_name: output image(s) name\n";
        exit(1);
    }
    // load input
    cv::Mat image = cv::imread(argv[1], cv::IMREAD_COLOR);

    Linkage link = static_cast<Linkage>(atoi(argv[2]));
    int num_nb = atoi(argv[3]);
    assert(num_nb==4 || num_nb==8);
    double sigma = atof(argv[4]);
    SegFormat seg_format = static_cast<SegFormat>(atoi(argv[5]));
    int target_clus = atoi(argv[6]);
    std::cout << "target number of clusters: " << target_clus << std::endl;

    std::vector<cv::Mat> segmentations;
    SegHAC* seg_hac =
      new SegHAC(image, link, num_nb, sigma, seg_format, target_clus, false);
    auto c0 = std::chrono::steady_clock::now();
    seg_hac->run_ms(&segmentations);
    cudaDeviceSynchronize();
    auto c1 = std::chrono::steady_clock::now();
    std::cout
      << "time for hierarchical image segmentation: "
      << std::chrono::duration_cast<std::chrono::milliseconds>(c1 - c0).count()
      << " ms" << std::endl;

    cv::String out_name(argv[7]);
    int num_images = 0;
    for (auto& seg : segmentations)
        cv::imwrite(out_name + cv::format("%03d.png", num_images++), seg);

    delete seg_hac;
    return 0;
}
