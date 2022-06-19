#include "segment.hpp"

int
main(int argc, char** argv)
{
    if (argc != 6) {
        std::cerr << "usage: perform superpixel segmentation or HIS for video streams " << argv[0]
                  << "linkage num_neighbor sigma seg_format target_clus\n"
                     "linkage: 0 - MinLink, 1 - MaxLink, 2 - CentoridLink, 3 - WardLink\n"
                     "num_neighbor: 4 or 8, the number of neighbors for the initial pixel\n"
                     "sigma: for the preprocess Gaussian filtering, default is 0.0 (not used)\n"
                     "seg_format: 0 - per-pixel Uchar3 label, 1 - per-pixel Int label, "
                     "2 - Boundary overlapped on the color image, 3 - Int label on GPU, "
                     "4 - Empty label\n"
                     "target_clus: > 1 for superpixel, = 1 for HIS\n";
        exit(1);
    }

    Linkage link = static_cast<Linkage>(atoi(argv[1]));
    int num_nb = atoi(argv[2]);
    assert(num_nb==4 || num_nb==8);
    double sigma = atof(argv[3]);
    SegFormat seg_format = static_cast<SegFormat>(atoi(argv[4]));
    int target_clus = atoi(argv[5]);
    std::cout << "target number of clusters: " << target_clus << std::endl;

    cv::VideoCapture cap;
    cap.open(0);
    if (!cap.isOpened()) {
        std::cerr << "ERROR! Unable to open camera!" << std::endl;
        return -1;
    }

    cv::Mat frame;
    cap.read(frame);
    cv::resize(frame, frame, cv::Size(852, 480));

    std::vector<cv::Mat> segmentations;
    SegHAC* seg_hac =
      new SegHAC(frame, link, num_nb, sigma, seg_format, target_clus, false);

    std::cout << "Start grabbing" << std::endl;
    std::cout << "Press 'q' to terminate" << std::endl;

    do {
        segmentations.clear();
        cap.read(frame);
        cv::resize(frame, frame, cv::Size(852, 480));
        seg_hac->set_frame(frame);

        auto c0 = std::chrono::steady_clock::now();
        seg_hac->run_ms(&segmentations);
        cudaDeviceSynchronize();
        auto c1 = std::chrono::steady_clock::now();
        std::cout << "time for hierarchical image segmentation: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(c1 -
                                                                           c0)
                       .count()
                  << " ms" << std::endl;

        int num_images = 0;
        for (auto& seg : segmentations) {
            cv::imshow(cv::format("level_%03d", num_images++), seg);
        }
    } while (cv::waitKey(1) != 'q');

    delete seg_hac;
    return 0;
}
