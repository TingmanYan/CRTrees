#!/bin/bash

./build/CRTREES_video 3 4 0.0 2 1
# Perform superpixel segmentation or hierarchical image segmentation (HIS) for video streams
# ./build/CRTREES_video linkage num_neighbor sigma seg_format target_clus
#             linkage: 0 - MinLink, 1 - MaxLink, 2 - CentoridLink, 3 - WardLink
#             num_neighbor: 4 or 8, the number of neighbors for the initial pixel
#             sigma: for the preprocess Gaussian filtering, default is 0.0 (not used)
#             seg_format: 0 - per-pixel Uchar3 label, 1 - per-pixel Int label, 
#                         2 - Boundary overlapped on the color image, 3 - Int label on GPU, 
#                         4 - Empty label\n"
#             target_clus: > 1 for superpixel, = 1 for HIS
# press 'q' to quit
