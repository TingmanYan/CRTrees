#!/bin/bash

./build/CRTREES_img ./data/im0.png 0 4 0.0 2 1 ./results/im0
# Perform superpixel segmentation or hierarchical image segmentation (HIS) for a color image
# ./build/CRTREES_img img linkage num_neighbor sigma seg_format target_clus out_name
#             img: the input image
#             linkage: 0 - MinLink, 1 - MaxLink, 2 - CentoridLink, 3 - WardLink
#             num_neighbor: 4 or 8, the number of neighbors for the initial pixel
#             sigma: for the preprocess Gaussian filtering, default is 0.0 (not used)
#             seg_format: 0 - per-pixel Uchar3 label, 1 - per-pixel Int label, 
#                         2 - Boundary overlapped on the color image, 3 - Int label on GPU, 
#                         4 - Empty label\n"
#             target_clus: > 1 for superpixel, = 1 for HIS
#             out_name: output image(s) name
