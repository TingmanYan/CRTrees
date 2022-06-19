#!/bin/bash
if [ ! -d "build" ]; then
    mkdir build
    cd build
    cmake .. -DCMAKE_CUDA_FLAGS="-lineinfo"
    # cmake ..
else
    cd build
fi
make -j 12

#cuda-memcheck ./CRTREES_img ../data/im0.png 3 8 0.0 2 1 ../results/im0
