# Graph-based Parallel Clustering for Hierarchical Image Segmentation
## Dependencies
- CUDA >= 6.0
- OpenCV >= 3.0

The code has been tested on ubuntu.
## Usage
compile the code
```
unzip lib_eval.zip // this is for evaluation on datasets
bash build.sh
```
test on a single image (the first run takes time to load the GPU)
```
bash run_img.sh
```
![level3_8](/data/level3-8.png)

The results of segmentation hierarchies from level 3 to 8.

Video streams (a web camera is required)
```
bash run_video.sh
```
It can achieve about 20fps for 480P video streams on an NVIDIA MX350 GPU and consume less than 200MB GPU memory. 200+fps can be achieved on a Titan Xp GPU.
## Benchmark
The superpixel benchmark [[repo](https://github.com/davidstutz/superpixel-benchmark)] shall be put in the same level dir as this repo. See the dir in `bench_superpixels.sh` for dietails.
```
bash bench_superpixels.sh BSDS500
```
The same results as in the paper can be obtained.