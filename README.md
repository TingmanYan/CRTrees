# Hierarchical Image Segmentation Superpixel Segmentation by Parallel CRTrees Labeling
# Workflow
![CRTREES](workflow.png)
# Dependency
- CUDA >= 6.0
- OpenCV >= 3.0

The code has been tested on ubuntu.
# Usage
Compile the code
```
unzip lib_eval.zip // this is for evaluation on datasets
bash build.sh
```
Test on a single image (the first run takes time to load the GPU)
```
bash run_img.sh
```
Video streams (a web camera is required)
```
bash run_video.sh
```
It can achieve about 20fps for 480P video streams on an NVIDIA MX350 GPU and consume less than 200MB GPU memory. 200+fps can be achieved on a Titan Xp GPU. Faster speed can be achieved if the GPU version of OpenCV is used.
## Benchmark
The superpixel benchmark (https://github.com/davidstutz/superpixel-benchmark) shall be put in the same level directory as this project.
```
bash bench_superpixels.sh BSDS500
```
## The same results as in the paper can be obtained.