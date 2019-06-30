# eco_tracker

This is a C++ reimplementation of algorithm presented in "ECO Efficient Convolution Operators for Tracking" paper. For more info and implementation in Matlab visit [ECO tracker](https://github.com/martin-danelljan/ECO).

The code mainly depends on Eigen 3.3+ and OpenCV 2.4+ library and is build via cmake.

# Quick Start

If you want to compile and run the project, you can create a build folder first, then run the command:
```
mkdir build;
cd build;
cmake ..;
make;
run ./eco_tracker
```

## Some tips:
1. It uses fHoG feature to build feature map default, but you can extend to use CNN feature via setting USE_CNN;
2. About Deep features, this project supports extracting by Caffe and extracting by NCNN, the latter is suitable for ARM platform. For more info about NCNN, you can visit [NCNN](https://github.com/Tencent/ncnn);
3. Extracting Color Names features will be added in recently.

# Reference

Danelljan M, Bhat G, Shahbaz Khan F, et al. ECO: efficient convolution operators for tracking[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2017: 6638-6646.
