cmake_minimum_required(VERSION 2.8.3)
project(eco_tracker)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g -march=native")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR} ./
                    /usr/local/include/opencv/
					./featuretools/inc/
                    ./mathtools/inc/
                    ./moduletest/
					./app/inc/
)

find_package(OpenCV REQUIRED)
add_executable(feature_extractor
    moduletest/feature_extractor_test.cpp
    app/src/feature_extractor.cpp
    featuretools/src/fhog.cpp
    mathtools/src/ffttools.cpp
    )

target_link_libraries(feature_extractor ${OpenCV_LIBS})

add_executable(conv_test
    moduletest/convolution_test.cpp
    mathtools/src/ffttools.cpp
    )

target_link_libraries(conv_test ${OpenCV_LIBS})

add_executable(eco_tracker
    app/src/run_eco_tracker.cpp
    app/src/eco_util.cpp
    app/src/eco.cpp
    app/src/get_reg_filter.cpp
    app/src/interpolator.cpp
    app/src/optimize_scores.cpp
    app/src/sample_space.cpp
    app/src/train_filter.cpp
    app/src/feature_extractor.cpp
    featuretools/src/fhog.cpp
    mathtools/src/matrix_operator.cpp
    mathtools/src/ffttools.cpp
    )

target_link_libraries(eco_tracker ${OpenCV_LIBS})


