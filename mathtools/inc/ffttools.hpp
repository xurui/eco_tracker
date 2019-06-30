#ifndef _FFTTOOLS_HPP_
#define _FFTTOOLS_HPP_

#include <iostream>
#include <iomanip>
#include <complex>
#include <cmath>
#include <Eigen/Dense>
#include <eigen3/Eigen/Dense>
#include <unsupported/Eigen/FFT>

namespace fft_tools {

Eigen::MatrixXcf fft(const Eigen::MatrixXcf& timeMat);

Eigen::MatrixXcf fft(const Eigen::MatrixXf& timeMat);

Eigen::MatrixXcf ifft(const Eigen::MatrixXcf& freqMat);

Eigen::MatrixXcf fftshift(Eigen::MatrixXcf x, bool inverse_flag = false);

Eigen::MatrixXcf circshift(Eigen::MatrixXcf data, int shift_r, int shift_c);

/**
 * @brief 
 * 
 * @tparam Scalar 
 * @tparam SizeX 
 * @tparam  
 * @tparam KSizeX 
 * @tparam KSizeY 
 * @param I 
 * @param kernel 
 * @param border_type 0: zero 1: border
 * @param conv_mode 0: full 1: valid
 * @return Eigen::Matrix< Scalar, SizeX, SizeY > 
 */
Eigen::MatrixXcf  Convolution2(
    const Eigen::MatrixXcf &I,
    const Eigen::MatrixXcf &kernel,
    int border_type,
    int conv_mode);

Eigen::MatrixXcf  Convolution2(
    const Eigen::MatrixXcf &I,
    const Eigen::MatrixXcf &kernel,
    int conv_mode = 0);

std::complex<float> getMatValue(
    const Eigen::MatrixXcf& mat,
    int r, int c, const int& border_type);
}

#endif