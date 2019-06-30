#ifndef _ECO_UTIL_HPP_
#define _ECO_UTIL_HPP_

#include <iostream>
#include <string>
#include <math.h>
#include <algorithm>
#include <complex>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include "ffttools.hpp"
#include "matrix_operator.hpp"

namespace eco_tracker {

typedef std::vector<std::vector<Eigen::MatrixXcf> > EcoFeats;

EcoFeats runFFt(const EcoFeats &xlw);

void genGaussianYf(
    const float& sigma_y,
    const float& T_size,
    const std::vector<Eigen::MatrixXcf>& k_x,
    const std::vector<Eigen::MatrixXcf>& k_y,
    std::vector<Eigen::MatrixXcf>& y_f);

void genCosWindow(
    const int& row,
    const int& col,
    Eigen::MatrixXcf& cos_window);

EcoFeats initInterpolateFFt(
    const EcoFeats& xlf,
    const std::vector<Eigen::MatrixXcf>& interp1_fs,
    const std::vector<Eigen::MatrixXcf>& interp2_fs);

EcoFeats interpolateDFT(
    const EcoFeats& xlf,
    const std::vector<Eigen::MatrixXcf>& interp1_fs,
    const std::vector<Eigen::MatrixXcf>& interp2_fs);

EcoFeats computeFeautrePower(const EcoFeats &feats);

EcoFeats compactFourierCoeff(const EcoFeats &xf);

EcoFeats fullFourierCoeff(const EcoFeats &xf);

std::vector<Eigen::MatrixXcf> vectorFeature(const EcoFeats &x);

void initProjectionMatrix(
    const EcoFeats& init_sample,
	const std::vector<int>& compressed_dim,
    std::vector<Eigen::MatrixXcf>& project_matrix);

EcoFeats projectFeature(
    const EcoFeats& x, 
    const std::vector<Eigen::MatrixXcf>& project_matrix);

EcoFeats projectFeatureMultScale(
    const EcoFeats& x, 
    const std::vector<Eigen::MatrixXcf>& project_matrix);

EcoFeats EcoFeatureDotDivide(
    const EcoFeats &a,
    const EcoFeats &b);

std::complex<float> EcoFeatInnerProduct(
    const EcoFeats& f1,
    const EcoFeats& f2);

EcoFeats shiftSample(EcoFeats &xf,
                     float x,
                     float y,
                     std::vector<Eigen::MatrixXcf> kx,
                     std::vector<Eigen::MatrixXcf> ky);

std::vector<Eigen::MatrixXcf> getProjectMatEnergy(
    const std::vector<Eigen::MatrixXcf> project_mat,
    const std::vector<int>& feature_dim,
	const std::vector<Eigen::MatrixXcf>& yf);

} // namespace eco_tracker
#endif