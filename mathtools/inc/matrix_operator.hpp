#ifndef _MATRIX_OPERATOR_HPP_
#define _MATRIX_OPERATOR_HPP_

#include <iostream>
#include <iomanip>
#include <complex>
#include <cmath>
#include <Eigen/Dense>
#include <eigen3/Eigen/Dense>
#include <unsupported/Eigen/FFT>

typedef std::vector<std::vector<Eigen::MatrixXcf> > EcoFeats;

struct ECO_Train {
    EcoFeats part1;
    std::vector<Eigen::MatrixXcf> part2;
};

EcoFeats operator+(const EcoFeats &feats1, const EcoFeats &feats2);

EcoFeats operator-(const EcoFeats &feats1, const EcoFeats &feats2);

EcoFeats operator*(const EcoFeats &feats, const float& coef);

std::vector<Eigen::MatrixXcf> operator+(
    const std::vector<Eigen::MatrixXcf>& data1,
    const std::vector<Eigen::MatrixXcf>& data2);

std::vector<Eigen::MatrixXcf> operator-(
    const std::vector<Eigen::MatrixXcf>& data1,
    const std::vector<Eigen::MatrixXcf>& data2);

std::vector<Eigen::MatrixXcf> operator*(
    const std::vector<Eigen::MatrixXcf>& data,
    const float& coef);

ECO_Train operator+(const ECO_Train& data1, const ECO_Train& data2);

ECO_Train operator-(const ECO_Train& data1, const ECO_Train& data2);

ECO_Train operator*(const ECO_Train& data, const float& scale);

#endif