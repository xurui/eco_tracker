#ifndef _INTERPOLATOR_HPP_
#define _INTERPOLATOR_HPP_

#include <iostream>
#include <iomanip>
#include <complex>
#include <cmath>
#include <Eigen/Dense>
#include <eigen3/Eigen/Dense>

namespace eco_tracker {

	void getInterpFourier(const int& filter_width,
						  const int& filter_height,
						  Eigen::MatrixXcf& interp1_fs,
						  Eigen::MatrixXcf& interp2_fs, 
						  float a);

    Eigen::MatrixXcf CubicSplineFourier(Eigen::MatrixXcf f, float a);
} // namespace eco_tracker

#endif