#ifndef _GET_REG_FILTER_HPP
#define _GET_REG_FILTER_HPP

#include <cmath>

#include "parameters.hpp"
#include "ffttools.hpp"

namespace reg_filter {

Eigen::MatrixXcf get_regularization_filter(
	int img_width,
	int img_height,
	int target_width,
	int target_height,
	const EcoParameters &params);

Eigen::MatrixXcf filtRegWindow(Eigen::MatrixXcf reg_window_dft);
} // namespace reg_filter
#endif