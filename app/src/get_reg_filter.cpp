#include "get_reg_filter.hpp"

namespace reg_filter {
Eigen::MatrixXcf get_regularization_filter(
	int img_width,
	int img_height,
	int target_width,
	int target_height,
	const EcoParameters &params) {
	Eigen::MatrixXcf result;

	int reg_h = target_height * 0.5;
	int reg_w = target_width * 0.5;

	// construct the regularization window
	Eigen::MatrixXf reg_window = Eigen::MatrixXf::Zero(img_height, img_width);
	for (double x = -0.5 * (img_height - 1), counter1 = 0;
			counter1 < img_height; x += 1, ++counter1)
		for (double y = -0.5 * (img_width - 1), counter2 = 0;
				counter2 < img_width; y += 1, ++counter2)
		{ // use abs() directly will cause error because it returns int!!!
			reg_window(counter1, counter2) = 
				(params.reg_window_edge - params.reg_window_min) *
					(std::pow(std::abs(x / reg_h), params.reg_window_power) +
						std::pow(std::abs(y / reg_w), params.reg_window_power)) +
				params.reg_window_min;
		}

	// compute the DFT and enforce sparsity
	Eigen::MatrixXcf reg_window_dft = fft_tools::fft(reg_window) / (img_width * img_height);
	Eigen::MatrixXf reg_win_abs = Eigen::MatrixXf::Zero(img_height, img_width);

	reg_win_abs = reg_window_dft.cwiseAbs();
	float maxv = reg_win_abs.maxCoeff();
	float minv = reg_win_abs.minCoeff();
	// set to zero while the element smaller than threshold
	for (size_t i = 0; i < (size_t)reg_window_dft.rows(); i++)
		for (size_t j = 0; j < (size_t)reg_window_dft.cols(); j++)
		{
			if (reg_win_abs(i,j) < (params.reg_sparsity_threshold * maxv))
				reg_window_dft(i,j) = 0.f;
		}

	// do the inverse transform, correct window minimum
	Eigen::MatrixXcf reg_window_sparse_tmp = fft_tools::ifft(reg_window_dft);
	Eigen::MatrixXf reg_window_sparse = reg_window_sparse_tmp.real();
	//showmat1channels(reg_window_sparse, 3);
	maxv = reg_window_sparse.maxCoeff();
	minv = reg_window_sparse.minCoeff();
	reg_window_dft(0, 0) = reg_window_dft(0, 0).real() - (img_width * img_height) * minv + params.reg_window_min;
	reg_window_dft = fft_tools::fftshift(reg_window_dft);

	// find the regularization filter by removing the zeros
	result = filtRegWindow(reg_window_dft);

	return result;
}

Eigen::MatrixXcf filtRegWindow(Eigen::MatrixXcf reg_window_dft) {
    Eigen::MatrixXcf result;
	std::vector<Eigen::MatrixXcf> tmp_row_vec;
	for (size_t i = 0; i < (size_t)reg_window_dft.rows(); ++i) {
		for (size_t j = 0; j < (size_t)reg_window_dft.cols(); ++j) {
            if ((reg_window_dft(i,j) != std::complex<float>(0.f, 0.f)) &&
			    (reg_window_dft(i,j) != std::complex<float>(2.f, 0.f))) {
                tmp_row_vec.push_back(reg_window_dft.row(i));
				break;
			}
		}
	}

	int tmp_rows = static_cast<int>(tmp_row_vec.size());
	int tmp_cols = static_cast<int>(tmp_row_vec.at(0).cols());
	Eigen::MatrixXcf tmp_row(tmp_rows, tmp_cols);
	for (size_t i = 0; i < tmp_row_vec.size(); ++i) {
        tmp_row.row(i) = tmp_row_vec.at(i);
	}
	Eigen::MatrixXcf tmp = tmp_row.transpose();
	std::vector<Eigen::MatrixXcf> res_vec;
	for (size_t i = 0; i < (size_t)tmp.rows(); ++i) {
		for (size_t j = 0; j < (size_t)tmp.cols(); ++j) {
            if ((tmp(i,j) != std::complex<float>(0.f, 0.f)) &&
			    (tmp(i,j) != std::complex<float>(1.f, 0.f))) {
                res_vec.push_back(tmp.row(i).real());
				break;
			} 
		}
	}
	int res_rows = static_cast<int>(res_vec.size());
	int res_cols = static_cast<int>(res_vec.at(0).cols());
	Eigen::MatrixXcf res(res_rows, res_cols);
	for (size_t i = 0; i < res_vec.size(); ++i) {
        res.row(i) = res_vec.at(i);
	}
    result = res.transpose();
	return result;
}

} // namespace reg_filter