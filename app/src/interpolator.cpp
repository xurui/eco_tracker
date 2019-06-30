#include "interpolator.hpp"

namespace eco_tracker {

void getInterpFourier(const int& filter_width,
					  const int& filter_height,
					  Eigen::MatrixXcf& interp1_fs,
					  Eigen::MatrixXcf& interp2_fs, 
					  float a) {

    Eigen::MatrixXcf temp1(filter_height, 1);
	Eigen::MatrixXcf temp2(1, filter_width);

	for (int j = 0; j < temp1.rows(); j++) {
		temp1(j, 0) = j - temp1.rows() / 2;
	}
	for (int j = 0; j < temp2.cols(); j++) {
		temp2(0, j) = j - temp2.cols() / 2;
	}

	interp1_fs = CubicSplineFourier(temp1 / filter_height, a) / filter_height;
	interp2_fs = CubicSplineFourier(temp2 / filter_width, a) / filter_width;

	// Multiply Fourier coeff with e ^ (-i*pi*k / N):[cos(pi*k/N), -sin(pi*k/N)]
	Eigen::MatrixXcf result1 = Eigen::MatrixXcf::Zero(temp1.rows(), temp1.cols());
	Eigen::MatrixXcf result2 = Eigen::MatrixXcf::Zero(temp1.rows(), temp1.cols());

	temp1 = temp1 / filter_height;
	temp2 = temp2 / filter_width;
	for (int r = 0; r < temp1.rows(); ++r) {
        for (int c = 0; c < temp1.cols(); ++c) {
			result1(r,c) = std::cos(temp1(r,c) * static_cast<float>(M_PI));
			result2(r,c) = std::sin(temp1(r,c) * static_cast<float>(M_PI));
		}
	}

	//cv::Mat planes1[] = {interp1_fs.mul(result1), -interp1_fs.mul(result2)};
	//cv::merge(planes1, 2, interp1_fs);
	//interp2_fs = interp1_fs.t();
	Eigen::MatrixXcf temp = Eigen::MatrixXcf::Zero(interp1_fs.rows(), interp1_fs.cols());
	Eigen::MatrixXcf tempT = Eigen::MatrixXcf::Zero(interp1_fs.cols(), interp1_fs.rows());
	for(int r = 0; r < temp1.rows(); r++) {
		for(int c = 0; c < temp1.cols(); c++) {
			float t_real = (interp1_fs(r, c) * result1(r, c)).real();
			float t_imag = -(interp1_fs(r, c) * result2(r, c)).real();
            std::complex<float> temp_cf(t_real, t_imag);
			temp(r, c) = temp_cf;
			tempT(c, r) = temp(r, c);
		}
	}
	interp1_fs = temp;
	interp2_fs = tempT;
}

Eigen::MatrixXcf CubicSplineFourier(Eigen::MatrixXcf f, float a) {

	if (f.rows() < 1) {
		assert(0 && "error: input mat is empty!");
	}

	Eigen::MatrixXcf bf(f.rows(), f.cols());
	for(int r = 0; r < bf.rows(); r++) {
		for(int c = 0; c < bf.cols(); c++) {
			bf(r, c) = 6.f * (1.f - std::cos(2.f * f(r, c) * static_cast<float>(M_PI))) 
			+ 3.f * a * (1.f - std::cos(4.f * f(r, c) * static_cast<float>(M_PI)))
			- (6.f + a * 8.f) * static_cast<float>(M_PI) * f(r, c) * std::sin(2.f * f(r, c) * static_cast<float>(M_PI))
			- 2.f * a * static_cast<float>(M_PI) * f(r, c) * std::sin(4.f * f(r, c) * static_cast<float>(M_PI));
			float L = 4.f * std::pow(f(r, c).real() * static_cast<float>(M_PI), 4);
			bf(r, c) /= L;
		}
	}
	bf(bf.rows() / 2, bf.cols() / 2) = 1;

	return bf;
}
} // namespace interpolator