#include <numeric>
#include "eco_util.hpp"

namespace eco_tracker {

EcoFeats runFFt(const EcoFeats &xlw) {
	EcoFeats xlf;
	for (size_t i = 0; i < xlw.size(); i++) {
		std::vector<Eigen::MatrixXcf> temp;
		for (size_t j = 0; j < xlw[i].size(); j++)
		{
			int size = xlw[i][j].rows();
			Eigen::MatrixXf xlw_real = xlw[i][j].real();
			if (size % 2 == 1)
				temp.push_back(fft_tools::fftshift(fft_tools::fft(xlw_real)));
			else
			{
				Eigen::MatrixXcf xf = fft_tools::fftshift(fft_tools::fft(xlw_real));
                Eigen::MatrixXcf xf_pad = xf;
                xf_pad.conservativeResize(size + 1, size + 1);
				for (int k = 0; k < xf_pad.rows(); k++)
				{
					xf_pad(size, k) = std::conj(xf_pad(size - 1, k));
					xf_pad(k, size) = std::conj(xf_pad(k, size - 1));
				}
				temp.push_back(xf_pad);
			}
		}

		xlf.push_back(temp);
	}
	return xlf;
}

void genGaussianYf(
    const float& sigma_y,
    const float& T_size,
    const std::vector<Eigen::MatrixXcf>& k_x,
    const std::vector<Eigen::MatrixXcf>& k_y,
    std::vector<Eigen::MatrixXcf>& y_f) {

    assert(k_x.size() == k_y.size());
	float tmp1 = M_PI * sigma_y / T_size;
	float tmp2 = std::sqrt(2 * M_PI) * sigma_y / T_size;

	for (unsigned int i = 0; i < k_x.size(); i++) // for each filter
	{
		// 2 dimension version of (9)
        int row = static_cast<int>(k_y.at(i).rows());
        int col = static_cast<int>(k_x.at(i).cols());
		Eigen::MatrixXcf temp(row, col);
		for (int r = 0; r < row; r++)
		{
			std::complex<float> tempy = tmp1 * k_y.at(i)(r,0);
			tempy = tmp2 * std::exp(-2.0f * tempy * tempy);
			for (int c = 0; c < col; c++)
			{
				std::complex<float> tempx = tmp1 * k_x.at(i)(0,c);
				tempx = tmp2 * std::exp(-2.0f * tempx * tempx);
				temp(r, c) = tempy * tempx;
			}
		}
		y_f.push_back(temp);
	}
}

void genCosWindow(
    const int& row,
    const int& col,
    Eigen::MatrixXcf& cos_window) {

    Eigen::MatrixXcf temp(row, col);
    for (int r = 0; r < row; r++)
    {
        float tempy = 0.5f * (1 - std::cos(2 * M_PI * (float)(r + 1) / (col + 1)));
        for (int c = 0; c < col; c++)
        {
            temp(r, c) = tempy * 0.5f * (1 - std::cos(2 * (float)(c + 1) * M_PI / (row + 1)));
        }
    }
    cos_window = std::move(temp);
}

EcoFeats initInterpolateFFt(
    const EcoFeats& xlf,
    const std::vector<Eigen::MatrixXcf>& interp1_fs,
    const std::vector<Eigen::MatrixXcf>& interp2_fs) {

	EcoFeats result;
	for (size_t i = 0; i < xlf.size(); i++)
	{
        int interp1_N = interp1_fs[i].rows();
        int interp2_N = interp2_fs[i].cols();
		Eigen::MatrixXcf interp1_fs_mat(interp1_N, interp1_N);
        Eigen::MatrixXcf interp2_fs_mat(interp2_N, interp2_N);
		Eigen::MatrixXcf interp1_fs_i = interp1_fs[i];
		Eigen::MatrixXcf interp2_fs_i = interp2_fs[i];
		int org_1_r = interp1_fs_i.rows() - 1;
		int org_1_c = interp1_fs_i.cols() - 1;
        interp1_fs_mat = Eigen::MatrixXcf::NullaryExpr(interp1_N, interp2_N,
                         [&interp1_fs_i, &org_1_r, &org_1_c, interp1_N, interp2_N] (Eigen::Index i,Eigen::Index j) {
                         return interp1_fs_i(std::min<Eigen::Index>(org_1_r,i),
                                             std::min<Eigen::Index>(org_1_c,j) ); } );
		int org_2_r = interp2_fs_i.rows() - 1;
		int org_2_c = interp2_fs_i.cols() - 1;
        interp2_fs_mat = Eigen::MatrixXcf::NullaryExpr(interp1_N, interp2_N,
                         [&interp2_fs_i, &org_2_r, &org_2_c, interp1_N, interp2_N] (Eigen::Index i,Eigen::Index j) {
                         return interp2_fs_i(std::min<Eigen::Index>(org_2_r,i),
                                             std::min<Eigen::Index>(org_2_c,j) ); } );
		std::vector<Eigen::MatrixXcf> temp;
		for (size_t j = 0; j < xlf[i].size(); j++)
		{
			temp.push_back(interp1_fs_mat.cwiseProduct(xlf[i][j]).cwiseProduct(interp2_fs_mat));
		}
		result.push_back(temp);
	}
	return result;
}

EcoFeats computeFeautrePower(const EcoFeats &feats) {
	EcoFeats result;
	if (feats.empty()) {
		return feats;
	}

    // for each feature
	for (size_t i = 0; i < feats.size(); i++) {
		std::vector<Eigen::MatrixXcf> feat_vec;
        // for each dimension
		for (size_t j = 0; j < (size_t)feats[i].size(); j++) {
            Eigen::MatrixXcf temp = feats[i][j];
			for (size_t r = 0; r < (size_t)feats[i][j].rows(); r++) {
				for (size_t c = 0; c < (size_t)feats[i][j].cols(); c++) {
					temp(r, c) = std::pow(temp(r, c).real(), 2) + 
						std::pow(temp(r, c).imag(), 2);
				}
			}
			feat_vec.push_back(temp);
		}
		result.push_back(feat_vec);
	}
	return result;
}

// Take half of the fourier coefficient.
EcoFeats compactFourierCoeff(const EcoFeats &xf)
{
	EcoFeats result;
	for (size_t i = 0; i < xf.size(); i++) // for each feature
	{
		std::vector<Eigen::MatrixXcf> temp;
		for (size_t j = 0; j < xf[i].size(); j++) // for each dimension
			temp.push_back(xf[i][j].leftCols((xf[i][j].cols() + 1) / 2));
		result.push_back(temp);
	}
	return result;
}

// Get the full fourier coefficient of xf, using the property X(N-k)=conv(X(k))
EcoFeats fullFourierCoeff(const EcoFeats &xf)
{
	EcoFeats res;
	for (size_t i = 0; i < xf.size(); i++) // for each feature
	{
		std::vector<Eigen::MatrixXcf> tmp;
		for (size_t j = 0; j < xf[i].size(); j++) // for each dimension
		{
			Eigen::MatrixXcf temp_full = Eigen::MatrixXcf::Zero(xf[i][j].rows(), 2 * xf[i][j].cols() - 1);
			Eigen::MatrixXcf temp = xf[i][j].leftCols(xf[i][j].cols() - 1);
            Eigen::MatrixXcf temp_flip1 = temp.colwise().reverse().eval();
            Eigen::MatrixXcf temp_flip2 = temp_flip1.rowwise().reverse().eval();
            temp_full.leftCols(xf[i][j].cols()) = xf[i][j];
            temp_full.rightCols(xf[i][j].cols() - 1) = temp_flip2.conjugate();
			tmp.push_back(temp_full);
		}
		res.push_back(tmp);
	}

	return res;
}

void initProjectionMatrix(const EcoFeats& init_sample,
						  const std::vector<int>& compressed_dim,
                          std::vector<Eigen::MatrixXcf>& project_matrix)
{
	std::vector<Eigen::MatrixXcf> result;
	for (size_t i = 0; i < init_sample.size(); i++) // for each feature
	{
		// vectorize mat init_sample
		Eigen::MatrixXcf feat_vec(init_sample[i][0].rows() * init_sample[i][0].cols(),
                                 static_cast<int>(init_sample[i].size()));
		for (unsigned int j = 0; j < init_sample[i].size(); j++) // for each dimension of the feature
		{
			std::complex<float> mean = init_sample[i][j].mean(); // get the mean value of the mat;
            Eigen::MatrixXcf mean_mat = mean * 
                   Eigen::MatrixXcf::Ones(init_sample[i][j].rows(), init_sample[i][j].cols());
            Eigen::MatrixXcf tmp = init_sample[i][j] - mean_mat;
            Eigen::Map<Eigen::VectorXcf> tmp_v(tmp.data(), tmp.cols() * tmp.rows());
            feat_vec.col(j) = tmp_v;
		}
		result.push_back(feat_vec);
	}
	//printMat(result[0]); // 3844 x 31

	// do SVD and dimension reduction for each feature
    project_matrix.reserve(static_cast<int>(result.size()));
	for (size_t i = 0; i < result.size(); i++)
	{
        Eigen::MatrixXcf J = result.at(i).transpose() * result.at(i);
        Eigen::JacobiSVD<Eigen::MatrixXcf> svd(J, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::MatrixXcf V = svd.matrixV();
		project_matrix.push_back(V.leftCols(compressed_dim[i])); // get the previous compressed number components
	}
}
// Do projection row vector x_mat[i] * projection_matrix[i]
EcoFeats projectFeature(
    const EcoFeats& x, 
    const std::vector<Eigen::MatrixXcf>& project_matrix) {
    EcoFeats res;
    // for each feature
	for (size_t i = 0; i < x.size(); i++) {
		// vectorize the mat
        Eigen::MatrixXcf x_mat(static_cast<int>(x[i].size()), x[i][0].cols() * x[i][0].rows());
		std::vector<Eigen::MatrixXcf> temp;
		for (size_t j = 0; j < x[i].size(); j++) // for each channel of original feature
		{
			// transform x[i][j]:
			// x1 x4
			// x2 x5
			// x3 x6
			// to [x1 x2 x3 x4 x5 x6]
			Eigen::MatrixXcf x_ij = x[i][j];
            Eigen::Map<Eigen::VectorXcf> tmp_v(x_ij.data(), x_ij.cols() * x_ij.rows());
            Eigen::MatrixXcf tmp_vt = tmp_v.transpose();
            x_mat.row(j) = tmp_vt;
		}
		// do projection by matrix production
		Eigen::MatrixXcf res_temp = x_mat.transpose() * project_matrix[i]; // each col is a reduced feature

		// transform back to standard formation
		for (size_t j = 0; j < (size_t)res_temp.cols(); j++) // for each channel of reduced feature
		{
            Eigen::Map<Eigen::MatrixXcf> temp1(res_temp.col(j).data(), x[i][0].rows(), x[i][0].cols());
			temp.push_back(temp1);
		}
		res.push_back(temp);
	}
	return res;
}

EcoFeats projectFeatureMultScale(
    const EcoFeats& x, 
    const std::vector<Eigen::MatrixXcf>& project_matrix) {
    EcoFeats res;
	for (size_t i = 0; i < x.size(); i++) {
		int org_dim = project_matrix[i].rows();
		int numScales = x[i].size() / org_dim;
		std::vector<Eigen::MatrixXcf> temp;

		for (size_t s = 0; s < (size_t)numScales; s++) // for every scale
		{
			Eigen::MatrixXcf x_mat(org_dim, x[i][0].cols() * x[i][0].rows());
			for (size_t j = s * org_dim; j < (s + 1) * org_dim; j++)
			{
				Eigen::MatrixXcf x_ij = x[i][j];
                Eigen::Map<Eigen::VectorXcf> tmp_v(x_ij.data(), x_ij.cols() * x_ij.rows());
                Eigen::MatrixXcf tmp_vt = tmp_v.transpose();
                x_mat.row(j - s * org_dim) = tmp_vt;
			}

            // do projection by matrix production
            Eigen::MatrixXcf res_temp = x_mat.transpose() * project_matrix[i]; // each col is a reduced feature

            // transform back to standard formation
            for (size_t j = 0; j < (size_t)res_temp.cols(); j++) // for each channel of reduced feature
            {
                Eigen::Map<Eigen::MatrixXcf> temp1(res_temp.col(j).data(), x[i][0].rows(), x[i][0].cols());
                temp.push_back(temp1);
            }
		}
		res.push_back(temp);
	}
	return res;
}

std::complex<float> EcoFeatInnerProduct (    
    const EcoFeats& f1,
    const EcoFeats& f2) {

    assert (f1.size() == f2.size());
    std::complex<float> res;
    for (int n = 0; n < static_cast<int>(f2.size()); ++n) {
        for (int m = 0; m < static_cast<int>(f2.at(n).size()); ++m) {
            Eigen::MatrixXcf temp_conj = f2.at(n).at(m).conjugate();
            res += (f1.at(n).at(m).cwiseProduct(temp_conj)).sum();
        }
    }
    return res;
}

// vectorize features
std::vector<Eigen::MatrixXcf> vectorFeature(const EcoFeats &x) {

	if (x.empty())
		return std::vector<Eigen::MatrixXcf>();

	std::vector<Eigen::MatrixXcf> res;
	for (size_t i = 0; i < x.size(); i++) {
		Eigen::MatrixXcf temp(static_cast<int>(x[i].size()), x[i][0].cols() * x[i][0].rows());
		for (size_t j = 0; j < x[i].size(); j++) {
			Eigen::MatrixXcf x_ij = x[i][j];
            Eigen::Map<Eigen::VectorXcf> tmp_v(x_ij.data(), x_ij.cols() * x_ij.rows());
            Eigen::MatrixXcf tmp_vt = tmp_v.transpose();
            temp.row(j) = tmp_vt;
		}
		res.push_back(temp);
	}
	return res;
}

EcoFeats EcoFeatureDotDivide(const EcoFeats &a, const EcoFeats &b) {
	EcoFeats res;
	if (a.size() != b.size())
		assert(0 && "Unmatched feature size!");

	for (size_t i = 0; i < a.size(); i++) {
		std::vector<Eigen::MatrixXcf> temp;
		for (size_t j = 0; j < a[i].size(); j++) {
			temp.push_back(a[i][j].cwiseQuotient(b[i][j]));
		}
		res.push_back(temp);
	}
	return res;
}

std::vector<Eigen::MatrixXcf> getProjectMatEnergy(
    const std::vector<Eigen::MatrixXcf> project_mat,
    const std::vector<int>& feature_dim,
	const std::vector<Eigen::MatrixXcf>& yf) {

	std::vector<Eigen::MatrixXcf> result;
	for (size_t i = 0; i < yf.size(); i++)
	{
        Eigen::MatrixXcf temp = Eigen::MatrixXcf::Zero(project_mat.at(i).rows(), project_mat.at(i).cols());
		float sum_dim = std::accumulate(feature_dim.begin(),
										feature_dim.end(),
										0.0f);
        Eigen::MatrixXcf x = yf[i].cwiseProduct(yf[i]);
		temp = (2.f * x.sum()) / sum_dim *
			   Eigen::MatrixXcf::Ones(project_mat.at(i).rows(), project_mat.at(i).cols());
		result.push_back(temp);
	}
	return result;
}

// Shift a sample in the Fourier domain. The shift should be normalized to the range [-pi, pi]
EcoFeats shiftSample(EcoFeats &xf,
                     float x,
                     float y,
                     std::vector<Eigen::MatrixXcf> kx,
                     std::vector<Eigen::MatrixXcf> ky) {

	EcoFeats res;
    // for each feature
	for (size_t i = 0; i < xf.size(); ++i) {
		Eigen::MatrixXcf shift_exp_y(ky[i].rows(), ky[i].cols());
		Eigen::MatrixXcf shift_exp_x(kx[i].rows(), kx[i].cols());
		int xf_row = xf[i][0].rows();
		int xf_col = xf[i][0].cols();
		for (size_t j = 0; j < (size_t)ky[i].rows(); j++)
		{
            std::complex<float> y_temp(std::cos(y * ky[i](j, 0)).real(), std::sin(y * ky[i](j, 0)).real());
			shift_exp_y(j, 0) = y_temp;
		}
		for (size_t j = 0; j < (size_t)kx[i].cols(); j++)
		{
            std::complex<float> x_temp(std::cos(x * kx[i](0, j)).real(), std::sin(x * kx[i](0, j)).real());
			shift_exp_x(0, j) = x_temp;
		}
		int sy_r = shift_exp_y.rows() - 1;
		int sy_c = shift_exp_y.cols() - 1;
		Eigen::MatrixXcf shift_exp_y_mat = Eigen::MatrixXcf::NullaryExpr(xf[i][0].rows(), xf[i][0].cols(),
                                [&shift_exp_y, &sy_r, &sy_c, xf_row, xf_col] (Eigen::Index i,Eigen::Index j) {
                                return shift_exp_y(std::min<Eigen::Index>(sy_r,i),
                                                   std::min<Eigen::Index>(sy_c,j) ); } );
		int sx_r = shift_exp_x.rows() - 1;
		int sx_c = shift_exp_x.cols() - 1;
		Eigen::MatrixXcf shift_exp_x_mat = Eigen::MatrixXcf::NullaryExpr(xf[i][0].rows(), xf[i][0].cols(),
                                [&shift_exp_x, &sx_r, &sx_c, xf_row, xf_col] (Eigen::Index i,Eigen::Index j) {
                                return shift_exp_x(std::min<Eigen::Index>(sx_r,i),
                                                   std::min<Eigen::Index>(sx_c,j) ); } );

		std::vector<Eigen::MatrixXcf> tmp;
        // for each dimension of the feature, do complex element-wise multiplication
		for (size_t j = 0; j < xf[i].size(); j++) {
			tmp.push_back((shift_exp_y_mat.cwiseProduct(xf[i][j])).cwiseProduct(shift_exp_x_mat));
		}
		res.push_back(tmp);
	}
	return res;
}

} // namespace eco_tracker