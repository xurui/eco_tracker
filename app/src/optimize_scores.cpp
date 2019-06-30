#include"optimize_scores.hpp"

namespace eco_tracker {

  void computeScores(
      const std::vector<Eigen::MatrixXcf>& scores_fs,
      const int& max_iteration,
	  int& scale_ind,
      float& opt_score,
      float& opt_pos_x,
      float& opt_pos_y) {

	std::vector<Eigen::MatrixXcf> sampled_scores;
	// Do inverse fft to the scores in the Fourier domain back to spacial domain
    // for each scale
	for (size_t i = 0; i < scores_fs.size(); ++i) {
		int area = scores_fs[i].rows() * scores_fs[i].cols();
        // inverse dft
		Eigen::MatrixXcf tmp = fft_tools::ifft(fft_tools::fftshift(scores_fs[i], true));
        // spacial domain only contains real part
		sampled_scores.push_back(tmp * area);
	}

	// to store the position of maximum value of response
	std::vector<int> row, col;
    // inialized max score
	std::vector<float> 	init_max_score;
    // for each scale
	for (size_t i = 0; i < scores_fs.size(); ++i) {
        int pos_y, pos_x;
		Eigen::MatrixXf temp_real = sampled_scores[i].real();
		float max_value = temp_real.maxCoeff(&pos_y, &pos_x);
		row.push_back(pos_y);
		col.push_back(pos_x);
		init_max_score.push_back(max_value);
	}

	// Shift and rescale the coordinate system to [-pi, pi]
	int h = scores_fs[0].rows(), w = scores_fs[0].cols();
	std::vector<float> max_pos_y, max_pos_x, init_pos_y, init_pos_x;
	for (size_t i = 0; i < row.size(); ++i) {
		max_pos_y.push_back( (row[i] + (h - 1) / 2) %  h - (h - 1) / 2);
		max_pos_y[i] *= 2 * M_PI / h;
		max_pos_x.push_back( (col[i] + (w - 1) / 2) %  w - (w - 1) / 2); 
		max_pos_x[i] *= 2 * M_PI / w;
	}
	init_pos_y = max_pos_y; init_pos_x = max_pos_x;
	// Construct grid
    Eigen::MatrixXf ky(1, h);
    Eigen::MatrixXf ky2(1, h);
    Eigen::MatrixXf kx(w, 1);
    Eigen::MatrixXf kx2(w, 1);

	for (int i = 0; i < h; ++i) {
		ky(0, i) = i - (h - 1) / 2;
		ky2(0, i) = ky(0, i) * ky(0, i);
	}
	for (int i = 0; i < w; ++i) {
		kx(i, 0) = i - (w - 1) / 2;
		kx2(i, 0) = kx(i, 0) * kx(i, 0);
	}
	// Pre-compute complex exponential 
	std::vector<Eigen::MatrixXcf> exp_iky, exp_ikx;
	for (unsigned int i = 0; i < scores_fs.size(); ++i)
	{
		Eigen::MatrixXcf tempy(1, h);
		Eigen::MatrixXcf tempx(w, 1);
		for (int y = 0; y < h; ++y) {
            std::complex<float> temp(cos(ky(0, y) * max_pos_y[i]), sin(ky(0, y) * max_pos_y[i]));
            tempy(0, y) = temp;
        }
		for (int x = 0; x < w; ++x) {
            std::complex<float> temp(cos(kx(x, 0) * max_pos_x[i]), sin(kx(x, 0) * max_pos_x[i]));
            tempx(x, 0) = temp;            
        }
		exp_iky.push_back(tempy);
		exp_ikx.push_back(tempx);
	}

	for (int ite = 0; ite < max_iteration; ++ite)
	{
		// Compute gradient
		std::vector<Eigen::MatrixXcf> ky_exp_ky, kx_exp_kx, y_resp, resp_x, grad_y, grad_x;
		std::vector<Eigen::MatrixXcf> ival, H_yy, H_xx, H_xy, det_H;
		for (unsigned int i = 0; i < scores_fs.size(); i++)
		{
			ky_exp_ky.push_back(ky.cwiseProduct(exp_iky[i]));
			kx_exp_kx.push_back(kx.cwiseProduct(exp_ikx[i]));
		
			y_resp.push_back(exp_iky[i] * scores_fs[i]);
			resp_x.push_back(scores_fs[i] * exp_ikx[i]);

			grad_y.push_back(-1 * ky_exp_ky[i] * resp_x[i]);
			grad_x.push_back(-1 * y_resp[i] * kx_exp_kx[i]);

			// Compute Hessian
            std::complex<float> i_mul(0.f, 1.f);
			ival.push_back(i_mul * (exp_iky[i] * resp_x[i]));

			H_yy.push_back(-1 * (ky2.cwiseProduct(exp_iky[i])) * resp_x[i] + ival[i]);
			H_xx.push_back(-1 * y_resp[i] * (kx2.cwiseProduct(exp_ikx[i])) + ival[i]);
			H_xy.push_back(-1 * ky_exp_ky[i] * (scores_fs[i] * kx_exp_kx[i]));

			det_H.push_back(H_yy[i] * H_xx[i] - H_xy[i] * H_xy[i]);
		
			// Compute new position using newtons method
			Eigen::MatrixXcf tmp1 = (H_xx[i] * grad_y[i] - H_xy[i] * grad_x[i]) / det_H[i](0, 0);
            Eigen::MatrixXcf tmp2 = (H_yy[i] * grad_x[i] - H_xy[i] * grad_y[i]) / det_H[i](0, 0);

			max_pos_y[i] -= tmp1(0, 0).real();
			max_pos_x[i] -= tmp2(0, 0).real();

			// Evaluate maximum
            Eigen::MatrixXcf tempy(1, h);
            Eigen::MatrixXcf tempx(w, 1);
            for (int y = 0; y < h; ++y) {
                std::complex<float> temp(cos(ky(0, y) * max_pos_y[i]), sin(ky(0, y) * max_pos_y[i]));
                tempy(0, y) = temp;
            }
            for (int x = 0; x < w; ++x) {
                std::complex<float> temp(cos(kx(x, 0) * max_pos_x[i]), sin(kx(x, 0) * max_pos_x[i]));
                tempx(x, 0) = temp;            
            }
            exp_iky.push_back(tempy);
            exp_ikx.push_back(tempx);
		}
	}
	// Evaluate the Fourier series at the estimated locations to find the corresponding scores.
	std::vector<float> tmp_max_score;
	for (size_t i = 0; i < sampled_scores.size(); ++i)
	{
		// TODO:::::examine this result!!!!
		Eigen::MatrixXcf temp = exp_iky[i] * scores_fs[i] * exp_ikx[i];
		float new_scores = temp(0, 0).real();
		// check for scales that have not increased in score
		if (new_scores > init_max_score[i])
		{
			tmp_max_score.push_back(new_scores);
		}
		else
		{
			tmp_max_score.push_back(init_max_score[i]);
			max_pos_y[i] = init_pos_y[i];
			max_pos_x[i] = init_pos_x[i];
		}
			
	}
	 
	// Find the scale with the maximum response 
	std::vector<float>::iterator pos = max_element(tmp_max_score.begin(), tmp_max_score.end());
	scale_ind = pos - tmp_max_score.begin();
	opt_score = *pos;

	// Scale the coordinate system to output_sz
	opt_pos_y = (fmod(max_pos_y[scale_ind] + M_PI, M_PI * 2.0) - M_PI) / (M_PI * 2.0) * h;
	opt_pos_x = (fmod(max_pos_x[scale_ind] + M_PI, M_PI * 2.0) - M_PI) / (M_PI * 2.0) * w;
	//return sampled_scores;
}
}