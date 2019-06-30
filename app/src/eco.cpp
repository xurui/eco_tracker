#include "eco.hpp"

namespace eco_tracker {

void ECOTracker::init(cv::Mat &im, const Rect2f &rect, const nlohmann::json &cfg_param)
{
	// 0. Clean all the parameters.
	pos_.x = 0;
	pos_.y = 0;
	frames_since_last_train_ = 0;
	output_size_ = 0;
	output_index_ = 0;
	base_target_size_.height = 0;
	base_target_size_.width = 0;
	img_sample_size_.height = 0;
	img_sample_size_.width = 0;
	img_support_size_.height = 0;
	img_support_size_.width = 0;
	feature_size_.clear();
	filter_size_.clear();
	feature_dim_.clear();
	compressed_dim_.clear();
	currentScaleFactor_ = 0;
	nScales_ = 0;
	ky_.clear();
	kx_.clear();
	yf_.clear();
	cos_window_.clear();
	interp1_fs_.clear();
	interp2_fs_.clear();
	reg_filter_.clear();
	projection_matrix_.clear();
	reg_energy_.clear();
	scale_factors_.clear();
	sample_energy_.clear();
    hf_.clear();
	hf_full_.clear();

	// 1. Initialize all the parameters.
	// Image infomations
	if (im.channels() == 3) {
		is_color_image_ = true;
	} else {
		is_color_image_ = false;
	}

    initialization(im, rect, cfg_param);

	EcoFeats xl, xlf, xlf_porj;
	// 2. Extract features from the first frame.
	xl = extractor(im, pos_, vector<float>(1, currentScaleFactor_), params_, is_color_image_);

	// 3. Multiply the features by the cosine window.
    EcoFeats xlw;
	for (size_t i = 0; i < xl.size(); i++) {
		std::vector<Eigen::MatrixXcf> temp;
		for (size_t j = 0; j < xl[i].size(); j++)
			temp.push_back(cos_window_[i].cwiseProduct(xl[i][j]));
		xlw.push_back(temp);
	}

	// 4. Do DFT on the features.
	xlf = runFFt(xlw);

	// 5. Interpolate features to the continuous domain.
	xlf = initInterpolateFFt(xlf, interp1_fs_, interp2_fs_);
	xlf = compactFourierCoeff(xlf); // take half of the cols

	// 6. Initialize projection matrix P.
	initProjectionMatrix(xlw, compressed_dim_, projection_matrix_);

	// 7. Do the feature reduction for each feature.
	xlf_porj = projectFeature(xlf, projection_matrix_);

	// 8. Initialize and update sample space.
	sample_update_.init(filter_size_, compressed_dim_, params_.nSamples, params_.learning_rate);
	sample_update_.updateSampleSpaceModel(xlf_porj);

	// 9. Calculate sample energy and projection map energy.
	sample_energy_ = computeFeautrePower(xlf_porj);
	std::vector<Eigen::MatrixXcf> proj_energy = getProjectMatEnergy(projection_matrix_, feature_dim_, yf_);

	// 10. Initialize filter and it's derivative.
	for (size_t i = 0; i < xlf_porj.size(); i++) {
		hf_.push_back(std::vector<Eigen::MatrixXcf>(xlf_porj[i].size(), Eigen::MatrixXcf::Zero(xlf_porj[i][0].rows(), xlf_porj[i][0].cols())));
	}
	// 11. Train the tracker(train the filter and update the projection matrix).
	// 12. Update projection matrix P.
    trainFilterJoint(xlf, sample_energy_, reg_filter_, reg_energy_, proj_energy, yf_, params_, hf_, projection_matrix_);

	// 13. Re-project the sample and update the sample space.
    xlf_porj = projectFeature(xlf, projection_matrix_);
    sample_update_.replaceSample(xlf_porj, 0);

	// 14. Update distance matrix of sample space. Find the norm of the reprojected sample
	float new_sample_norm = (EcoFeatInnerProduct(xlf_porj, xlf_porj)).real();
	sample_update_.setGramMatrix(0, 0, 2 * new_sample_norm);

	// 15. Update filter f.
	hf_full_ = fullFourierCoeff(hf_);

}

bool ECOTracker::update(const cv::Mat &frame, Rect2f &roi) {
	cv::Point sample_pos;
	sample_pos.x = static_cast<int>(pos_.x);
	sample_pos.y = static_cast<int>(pos_.y);
	std::vector<float> samples_scales;
	for (size_t i = 0; i < scale_factors_.size(); ++i) {
		samples_scales.push_back(currentScaleFactor_ * scale_factors_[i]);
	}

	// 1: Extract features at multiple resolutions
	EcoFeats xt = extractor(frame, sample_pos, samples_scales, params_, is_color_image_);
	if (xt[0].size() == 0) {
		return false;
	}

	// 2:  project sample
	EcoFeats xt_proj = projectFeatureMultScale(xt, projection_matrix_);
	// 3: Do windowing of features
    EcoFeats xtw_proj;
	for (size_t i = 0; i < xt_proj.size(); i++) {
		std::vector<Eigen::MatrixXcf> temp;
		for (size_t j = 0; j < xt_proj[i].size(); j++)
			temp.push_back(cos_window_[i].cwiseProduct(xt_proj[i][j]));
		xtw_proj.push_back(temp);
	}

	// 4: Compute the fourier series
	EcoFeats xtf_proj = runFFt(xtw_proj);

	// 5: Interpolate features to the continuous domain
	xtf_proj = initInterpolateFFt(xtf_proj, interp1_fs_, interp2_fs_);

	// 6: Compute the scores in Fourier domain for different scales of target
	std::vector<Eigen::MatrixXcf> scores_fs_sum;
	for (size_t i = 0; i < scale_factors_.size(); i++)
		scores_fs_sum.push_back(Eigen::MatrixXcf::Zero(filter_size_[output_index_].height, filter_size_[output_index_].width));
	for (size_t i = 0; i < xtf_proj.size(); i++) {
		int pad = (filter_size_[output_index_].height - xtf_proj[i][0].rows()) / 2;
        int rows = xtf_proj[i][0].rows();
        int cols = xtf_proj[i][0].cols();
		for (size_t j = 0; j < xtf_proj[i].size(); j++)	{
			size_t k1 = j / hf_full_[i].size(); // for each scale
			size_t k2 = j % hf_full_[i].size(); // for each dimension of scale
			scores_fs_sum[k1].block(pad, pad, rows, cols) += xtf_proj[i][j].cwiseProduct(hf_full_[i][k2]);
		}
	}

	// 7: Calculate score by inverse DFT and
	// 8: Optimize the continuous score function with Newton's method.
    int track_score_scale_ind = 0;
    float track_score_pos_x = 0.f;
    float track_score_pos_y = 0.f;
    float track_score_value = 0.f;
    computeScores(scores_fs_sum, params_.newton_iterations, track_score_scale_ind,
        track_score_value, track_score_pos_x, track_score_pos_y);

	// Compute the translation vector in pixel-coordinates and round to the closest integer pixel.
	int scale_change_factor = track_score_scale_ind;
	float resize_scores_width = (img_support_size_.width / output_size_) * currentScaleFactor_ * scale_factors_[scale_change_factor];
	float resize_scores_height = (img_support_size_.height / output_size_) * currentScaleFactor_ * scale_factors_[scale_change_factor];
	float dx = track_score_pos_x * resize_scores_width;
	float dy = track_score_pos_y * resize_scores_height;
	//debug("scale_change_factor:%d, get_disp_col: %f, get_disp_row: %f, dx: %f, dy: %f", scale_change_factor, scores.get_disp_col(), scores.get_disp_row(), dx, dy);

	// 9: Update position
	pos_ = cv::Point2f(sample_pos.x + dx, sample_pos.y + dy);

	// Update the scale
	currentScaleFactor_ = currentScaleFactor_ * scale_factors_[scale_change_factor];

	// Adjust the scale to make sure we are not too large or too small
	if (currentScaleFactor_ < params_.min_scale_factor) {
		currentScaleFactor_ = params_.min_scale_factor;
	} else if (currentScaleFactor_ > params_.max_scale_factor) {
		currentScaleFactor_ = params_.max_scale_factor;
	}
	// 1: Get the sample calculated in localization
	EcoFeats xlf_proj;
	for (size_t i = 0; i < xtf_proj.size(); ++i) {
		std::vector<Eigen::MatrixXcf> tmp;
		int start_ind = scale_change_factor * projection_matrix_[i].cols();
		int end_ind = (scale_change_factor + 1) * projection_matrix_[i].cols();
		for (size_t j = start_ind; j < (size_t)end_ind; ++j) {
			tmp.push_back(xtf_proj[i][j].leftCols(xtf_proj[i][j].rows() / 2 + 1));
		}
		xlf_proj.push_back(tmp);
	}

	// 2: Shift the sample so that the target is centered,
	//  A shift in spatial domain means multiply by exp(i pi L k), according to shift property of Fourier transformation.
	cv::Point2f shift_samp =
		2.0f * CV_PI * cv::Point2f(pos_.x - sample_pos.x, pos_.y - sample_pos.y) *
		(1.0f / (currentScaleFactor_ * img_support_size_.width));
	xlf_proj = shiftSample(xlf_proj, shift_samp.x, shift_samp.y, kx_, ky_);

	// 3: Update the samples space to include the new sample, the distance matrix,
	// kernel matrix and prior weight are also updated
	sample_update_.updateSampleSpaceModel(xlf_proj);

	// 4: Train the tracker every Nsth frame, Ns in ECO paper
	bool train_tracker = frames_since_last_train_ >= (size_t)params_.train_gap;

	// Set conjugate gradient uptions
	params_.CG_opts.CG_use_FR = params_.CG_use_FR;
	params_.CG_opts.tol = 1e-6;
	params_.CG_opts.CG_standard_alpha = params_.CG_standard_alpha;
	params_.CG_opts.maxit = params_.CG_iter;

	if (train_tracker) {
		EcoFeats temp1 = sample_energy_ * (1 - params_.learning_rate);
		EcoFeats temp2 = computeFeautrePower(xlf_proj) * params_.learning_rate;
		sample_energy_ = temp1 + temp2;
        trainFilter(sample_update_.getSamples(), reg_filter_, sample_update_.getWeights(),
            sample_energy_, reg_energy_, yf_, params_, hf_);

		frames_since_last_train_ = 0;
	} else {
		++frames_since_last_train_;
	}
	// 5: Update projection matrix P.
	// projection_matrix_ = eco_trainer_.get_proj();

	// 6: Update filter f.
	hf_full_ = fullFourierCoeff(hf_);

	roi.width = base_target_size_.width * currentScaleFactor_;
	roi.height = base_target_size_.height * currentScaleFactor_;
	roi.x = pos_.x - roi.width / 2;
	roi.y = pos_.y - roi.height / 2;

	if (track_score_value >= params_.max_score_threshhold) {
		return true;
	} else {
		return false;
	}
}

void ECOTracker::initialization(
	const cv::Mat& im,
	const Rect2f& rect,
	const nlohmann::json &cfg_param) {
	// Get the ini position
	pos_.x = rect.x + (rect.width - 1.0) / 2.0;
	pos_.y = rect.y + (rect.height - 1.0) / 2.0;

	// Read in all the parameters
	readCfgParameters(cfg_param);
	printf("max_score_threshhold: %f\n", params_.max_score_threshhold);

	// Calculate search area and initial scale factor
	float search_area = rect.area() * std::pow(params_.search_area_scale, 2);
	if (search_area > params_.max_image_sample_size)
		currentScaleFactor_ = sqrt((float)search_area / params_.max_image_sample_size);
	else if (search_area < params_.min_image_sample_size)
		currentScaleFactor_ = sqrt((float)search_area / params_.min_image_sample_size);
	else
		currentScaleFactor_ = 1.0;

	// target size at the initial scale
	base_target_size_ = cv::Size2f(rect.size().width / currentScaleFactor_, rect.size().height / currentScaleFactor_);

	// window size, taking padding into account
    float img_sample_size__tmp;
    img_sample_size__tmp = std::sqrt(base_target_size_.area() * std::pow(params_.search_area_scale, 2));
    img_sample_size_ = cv::Size2i(img_sample_size__tmp, img_sample_size__tmp);

	initFeatureParameters();

    initCosWindow();

    initYf();

    initInterpFourier();

    initRegFilter();

    initScaleFactor();

	if (nScales_ > 0) {
		params_.min_scale_factor = std::pow(params_.scale_step, std::ceil(
			std::log(std::fmax(5 / (float)img_support_size_.width,
							   5 / (float)img_support_size_.height)) / std::log(params_.scale_step)));
		params_.max_scale_factor = std::pow(params_.scale_step, std::floor(
			std::log(std::fmin(im.cols / (float)base_target_size_.width,
							   im.rows / (float)base_target_size_.height)) / std::log(params_.scale_step)));
	}
	// Set conjugate gradient uptions
	params_.CG_opts.CG_use_FR = true;
	params_.CG_opts.tol = 1e-6;
	params_.CG_opts.CG_standard_alpha = true;
	if (params_.CG_forgetting_rate == INF || params_.learning_rate >= 1) {
		params_.CG_opts.init_forget_factor = 0;
	} else {
		params_.CG_opts.init_forget_factor = std::pow(1.0f - params_.learning_rate, params_.CG_forgetting_rate);
	}
	params_.CG_opts.maxit = std::ceil(params_.init_CG_iter / params_.init_GN_iter);
}

void ECOTracker::initCosWindow() {
	output_index_ = 0;
	output_size_ = 0;
	for (size_t i = 0; i != feature_size_.size(); ++i) {
		size_t size = feature_size_[i].width + (feature_size_[i].width + 1) % 2;
		filter_size_.push_back(cv::Size(size, size));
		output_index_ = size > output_size_ ? i : output_index_;
		output_size_ = std::max(size, output_size_);
		Eigen::MatrixXcf temp_cos_win;
		genCosWindow(feature_size_[i].height, feature_size_[i].width, temp_cos_win);
        cos_window_.push_back(temp_cos_win);
	}
}

void ECOTracker::initYf() {
	for (size_t i = 0; i < filter_size_.size(); ++i) {
		Eigen::MatrixXcf tempy(filter_size_[i].height, 1);
		Eigen::MatrixXcf tempx(1, filter_size_[i].height / 2 + 1);
		for (int j = 0; j < tempy.rows(); j++) {
			tempy(j, 0) = j - (tempy.rows() / 2);
		}
		ky_.push_back(tempy);
		for (int j = 0; j < tempx.cols(); j++) {
			tempx(0, j) = j - (filter_size_[i].height / 2);
		}
		kx_.push_back(tempx);
	}
	float sigma_y = sqrt(int(base_target_size_.width) *
						int(base_target_size_.height)) *
				   (params_.output_sigma_factor) *
				   (float(output_size_) / img_support_size_.width);
    
    genGaussianYf(sigma_y, output_size_, kx_, ky_, yf_);
}

void ECOTracker::initInterpFourier() {
	for (size_t i = 0; i < filter_size_.size(); ++i) {

		Eigen::MatrixXcf interp1_fs1, interp2_fs1;
		getInterpFourier(filter_size_[i].width,
                         filter_size_[i].height,
						 interp1_fs1,
						 interp2_fs1,
						 params_.interpolation_bicubic_a);
		interp1_fs_.push_back(interp1_fs1);
		interp2_fs_.push_back(interp2_fs1);
	}
}

void ECOTracker::initRegFilter() {
	for (size_t i = 0; i < filter_size_.size(); i++) {
		Eigen::MatrixXcf temp_d = reg_filter::get_regularization_filter(img_support_size_.width,
		                                           img_support_size_.height,
												   base_target_size_.width,
												   base_target_size_.height,
												   params_);
		reg_filter_.push_back(temp_d);
		Eigen::MatrixXcf temp_d_d = temp_d.cwiseProduct(temp_d);
		float energy = temp_d_d.sum().real();
		// this energy is used for preconditioner
		reg_energy_.push_back(energy);
	}
}


void ECOTracker::initScaleFactor() {
	nScales_ = params_.number_of_scales;
	scale_step_ = params_.scale_step;
	if (nScales_ % 2 == 0) {
		nScales_++;
	}
	int scalemin = std::floor((1.0 - (float)nScales_) / 2.0);
	int scalemax = std::floor(((float)nScales_ - 1.0) / 2.0);
	for (int i = scalemin; i <= scalemax; i++) {
		scale_factors_.push_back(std::pow(scale_step_, i));
	}
}

void ECOTracker::readCfgParameters(const nlohmann::json &cfg_param) {
	cfg_param.at("useHogFeature").get_to(params_.useHogFeature);
	// params_.useCnFeature = parameters.useCnFeature;

#ifdef USE_CNN
    cfg_param.at("CNNParam").at("proto").get_to(params_.cnn_features.fparams.proto);
	cfg_param.at("CNNParam").at("model").get_to(params_.cnn_features.fparams.model);
	cfg_param.at("CNNParam").at("mean_file").get_to(params_.cnn_features.fparams.mean_file);
	cfg_param.at("CNNParam").at("ncnn_param").get_to(params_.cnn_features.fparams.ncnn_param);
	cfg_param.at("CNNParam").at("ncnn_bin").get_to(params_.cnn_features.fparams.ncnn_bin);	
#endif

	cfg_param.at("HogParam").at("cell_size").get_to(params_.hog_features.fparams.cell_size);
	// params_.cn_features.fparams.tablename = parameters.cn_features.fparams.tablename;

	// Extra parameters
    cfg_param.at("max_score_threshhold").get_to(params_.max_score_threshhold);

	// img sample parameters
    cfg_param.at("search_area_scale").get_to(params_.search_area_scale);
	cfg_param.at("min_image_sample_size").get_to(params_.min_image_sample_size);
    cfg_param.at("max_image_sample_size").get_to(params_.max_image_sample_size);

	// Detection parameters
	cfg_param.at("newton_iterations").get_to(params_.newton_iterations);

	// Learning parameters
	cfg_param.at("output_sigma_factor").get_to(params_.output_sigma_factor);
	cfg_param.at("learning_rate").get_to(params_.learning_rate);
	cfg_param.at("nSamples").get_to(params_.nSamples);
	cfg_param.at("train_gap").get_to(params_.train_gap);

	// Factorized convolution parameters
	cfg_param.at("projection_reg").get_to(params_.projection_reg);

	// Conjugate Gradient parameters
	cfg_param.at("CG_iter").get_to(params_.CG_iter);
	cfg_param.at("init_CG_iter").get_to(params_.init_CG_iter);
	cfg_param.at("init_GN_iter").get_to(params_.init_GN_iter);
	cfg_param.at("CG_use_FR").get_to(params_.CG_use_FR);
	cfg_param.at("CG_standard_alpha").get_to(params_.CG_standard_alpha);
	cfg_param.at("CG_forgetting_rate").get_to(params_.CG_forgetting_rate);
	cfg_param.at("precond_data_param").get_to(params_.precond_data_param);
	cfg_param.at("precond_reg_param").get_to(params_.precond_reg_param);
	cfg_param.at("precond_proj_param").get_to(params_.precond_proj_param);


	// Regularization window parameters
	cfg_param.at("use_reg_window").get_to(params_.use_reg_window);
	cfg_param.at("reg_window_min").get_to(params_.reg_window_min);
	cfg_param.at("reg_window_edge").get_to(params_.reg_window_edge);
	cfg_param.at("reg_window_power").get_to(params_.reg_window_power);
	cfg_param.at("reg_sparsity_threshold").get_to(params_.reg_sparsity_threshold);

	// Interpolation parameters
    cfg_param.at("interpolation_bicubic_a").get_to(params_.interpolation_bicubic_a);
	// Scale parameters for the translation model
	cfg_param.at("number_of_scales").get_to(params_.number_of_scales);
	cfg_param.at("scale_step").get_to(params_.scale_step);
}

void ECOTracker::initFeatureParameters() {

#ifdef USE_CNN
	// Init features parameters---------------------------------------
	if (params_.useDeepFeature) {
		printf("Setting up Caffe in CPU mode\n");
		caffe::Caffe::set_mode(caffe::Caffe::CPU);

		params_.cnn_features.fparams.net.reset(new caffe::Net<float>(params_.cnn_features.fparams.proto, caffe::TEST));
		params_.cnn_features.fparams.net->CopyTrainedLayersFrom(params_.cnn_features.fparams.model);

		params_.cnn_features.fparams.ncnn_net.reset(new ncnn::Net());
		params_.cnn_features.fparams.ncnn_net->load_param(params_.cnn_features.fparams.ncnn_param.c_str());
		params_.cnn_features.fparams.ncnn_net->load_model(params_.cnn_features.fparams.ncnn_bin.c_str());
		read_deep_mean(params_.cnn_features.fparams.mean_file,
		               params_.cnn_features.fparams.deep_mean_mat,
					   params_.cnn_features.fparams.deep_mean_mean_mat);

		params_.cnn_features.img_input_sz = img_sample_size_;
		params_.cnn_features.img_sample_sz = img_sample_size_;

		// Calculate the output size of the 2 output layer;
		// matlab version pad can be unbalanced, but caffe cannot for the moment;
		int cnn_output_sz0 = (int)((img_sample_size_.width - 7 + 0 + 0) / 2) + 1; //122
		int cnn_output_sz1 = (int)((cnn_output_sz0 - 3 + 0 + 0) / 2) + 1; //61
		cnn_output_sz1 = (int)((cnn_output_sz1 - 3 + 1 + 1) / 2) + 1;	 //15
		cnn_output_sz1 = (int)((cnn_output_sz1 - 3 + 0 + 0) / 2) + 1; //15
		int total_feature_sz0 = cnn_output_sz0;
		int total_feature_sz1 = cnn_output_sz1;
		// Re-calculate the output size of the 1st output layer;
		int support_sz = params_.cnn_features.fparams.stride[1] * cnn_output_sz1;	// 16 x 15 = 240
		cnn_output_sz0 = (int)(support_sz / params_.cnn_features.fparams.stride[0]); // 240 / 2 = 120

		int start_ind0 = (int)((total_feature_sz0 - cnn_output_sz0) / 2) + 1; // 2
		int start_ind1 = (int)((total_feature_sz1 - cnn_output_sz1) / 2) + 1; // 1
		int end_ind0 = start_ind0 + cnn_output_sz0 - 1;						  // 121
		int end_ind1 = start_ind1 + cnn_output_sz1 - 1;						  // 15

		params_.cnn_features.fparams.start_ind =
			{start_ind0, start_ind0, start_ind1, start_ind1};
		params_.cnn_features.fparams.end_ind =
			{end_ind0, end_ind0, end_ind1, end_ind1};
		params_.cnn_features.data_sz_block0 =
			cv::Size(cnn_output_sz0 / params_.cnn_features.fparams.downsample_factor[0],
					 cnn_output_sz0 / params_.cnn_features.fparams.downsample_factor[0]);
		params_.cnn_features.data_sz_block1 =
			cv::Size(cnn_output_sz1 / params_.cnn_features.fparams.downsample_factor[1],
					 cnn_output_sz1 / params_.cnn_features.fparams.downsample_factor[1]);
		params_.cnn_features.mean = params_.cnn_features.fparams.deep_mean_mean_mat;
		img_support_size_ = cv::Size(support_sz, support_sz);
	} else {
		params_.cnn_features.fparams.net =
			boost::shared_ptr<caffe::Net<float>>();
	}
#else
	params_.useDeepFeature = false;
#endif

	if (!params_.useDeepFeature) {
		img_support_size_ = img_sample_size_;
	}
	if (params_.useHogFeature) {
		params_.hog_features.img_input_sz = img_support_size_;
		params_.hog_features.img_sample_sz = img_support_size_;
		params_.hog_features.data_sz_block0 = cv::Size(
			params_.hog_features.img_sample_sz.width /
				params_.hog_features.fparams.cell_size,
			params_.hog_features.img_sample_sz.height /
				params_.hog_features.fparams.cell_size);
	}
	// if (params_.useCnFeature && is_color_image_)
	// {
	// 	params_.cn_features.img_input_sz = img_support_size_;
	// 	params_.cn_features.img_sample_sz = img_support_size_;
	// 	params_.cn_features.data_sz_block0 = cv::Size(
	// 		params_.cn_features.img_sample_sz.width /
	// 			params_.cn_features.fparams.cell_size,
	// 		params_.cn_features.img_sample_sz.height /
	// 			params_.cn_features.fparams.cell_size);

	// 	std::string s;
	// 	std::string path = params_.cn_features.fparams.tablename;
	// 	ifstream *read = new ifstream(path);
	// 	size_t rows = sizeof(params_.cn_features.fparams.table) / sizeof(params_.cn_features.fparams.table[0]);
	// 	size_t cols = sizeof(params_.cn_features.fparams.table[0]) / sizeof(float);
	// 	//debug("rows:%lu,cols:%lu", rows, cols);
	// 	for (size_t i = 0; i < rows; i++)
	// 	{
	// 		for (size_t j = 0; j < cols - 1; j++)
	// 		{
	// 			getline(*read, s, '\t');
	// 			params_.cn_features.fparams.table[i][j] = atof(s.c_str());
	// 		}
	// 		getline(*read, s);
	// 		params_.cn_features.fparams.table[i][cols - 1] = atof(s.c_str());
	// 	}
	// }

	// features setting-----------------------------------------------------
#ifdef USE_CNN
	if (params_.useDeepFeature) {
		feature_size_.push_back(params_.cnn_features.data_sz_block0);
		feature_size_.push_back(params_.cnn_features.data_sz_block1);
		feature_dim_ = params_.cnn_features.fparams.nDim;
		compressed_dim_ = params_.cnn_features.fparams.compressed_dim;
	}
#endif
	if (params_.useHogFeature) {
		feature_size_.push_back(params_.hog_features.data_sz_block0);
		feature_dim_.push_back(params_.hog_features.fparams.nDim);
		compressed_dim_.push_back(params_.hog_features.fparams.compressed_dim);
	}
	// if (params_.useCnFeature && is_color_image_)
	// {
	// 	feature_size_.push_back(params_.cn_features.data_sz_block0);
	// 	feature_dim_.push_back(params_.cn_features.fparams.nDim);
	// 	compressed_dim_.push_back(params_.cn_features.fparams.compressed_dim);
	// }
}

} // namespace eco_tracker