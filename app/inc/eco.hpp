#ifndef _ECO_HPP_
#define _ECO_HPP_

#include <iostream>
#include <string>
#include <math.h>
#include <algorithm>
#include <nlohmann/json.hpp>
#include "feature_extractor.hpp"
#include "parameters.hpp"
#include "interpolator.hpp"
#include "get_reg_filter.hpp"
#include "sample_space.hpp"
#include "train_filter.hpp"
#include "optimize_scores.hpp"
#include "eco_util.hpp"

namespace eco_tracker {

class ECOTracker {
  public:
	ECOTracker() {};
	virtual ~ECOTracker() {}

	void initialization(
		const cv::Mat& im,
		const Rect2f& rect,
		const nlohmann::json &cfg_param);

	void init(cv::Mat &im, const Rect2f &rect, const nlohmann::json &cfg_param); 

	bool update(const cv::Mat &frame, Rect2f &roi);
	
	void readCfgParameters(const nlohmann::json &cfg_param);

	void initFeatureParameters();

	void initCosWindow();

  void initYf();

  void initInterpFourier();

  void initRegFilter();

  void initScaleFactor();

  private:
	bool				                    is_color_image_;
	EcoParameters 		              params_;
	cv::Point2f 		                pos_;
	size_t 				                  frames_since_last_train_;
	size_t 				                  output_size_, output_index_; 	

  // target size without scale
	cv::Size2f 			                base_target_size_;
	// base_target_sz * sarch_area_scale
	cv::Size2i			                img_sample_size_;
	// the corresponding size in the image
	cv::Size2i			                img_support_size_;

	std::vector<cv::Size> 	        feature_size_, filter_size_;
	std::vector<int> 		            feature_dim_, compressed_dim_;

	int 				                    nScales_;
	float 				                  scale_step_;
	std::vector<float>		          scale_factors_;
	float 				                  currentScaleFactor_;

	std::vector<Eigen::MatrixXcf> 	ky_, kx_, yf_; 
	std::vector<Eigen::MatrixXcf> 	interp1_fs_, interp2_fs_; 
	std::vector<Eigen::MatrixXcf> 	cos_window_;
	std::vector<Eigen::MatrixXcf> 	projection_matrix_;

	std::vector<Eigen::MatrixXcf> 	reg_filter_;
	std::vector<float> 		          reg_energy_;

	SampleSpace 		                sample_update_;
	EcoFeats 			                  sample_energy_;

  EcoFeats 			                  hf_;
	EcoFeats 			                  hf_full_;
};

} // namespace eco_tracker
#endif