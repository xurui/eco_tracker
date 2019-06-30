#ifndef _PARAMETERS_HPP
#define _PARAMETERS_HPP

#include <memory>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#define INF 0x7f800000
// #define USE_CNN

#ifdef USE_CNN
#include <caffe/caffe.hpp>
#include <caffe/util/io.hpp>
#include "net.h"
#endif

using std::string;
using std::vector;

typedef cv::Rect_<float> Rect2f;
typedef std::vector<std::vector<Eigen::MatrixXcf> > EcoFeats;

#ifdef USE_CNN
struct CnnParameters {
	string proto;
	string model;
	string mean_file;

	string ncnn_param;
	string ncnn_bin;

    boost::shared_ptr<ncnn::Net> ncnn_net;
	boost::shared_ptr<caffe::Net<float>> net;
	cv::Mat deep_mean_mat, deep_mean_mean_mat;

	vector<int> stride = {2, 16};			// stride in total
	vector<int> cell_size = {4, 16};		// downsample_factor
	vector<int> output_layer = {3, 14};		// Which layers to use
	vector<int> downsample_factor = {2, 1}; // How much to downsample each output layer
	int input_size_scale = 1;				// Extra scale factor of the input samples to the network (1 is no scaling)
	vector<int> nDim = {96, 512};			// Original dimension of features (ECO Paper Table 1)
	vector<int> compressed_dim = {16, 64};  // Compressed dimensionality of each output layer (ECO Paper Table 1)
	vector<float> penalty = {0, 0};

	vector<int> start_ind = {3, 3, 1, 1};	 // sample feature start index
	vector<int> end_ind = {106, 106, 13, 13}; // sample feature end index
};
struct CnnFeatures {
	CnnParameters fparams;
	cv::Size img_input_sz = cv::Size(224, 224); // VGG default input sample size
	cv::Size img_sample_sz;						// the size of sample
	cv::Size data_sz_block0, data_sz_block1;
	cv::Mat mean;
};
#endif

struct HogParameters {
	int cell_size = 6;
	int compressed_dim = 10;
	int nOrients = 9;
	size_t nDim = 31;
};
struct HogFeatures {
	HogParameters fparams;
	cv::Size img_input_sz;
	cv::Size img_sample_sz;
	cv::Size data_sz_block0;
};

struct CgOpts {
	bool debug;
	bool CG_use_FR;
	float tol;
	bool CG_standard_alpha;
	float init_forget_factor;
	int maxit;
};

struct EcoParameters {
	// Features
	bool useDeepFeature;
	bool useHogFeature;

	HogFeatures hog_features;

	// extra parameters
	CgOpts CG_opts;
	float max_score_threshhold;

	// img sample parameters
	float search_area_scale;
	int min_image_sample_size;
	int max_image_sample_size;

	// Detection parameters
	int newton_iterations;

	// Learning parameters
	float output_sigma_factor;
	float learning_rate;
	size_t nSamples;
	int train_gap;

	// Factorized convolution parameters
	float projection_reg;

	// Conjugate Gradient parameters
	int CG_iter;
	int init_CG_iter;
	int init_GN_iter;
	bool CG_use_FR;
	bool CG_standard_alpha;
	int CG_forgetting_rate;
	float precond_data_param;
	float precond_reg_param;
	int precond_proj_param;

	// Regularization window parameters
	bool use_reg_window;
	double reg_window_min;
	double reg_window_edge;
	size_t reg_window_power;
	float reg_sparsity_threshold;

	float interpolation_bicubic_a;
	size_t number_of_scales;
	float scale_step;
	float min_scale_factor;
	float max_scale_factor;
};
#endif
