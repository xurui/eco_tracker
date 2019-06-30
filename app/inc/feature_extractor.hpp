#ifndef _FEATURE_EXTRACTOR_HPP_
#define _FEATURE_EXTRACTOR_HPP_

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <math.h>
#include <vector>
#include <numeric>
#include "fhog.h"
#include "parameters.hpp"
#include "ffttools.hpp"

namespace eco_tracker {

typedef std::vector<std::vector<Eigen::MatrixXcf> > EcoFeats;

std::vector<cv::Mat> getHoGfeature(
	const HogFeatures& hog_features,
	const vector<cv::Mat>& ims);

std::vector<cv::Mat> normalizeHoGfeature(
	const HogFeatures& hog_features,
	std::vector<cv::Mat> &hog_feat_maps);

#ifdef USE_CNN
    EcoFeats getCNNlayersbyNCNN(
		const CnnFeatures& cnn_features,
		const cv::Mat& deep_mean_mat,
		boost::shared_ptr<ncnn::Net> ncnn_net,
		std::vector<cv::Mat> im);
	EcoFeats getCNNlayersbyCaffe(
		const CnnFeatures& cnn_features,
		const cv::Mat& deep_mean_mat,
		boost::shared_ptr<caffe::Net<float>> net,
		std::vector<cv::Mat> im);
	cv::Mat sample_pool(
		const cv::Mat &im,
		int smaple_factor,
		int stride);
	void cnn_feature_normalization(
		const CnnFeatures& cnn_features,
		EcoFeats &feature);
#endif

Eigen::MatrixXcf transferMat2Matrix(cv::Mat mat);

std::vector<Eigen::MatrixXcf> transferMat2Matrix(
		               std::vector<cv::Mat> mat);

template <typename t>
t x2(const cv::Rect_<t> &rect) {
	return rect.x + rect.width;
}

template <typename t>
t y2(const cv::Rect_<t> &rect) {
    return rect.y + rect.height;
}

template <typename t>
void limit(cv::Rect_<t> &rect, cv::Rect_<t> limit) {
	if (rect.x + rect.width > limit.x + limit.width) {
		rect.width = limit.x + limit.width - rect.x;
	} if (rect.y + rect.height > limit.y + limit.height) {
		rect.height = limit.y + limit.height - rect.y;
	} if (rect.x < limit.x) {
		rect.width -= (limit.x - rect.x);
		rect.x = limit.x;
	} if (rect.y < limit.y) {
		rect.height -= (limit.y - rect.y);
		rect.y = limit.y;
	} if (rect.width < 0) {
		rect.width = 0;
	} if (rect.height < 0) {
		rect.height = 0;
	}
}

template <typename t>
void limit(cv::Rect_<t> &rect, t width, t height, t x = 0, t y = 0){
	limit(rect, cv::Rect_<t > (x, y, width, height));
}

template <typename t>
cv::Rect getBorder(const cv::Rect_<t > &original, cv::Rect_<t > & limited) {
	cv::Rect_<t > res;
	res.x = limited.x - original.x;
	res.y = limited.y - original.y;
	res.width = x2(original) - x2(limited);
	res.height = y2(original) - y2(limited);
	assert(res.x >= 0 && res.y >= 0 && res.width >= 0 && res.height >= 0);
	return res;
}

cv::Mat subwindow(const cv::Mat &in, const cv::Rect & window, int borderType = cv::BORDER_CONSTANT);

EcoFeats extractor(const cv::Mat image,
					const cv::Point2f pos,
					const vector<float> scales,
					const EcoParameters &params,
					const bool &is_color_image);

cv::Mat getSamplePatch(const cv::Mat im,
					   const cv::Point2f pos,
					   cv::Size2f sample_sz,
					   cv::Size2f input_sz);
}

#endif
