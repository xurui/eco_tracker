#include "feature_extractor.hpp"

namespace eco_tracker {

EcoFeats extractor(const cv::Mat image,
				   const cv::Point2f pos,
				   const std::vector<float> scales,
				   const EcoParameters &params,
				   const bool &is_color_image) {
	int num_features = 0, num_scales = scales.size();
	std::vector<cv::Size2f> img_sample_sz;
	std::vector<cv::Size2f> img_input_sz;

#ifdef USE_CNN
	boost::shared_ptr<caffe::Net<float>> net;
	boost::shared_ptr<ncnn::Net> ncnn_net;
	CnnFeatures cnn_features;
	int cnn_feat_ind = -1;
	EcoFeats cnn_feat_maps;

	cv::Mat new_deep_mean_mat;
	if (params.useDeepFeature) {
		cnn_feat_ind = num_features;
		num_features++;
		cnn_features = params.cnn_features;
		net = params.cnn_features.fparams.net;
		ncnn_net = params.cnn_features.fparams.ncnn_net;
		resize(params.cnn_features.fparams.deep_mean_mat, new_deep_mean_mat,
			   params.cnn_features.img_input_sz, 0, 0, cv::INTER_CUBIC);
		img_sample_sz.push_back(cnn_features.img_sample_sz);
		img_input_sz.push_back(cnn_features.img_input_sz);
	}
#endif

	HogFeatures hog_features;
	int hog_feat_ind = -1;
	std::vector<cv::Mat> hog_feat_maps;

	if (params.useHogFeature) {
		hog_feat_ind = num_features;
		num_features++;
		hog_features = params.hog_features;
		img_sample_sz.push_back(hog_features.img_sample_sz);
		img_input_sz.push_back(hog_features.img_input_sz);
	}
	// if (params.useCnFeature && is_color_image)
	// {
	// 	cn_feat_ind_ = num_features;
	// 	num_features++;
	// 	cn_features_ = params.cn_features;
	// 	img_sample_sz.push_back(cn_features_.img_sample_sz);
	// 	img_input_sz.push_back(cn_features_.img_input_sz);
	// }
	// Extract images
	std::vector<std::vector<cv::Mat> > img_samples;
	for (int i = 0; i < num_features; ++i) {
		std::vector<cv::Mat> img_samples_temp(num_scales);
		for (unsigned int j = 0; j < scales.size(); ++j) {
			img_samples_temp[j] = getSamplePatch(image, pos, img_sample_sz[i] * scales[j], img_input_sz[i]);
		}
		img_samples.push_back(img_samples_temp);
	}

    // Extract features
	EcoFeats sum_features;
#ifdef USE_CNN
	if (params.useDeepFeature) {
		cnn_feat_maps = getCNNlayersbyCaffe(cnn_features,
		                               new_deep_mean_mat,
									   net,
		                               img_samples[cnn_feat_ind]);
		cnn_feature_normalization(cnn_features, cnn_feat_maps);
		sum_features = cnn_feat_maps;
	}
#endif

	if (params.useHogFeature) {
		hog_feat_maps = getHoGfeature(hog_features, img_samples[hog_feat_ind]);
		hog_feat_maps = normalizeHoGfeature(hog_features, hog_feat_maps);
        std::vector<Eigen::MatrixXcf> feat_maps = transferMat2Matrix(hog_feat_maps);
		sum_features.push_back(feat_maps);
	}
	// if (params.useCnFeature && is_color_image) {
	// 	cn_feat_maps_ = get_cn_features(img_samples[cn_feat_ind_]);
	// 	cn_feat_maps_ = cn_feature_normalization(cn_feat_maps_);
	// 	sum_features.push_back(cn_feat_maps_);
	// }
	return sum_features;
}

std::vector<Eigen::MatrixXcf> transferMat2Matrix(
	std::vector<cv::Mat> mat) {

    std::vector<Eigen::MatrixXcf> res;
    for (size_t i = 0; i < mat.size(); ++i) {
		Eigen::MatrixXcf temp(mat[i].rows, mat[i].cols);
		for (int r = 0; r < mat[i].rows; ++r) {
			for (int c = 0; c < mat[i].cols; ++c) {
				temp(r, c) = mat[i].at<float>(r,c);
			}
		}
		res.push_back(temp);
	}
	return res;
}

Eigen::MatrixXcf transferMat2Matrix(cv::Mat mat) {
	Eigen::MatrixXcf res(mat.rows, mat.cols);
	for (int r = 0; r < mat.rows; ++r) {
		for (int c = 0; c < mat.cols; ++c) {
            res(r, c) = mat.at<float>(r,c);
		}
	}
	return res;
}

cv::Mat getSamplePatch(const cv::Mat im,
					 const cv::Point2f posf,
					 cv::Size2f sample_sz,
					 cv::Size2f input_sz) {
	// Pos should be integer when input, but floor in just in case.
    cv::Point2i pos(posf.x, posf.y);

	// Downsample factor
	float resize_factor = std::min(sample_sz.width / input_sz.width,
								   sample_sz.height / input_sz.height);
	int df = std::max((float)std::floor(resize_factor - 0.1), float(1));

	cv::Mat new_im;
	im.copyTo(new_im);
	if (df > 1) {
		// compute offset and new center position
		cv::Point os((pos.x - 1) % df, ((pos.y - 1) % df));
		pos.x = (pos.x - os.x - 1) / df + 1;
		pos.y = (pos.y - os.y - 1) / df + 1;
		// new sample size
		sample_sz.width = sample_sz.width / df;
		sample_sz.height = sample_sz.height / df;
		// down sample image
		int r = (im.rows - os.y) / df + 1;
		int c = (im.cols - os.x) / df;
		cv::Mat new_im2(r, c, im.type());
		new_im = new_im2;
		for (size_t i = 0 + os.y, m = 0;
			 i < (size_t)im.rows && m < (size_t)new_im.rows;
			 i += df, ++m) {
			for (size_t j = 0 + os.x, n = 0;
				 j < (size_t)im.cols && n < (size_t)new_im.cols;
				 j += df, ++n) {
				if (im.channels() == 1) {
					new_im.at<uchar>(m, n) = im.at<uchar>(i, j);
				} else {
					new_im.at<cv::Vec3b>(m, n) = im.at<cv::Vec3b>(i, j);
				}
			}
		}
	}

	// make sure the size is not too small and round it
	sample_sz.width = std::max(std::round(sample_sz.width), 2.f);
	sample_sz.height = std::max(std::round(sample_sz.height), 2.f);

	cv::Point pos2(pos.x - std::floor((sample_sz.width + 1) / 2),
				   pos.y - std::floor((sample_sz.height + 1) / 2));
	cv::Mat im_patch = subwindow(new_im, cv::Rect(pos2, sample_sz), IPL_BORDER_REPLICATE);

	cv::Mat resized_patch;
	if (im_patch.cols == 0 || im_patch.rows == 0) {
		return resized_patch;
	}
	cv::resize(im_patch, resized_patch, input_sz);

	return resized_patch;
}

std::vector<cv::Mat> getHoGfeature(
	const HogFeatures& hog_features,
	const vector<cv::Mat>& ims) {
	if (ims.empty()) {
		return vector<cv::Mat>();
	}

	vector<cv::Mat> hog_feats;
	for (unsigned int k = 0; k < ims.size(); k++) {
		int h, w, d, binSize, nOrients, softBin, nDim, hb, wb, useHog;
		bool full = 1;
		useHog = 2;
		h = ims[k].rows;
		w = ims[k].cols;
		d = ims[k].channels();
		binSize = hog_features.fparams.cell_size;
		nOrients = hog_features.fparams.nOrients;
		softBin = -1;
		nDim = useHog == 0 ? nOrients : (useHog == 1 ? nOrients * 4 : nOrients * 3 + 5);
		hb = h / binSize;
		wb = w / binSize;
		float clipHog = 0.2f;
		float *I, *M, *O, *H;
		I = (float *)wrCalloc(h * w * d, sizeof(float));
		M = (float *)wrCalloc(h * w, sizeof(float));
		O = (float *)wrCalloc(h * w, sizeof(float));
		H = (float *)wrCalloc(hb * wb * nDim, sizeof(float));
		
		for (int i = 0; i < h; i++)
			for (int j = 0; j < w; j++) {
				*(I + j * h + i) = (float)ims[k].at<cv::Vec3b>(i, j)[2];
				*(I + h * w + j * h + i) = (float)ims[k].at<cv::Vec3b>(i, j)[1];
				*(I + 2 * h * w + j * h + i) = (float)ims[k].at<cv::Vec3b>(i, j)[0];
			}
		gradMag(I, M, O, h, w, d, 1);
		{
			fhog(M, O, H, h, w, binSize, nOrients, softBin, clipHog);
		}
		cv::Mat featuresMap = cv::Mat(cv::Size(wb, hb), CV_32FC(nDim - 1));
		for (int i = 0; i < hb; i++)
			for (int j = 0; j < wb; j++)
				for (int l = 0; l < nDim - 1; l++) {
					featuresMap.at<cv::Vec<float, 31>>(i, j)[l] = *(H + l * hb * wb + j * hb + i);
				}
		hog_feats.push_back(featuresMap);
		wrFree(I);
		wrFree(M);
		wrFree(O);
		wrFree(H);
	}
	return hog_feats;
}
std::vector<cv::Mat> normalizeHoGfeature(
    const HogFeatures& hog_features,
    std::vector<cv::Mat> &hog_feat_maps) {
	if (hog_feat_maps.empty()) {
		return std::vector<cv::Mat>();
	}
	std::vector<cv::Mat> hog_maps_vec;
	for (size_t i = 0; i < hog_feat_maps.size(); i++) {
		if (hog_feat_maps[i].cols == 0 || hog_feat_maps[i].rows == 0) {
			std::vector<cv::Mat> emptyMat;
			hog_maps_vec.insert(hog_maps_vec.end(), emptyMat.begin(), emptyMat.end());
		} else {
			cv::Mat temp = hog_feat_maps[i].mul(hog_feat_maps[i]);
			std::vector<cv::Mat> temp_vec, result_vec;
			float sum = 0;
			cv::split(temp, temp_vec);
			for (int j = 0; j < temp.channels(); j++)
			{
				sum += cv::sum(temp_vec[j])[0];
			}
			float para = hog_features.data_sz_block0.area() * hog_features.fparams.nDim;
			hog_feat_maps[i] *= sqrt(para / sum);
			cv::split(hog_feat_maps[i], result_vec);
			hog_maps_vec.insert(hog_maps_vec.end(), result_vec.begin(), result_vec.end());
		}
	}
	return hog_maps_vec;
}

cv::Mat subwindow(const cv::Mat &in, const cv::Rect & window, int borderType) {
	cv::Rect cutWindow = window;
	limit(cutWindow, in.cols, in.rows);

	if (cutWindow.height <= 0 || cutWindow.width <= 0) assert(0); 

	cv::Rect border = getBorder(window, cutWindow);
	cv::Mat res = in(cutWindow);

	if (border != cv::Rect(0, 0, 0, 0)) {
		cv::copyMakeBorder(res, res, border.y, border.height, border.x, border.width, borderType);
	}
	return res;
}

#ifdef USE_CNN
EcoFeats getCNNlayersbyNCNN(
		const CnnFeatures& cnn_features,
		const cv::Mat& deep_mean_mat,
		boost::shared_ptr<ncnn::Net> ncnn_net,
		std::vector<cv::Mat> im) {
	// Preprocess the images
	std::vector<ncnn::Mat> in_vec;
	in_vec.reserve(im.size()); 
	for (unsigned int i = 0; i < im.size(); ++i) {
		im[i].convertTo(im[i], CV_32FC3);
		im[i] = im[i] - deep_mean_mat;
		cv::Mat temp_hwc2chw(cv::Size(im[i].cols, im[i].rows*im[i].channels()), CV_32FC1);
		for (int j = 0; j < im[i].channels(); ++j) {
            cv::extractChannel(im[i], cv::Mat(im[i].rows, im[i].cols, CV_32FC1,
			&(temp_hwc2chw.at<float>(im[i].rows*im[i].cols*j))), j);
		}
		ncnn::Mat in(im[i].rows, im[i].cols, im[i].channels(), (float*)temp_hwc2chw.data);
		in_vec.emplace_back(in.clone());
	}
	EcoFeats feature_map;
	for (size_t idx = 0; idx < cnn_features.fparams.output_layer.size(); ++idx) {
		vector<cv::Mat> merge_feature;
		if (cnn_features.fparams.output_layer[idx] == 3) {
            for (unsigned int i = 0; i < in_vec.size(); ++i) {
                ncnn::Extractor ex = ncnn_net->create_extractor();
				ex.input("data", in_vec[i]);
				ncnn::Mat out;
				ex.extract("norm1", out);
				float* pstart = (float*)out.data;
				for (unsigned int c = 0; c < out.c; ++c) {
                    cv::Mat feat_map(out.h, out.w, CV_32FC1, pstart);
					pstart += out.h * out.w;
					cv::Mat extract_map = feat_map(cv::Range(cnn_features.fparams.start_ind[0 + 2 * idx] - 1, cnn_features.fparams.end_ind[0 + 2 * idx]),
												cv::Range(cnn_features.fparams.start_ind[0 + 2 * idx] - 1, cnn_features.fparams.end_ind[0 + 2 * idx]));
					extract_map = (cnn_features.fparams.downsample_factor[idx] == 1) ? extract_map : sample_pool(extract_map, 2, 2);
					merge_feature.push_back(extract_map);
				}
			}
		} else if (cnn_features.fparams.output_layer[idx] == 14) {
            for (unsigned int i = 0; i < in_vec.size(); ++i) {
                ncnn::Extractor ex = ncnn_net->create_extractor();
				ex.input("data", in_vec[i]);
				ncnn::Mat out;
				ex.extract("conv5", out);
				float* pstart = (float*)out.data;
				for (unsigned int c = 0; c < out.c; ++c) {
                    cv::Mat feat_map(out.h, out.w, CV_32FC1, pstart);
					pstart += out.h * out.w;
					cv::Mat extract_map = feat_map(cv::Range(cnn_features.fparams.start_ind[0 + 2 * idx] - 1, cnn_features.fparams.end_ind[0 + 2 * idx]),
												cv::Range(cnn_features.fparams.start_ind[0 + 2 * idx] - 1, cnn_features.fparams.end_ind[0 + 2 * idx]));
					extract_map = (cnn_features.fparams.downsample_factor[idx] == 1) ? extract_map : sample_pool(extract_map, 2, 2);
					merge_feature.push_back(extract_map);
				}
			}
		}
        std::vector<Eigen::MatrixXcf> feat_maps = transferMat2Matrix(merge_feature);
		feature_map.push_back(feat_maps);
	}
}

EcoFeats getCNNlayersbyCaffe(
		const CnnFeatures& cnn_features,
		const cv::Mat& deep_mean_mat,
		boost::shared_ptr<caffe::Net<float>> net,
		std::vector<cv::Mat> im) {
	caffe::Blob<float> *input_layer = net->input_blobs()[0];
	int width = input_layer->width();
	int height = input_layer->height();
	float *input_data = input_layer->mutable_cpu_data();
	input_layer->Reshape(im.size(), im[0].channels(), im[0].rows, im[0].cols);
	net->Reshape();

	// Preprocess the images
	for (unsigned int i = 0; i < im.size(); ++i) {
		im[i].convertTo(im[i], CV_32FC3);
		im[i] = im[i] - deep_mean_mat;
	}
	// Put the images to the input_data.
	std::vector<cv::Mat> input_channels;
	for (int i = 0; i < input_layer->channels() * input_layer->shape()[0]; ++i) {
		cv::Mat channel(height, width, CV_32FC1, input_data);
		input_channels.push_back(channel);
		input_data += width * height;
	}
	// Split each image and merge all together, then split to input_channels.
	std::vector<cv::Mat> im_split;
	for (unsigned int i = 0; i < im.size(); i++) {
		std::vector<cv::Mat> tmp_split;
		cv::split(im[i], tmp_split);
		im_split.insert(im_split.end(), tmp_split.begin(), tmp_split.end());
	}
	cv::Mat im_merge;
	cv::merge(im_split, im_merge);
	cv::split(im_merge, input_channels);

	net->Forward();

	EcoFeats feature_map;
	for (size_t idx = 0; idx < cnn_features.fparams.output_layer.size(); ++idx) {
		const float *pstart = NULL;
		vector<int> shape;
		if (cnn_features.fparams.output_layer[idx] == 3) {
			boost::shared_ptr<caffe::Blob<float>> layerData = net->blob_by_name("norm1");
			pstart = layerData->cpu_data();
			shape = layerData->shape();
		} else if (cnn_features.fparams.output_layer[idx] == 14) {
			boost::shared_ptr<caffe::Blob<float>> layerData = net->blob_by_name("conv5");
			pstart = layerData->cpu_data();
			shape = layerData->shape();
		}

		vector<cv::Mat> merge_feature;
		for (size_t i = 0; i < (size_t)(shape[0] * shape[1]); i++) {
			cv::Mat feat_map(shape[2], shape[3], CV_32FC1, (void *)pstart);
			pstart += shape[2] * shape[3];
			//  extract features according to fparams
			CnnParameters fparams = cnn_features.fparams;
			cv::Mat extract_map = feat_map(cv::Range(fparams.start_ind[0 + 2 * idx] - 1, fparams.end_ind[0 + 2 * idx]),
										   cv::Range(fparams.start_ind[0 + 2 * idx] - 1, fparams.end_ind[0 + 2 * idx]));
			extract_map = (cnn_features.fparams.downsample_factor[idx] == 1) ? extract_map : sample_pool(extract_map, 2, 2);
			merge_feature.push_back(extract_map);
		}
        std::vector<Eigen::MatrixXcf> feat_maps = transferMat2Matrix(merge_feature);
		feature_map.push_back(feat_maps);
	}

	return feature_map;
}

cv::Mat sample_pool(const cv::Mat &im, int smaple_factor, int stride) {
	if (im.empty())
		return cv::Mat();
	cv::Mat new_im(im.cols / 2, im.cols / 2, CV_32FC1);
	for (size_t i = 0; i < (size_t)new_im.rows; i++) {
		for (size_t j = 0; j < (size_t)new_im.cols; j++)
			new_im.at<float>(i, j) = 0.25 * (im.at<float>(2 * i, 2 * j) + im.at<float>(2 * i, 2 * j + 1) +
											 im.at<float>(2 * i + 1, 2 * j) + im.at<float>(2 * i + 1, 2 * j + 1));
	}
	return new_im;
}

void cnn_feature_normalization(
	const CnnFeatures& cnn_features,
	EcoFeats &cnn_feat_maps) {
	for (size_t i = 0; i < cnn_feat_maps.size(); i++) {
		vector<cv::Mat> temp = cnn_feat_maps[i];
		vector<float> sum_scales;
		for (size_t s = 0; s < temp.size(); s += cnn_features.fparams.nDim[i]) {
			float sum = 0.0f;
			for (size_t j = s; j < s + cnn_features.fparams.nDim[i]; j++)
				sum += cv::sum(temp[j].mul(temp[j]))[0];
			sum_scales.push_back(sum);
		}

		float para = 0.0f;
		if (i == 0)
			para = cnn_features.data_sz_block0.area() * cnn_features.fparams.nDim[i];
		else if (i == 1)
			para = cnn_features.data_sz_block1.area() * cnn_features.fparams.nDim[i];
		for (unsigned int k = 0; k < temp.size(); k++)
			cnn_feat_maps[i][k] /= sqrt(sum_scales[k / cnn_features.fparams.nDim[i]] / para);
	}
}
void read_deep_mean(
	const string &mean_file,
	cv::Mat& deep_mean_mat,
	cv::Mat& deep_mean_mean_mat) {
	caffe::BlobProto blob_proto;
	ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

	// Convert from BlobProto to Blob<float>
	int num_channels_ = 3;
	caffe::Blob<float> mean_blob;
	mean_blob.FromProto(blob_proto);

	//* The format of the mean file is planar 32-bit float BGR or grayscale.
	std::vector<cv::Mat> channels;
	float *data = mean_blob.mutable_cpu_data();
	for (int i = 0; i < num_channels_; ++i) {
		// Extract an individual channel.
		cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
		channels.push_back(channel);
		data += mean_blob.height() * mean_blob.width();
	}

	// Merge the separate channels into a single image.
	cv::merge(channels, deep_mean_mat);

	// Get the mean for each channel.
	deep_mean_mean_mat =
		cv::Mat(cv::Size(224, 224),
				deep_mean_mat.type(),
				cv::mean(deep_mean_mat));
}
#endif
}
