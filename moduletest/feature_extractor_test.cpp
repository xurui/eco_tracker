#include "feature_extractor.hpp"
#include "parameters.hpp"

int main() {

    cv::Mat im = cv::imread("/home/ubuntu/workspace/eco_tracker/moduletest/lena.jpg");
    EcoParameters params;
    bool is_color_image = true;

	float search_area = (im.rows * im.cols) * std::pow(params.search_area_scale, 2);
    float currentScaleFactor = 1.f;
	// if (search_area > params.max_image_sample_size)
	// 	currentScaleFactor = sqrt((float)search_area / params.max_image_sample_size);
	// else if (search_area < params.min_image_sample_size)
	// 	currentScaleFactor = sqrt((float)search_area / params.min_image_sample_size);
	// else
	// 	currentScaleFactor = 1.0;
    cv::Point2f pos;
	pos.x = (im.cols - 1.0) / 2.0;
	pos.y = (im.rows - 1.0) / 2.0;
    std::vector<float> scales(1, currentScaleFactor);
    params.hog_features.img_input_sz = cv::Size2i(std::sqrt(search_area), std::sqrt(search_area));
    params.hog_features.img_sample_sz = cv::Size2i(std::sqrt(search_area), std::sqrt(search_area));
    params.hog_features.data_sz_block0 = cv::Size(
			params.hog_features.img_sample_sz.width /
				params.hog_features.fparams.cell_size,
			params.hog_features.img_sample_sz.height /
				params.hog_features.fparams.cell_size);

	eco_tracker::EcoFeats xt = eco_tracker::extractor(im, pos, scales, params, is_color_image);

    std::cout << "----xt size: " << (int)xt.size() << std::endl;
    for (int n = 0; n < (int)xt.size(); ++n) {
        std::cout << "---xt " << n << " size: " << xt.at(n).size() << std::endl;
        for (int m = 0; m < (int)xt.at(n).size(); ++m) {
            std::cout << "---xt m: " << m << " row: " << xt.at(n).at(m).rows() << " col: " << xt.at(n).at(m).cols() << std::endl;
            std::cout << "---xt: " << xt.at(n).at(m) << std::endl;
        }
    }

    return 0;
}
