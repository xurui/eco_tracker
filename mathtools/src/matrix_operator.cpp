#include "matrix_operator.hpp"

EcoFeats operator+(const EcoFeats &feats1, const EcoFeats &feats2) {
	EcoFeats result;
	if (feats1.size() != feats2.size()) {
		printf("two feature size are not equal!\n");
		assert(false);
	}

    // for each feature
	for (size_t i = 0; i < feats1.size(); i++) {
		std::vector<Eigen::MatrixXcf> feat_vec;
        // for each dimension
		for (size_t j = 0; j < (size_t)feats1[i].size(); j++) {
            Eigen::MatrixXcf temp = feats1[i][j] + feats2[i][j];
			feat_vec.push_back(temp);
		}
		result.push_back(feat_vec);
	}
	return result;
}

EcoFeats operator-(const EcoFeats &feats1, const EcoFeats &feats2) {
	EcoFeats result;
	if (feats1.size() != feats2.size()) {
		printf("two feature size are not equal!\n");
		assert(false);
	}

    // for each feature
	for (size_t i = 0; i < feats1.size(); i++) {
		std::vector<Eigen::MatrixXcf> feat_vec;
        // for each dimension
		for (size_t j = 0; j < (size_t)feats1[i].size(); j++) {
            Eigen::MatrixXcf temp = feats1[i][j] - feats2[i][j];
			feat_vec.push_back(temp);
		}
		result.push_back(feat_vec);
	}
	return result;
}

EcoFeats operator*(const EcoFeats &feats, const float& coef) {
	EcoFeats result;
	if (feats.empty()) {
		return feats;
	}

    // for each feature
	for (size_t i = 0; i < feats.size(); i++) {
		std::vector<Eigen::MatrixXcf> feat_vec;
        // for each dimension
		for (size_t j = 0; j < (size_t)feats[i].size(); j++) {
            Eigen::MatrixXcf temp = feats[i][j] * coef;
			feat_vec.push_back(temp);
		}
		result.push_back(feat_vec);
	}
	return result;
}

std::vector<Eigen::MatrixXcf> operator+(
    const std::vector<Eigen::MatrixXcf>& data1,
    const std::vector<Eigen::MatrixXcf>& data2) {

    std::vector<Eigen::MatrixXcf> result;
	if (data1.size() != data2.size()) {
		printf("two data size are not equal!\n");
		assert(false);
	}

    for (size_t i = 0; i < data1.size(); i++) {
        Eigen::MatrixXcf temp = data1[i] + data2[i];
        result.push_back(temp);
    }
	return result;
}

std::vector<Eigen::MatrixXcf> operator-(
    const std::vector<Eigen::MatrixXcf>& data1,
    const std::vector<Eigen::MatrixXcf>& data2) {

    std::vector<Eigen::MatrixXcf> result;
	if (data1.size() != data2.size()) {
		printf("two data size are not equal!\n");
		assert(false);
	}

    for (size_t i = 0; i < data1.size(); i++) {
        Eigen::MatrixXcf temp = data1[i] - data2[i];
        result.push_back(temp);
    }
	return result;
}

std::vector<Eigen::MatrixXcf> operator*(
    const std::vector<Eigen::MatrixXcf>& data,
    const float& coef) {

	std::vector<Eigen::MatrixXcf> result;
	if (data.empty()) {
        printf("data is empty!\n");
		return data;
	}

    // for each feature
	for (size_t i = 0; i < data.size(); i++) {
        Eigen::MatrixXcf temp = data[i] * coef;
        result.push_back(temp);
	}
	return result;
}

ECO_Train operator+(const ECO_Train& data1, const ECO_Train& data2) {
    ECO_Train result;
	if (data1.part1.size() != data2.part1.size() ||
        data1.part2.size() != data2.part2.size()) {
		printf("two ECO_Train size are not equal!\n");
		assert(false);
	}
    result.part1 = data1.part1 + data2.part1;
    result.part2 = data1.part2 + data2.part2;

    return result;
}

ECO_Train operator-(const ECO_Train& data1, const ECO_Train& data2) {
    ECO_Train result;
	if (data1.part1.size() != data2.part1.size() ||
        data1.part2.size() != data2.part2.size()) {
		printf("two ECO_Train size are not equal!\n");
		assert(false);
	}
    result.part1 = data1.part1 - data2.part1;
    result.part2 = data1.part2 - data2.part2;

    return result;
}

ECO_Train operator*(const ECO_Train& data, const float& scale) {
    ECO_Train result;
	if (data.part1.empty() ||
        data.part2.empty()) {
        printf("data is empty!\n");
		return data;
	}
    result.part1 = data.part1 * scale;
    result.part2 = data.part2 * scale;

    return result;
}