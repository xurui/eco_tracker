#include "sample_space.hpp"

namespace eco_tracker {

void SampleSpace::init(
	const std::vector<cv::Size> &filter,
	const std::vector<int> &feature_dim,
	int sample_num, float learning_rate) {

	sample_num_ = sample_num;
	learning_rate_ = learning_rate;
	new_sample_id_ = -1;
	merged_sample_id_ = -1;
	training_sample_num_ = 0;
	target_samples_.clear();
	distance_mat_ = Eigen::MatrixXcf::Zero(sample_num, sample_num);
	gram_mat_ = Eigen::MatrixXcf::Zero(sample_num, sample_num);
	init_weights_.clear();
	init_weights_.resize(sample_num);

	for (size_t n = 0; n < sample_num; n++){
		EcoFeats temp;
		for (size_t j = 0; j < (size_t)feature_dim.size(); j++) {
			std::vector<Eigen::MatrixXcf> temp_single_feat;
			for (size_t i = 0; i < (size_t)feature_dim[j]; i++) {
				Eigen::MatrixXcf temp_single_dim(filter[j].height, (filter[j].width + 1) / 2);
                temp_single_feat.push_back(temp_single_dim);
			}
			temp.push_back(temp_single_feat);
		}
		target_samples_.push_back(temp);
	}
}

void SampleSpace::updateSampleSpaceModel(const EcoFeats& new_sample) {

    Eigen::VectorXcf gram_vec(sample_num_, 1);
    for (int n = 0; n < sample_num_; ++n) {
        // because of compact fourier coeff process
        // 2(ac + bd)
        gram_vec[n] = 2.f * EcoFeatInnerProduct(new_sample, target_samples_.at(n));
    }
    // c^2 + d^2
    float new_sample_norm = (2.f * EcoFeatInnerProduct(new_sample, new_sample)).real();
    Eigen::VectorXcf distance(sample_num_, 1);
    for (int n = 0; n < sample_num_; ++n) {
		// a^2 + b^2 + c^2 + d^2 - 2(ac+bd)
		std::complex<float> temp = new_sample_norm + gram_mat_(n,n) - 2.f * gram_vec[n];
		if (n < training_sample_num_) {
            distance[n] = (temp.real() > 0) ? temp : 0;
        } else {
            distance[n] = MAX_DIST;
        }
    }

    if (training_sample_num_ == sample_num_) {

		float min_sample_weight = MAX_DIST;
		size_t min_sample_id = 0;
		findMin(min_sample_weight, min_sample_id);

        const float minmum_sample_weight = 0.0036;

		if (min_sample_weight < minmum_sample_weight) {
			// If any init weight is less than the minimum allowed weight,
		    // replace that sample with the new sample
			updateSampleMatrix(gram_vec, new_sample_norm, min_sample_id, -1, 0, 1);
			init_weights_[min_sample_id] = 0;
			// normalize the init_weights_
			float sum = std::accumulate(init_weights_.begin(), init_weights_.end(), 0.0f);
			for (size_t i = 0; i < (size_t)sample_num_; i++) {
				init_weights_[i] = init_weights_[i] *
									(1 - learning_rate_) / sum;
			}
			// set the new sample's weight as learning_rate_
			init_weights_[min_sample_id] = learning_rate_;

			// update sampel space.
			merged_sample_id_ = -1;
			new_sample_id_ = min_sample_id;
            target_samples_[new_sample_id_] = new_sample;
		} else {
            // If no sample has low enough prior weight, then we either merge
			// the new sample with an existing sample, or merge two of the
			// existing samples and insert the new sample in the vacated position
			// Find the minimum distance between new sample and exsiting samples.
            int new_min_r, new_min_c;
			Eigen::VectorXf temp_real = distance.real();
			float new_sample_min_dist = temp_real.minCoeff(&new_min_r, &new_min_c);

			// Find the closest pair amongst existing samples.
            int exist_min_r, exist_min_c;
			Eigen::MatrixXf mat_real = distance_mat_.real();
			float existing_samples_min_dist = mat_real.minCoeff(&exist_min_r, &exist_min_c);

			if (exist_min_r == exist_min_c)
				assert(0 && "error: distance matrix diagonal filled wrongly.");

			if (new_sample_min_dist < existing_samples_min_dist) {
				// If the min distance of the new sample to the existing samples is less than the min distance
				// amongst any of the existing samples, we merge the new sample with the nearest existing

				// renormalize prior weights
				for (size_t i = 0; i < init_weights_[i]; i++)
					init_weights_[i] *= (1 - learning_rate_);

				// Set the position of the merged sample
				merged_sample_id_ = new_min_r;

				// Extract the existing sample to merge
				EcoFeats existing_sample_to_merge = target_samples_[merged_sample_id_];

				// Merge the new_sample with existing sample
				EcoFeats merged_sample =
					mergeSamples(existing_sample_to_merge,
								  new_sample,
								  init_weights_[merged_sample_id_], learning_rate_);

				// Update distance matrix and the gram matrix
				updateSampleMatrix(gram_vec, new_sample_norm, merged_sample_id_, -1, init_weights_[merged_sample_id_], learning_rate_);

				// Update the prior weight of the merged sample
				init_weights_[new_min_r] += learning_rate_;

				// update the merged sample and discard new sample
                target_samples_[merged_sample_id_] = merged_sample;
			} else {
				// we merge the nearest existing samples and insert the new sample in the vacated position

				// renormalize prior weights
				for (size_t i = 0; i < init_weights_[i]; i++)
					init_weights_[i] *= (1 - learning_rate_);

				// Ensure that the sample with higher prior weight is assigned id1.
				if (init_weights_[exist_min_c] >
					init_weights_[exist_min_r])
					std::swap(exist_min_c,
							  exist_min_r);

				// Merge the existing closest samples
				EcoFeats merged_sample =
					mergeSamples(target_samples_[exist_min_c],
								  target_samples_[exist_min_r],
								  init_weights_[exist_min_c],
                                  init_weights_[exist_min_r]);

				// Update distance matrix and the gram matrix
				updateSampleMatrix(gram_vec, new_sample_norm, exist_min_c, exist_min_r, init_weights_[exist_min_c], init_weights_[exist_min_r]);

				// Update prior weights for the merged sample and the new sample
				init_weights_[exist_min_c] +=
					init_weights_[exist_min_r];
				init_weights_[exist_min_r] = learning_rate_;

				// Update the merged sample and insert new sample
				merged_sample_id_ = exist_min_c;
				new_sample_id_ = exist_min_r;
                target_samples_[merged_sample_id_] = merged_sample;
                target_samples_[new_sample_id_] = new_sample;
			}
		}
	} else {
		size_t sample_position = training_sample_num_;
		updateSampleMatrix(gram_vec, new_sample_norm, sample_position, -1, 0, 1);
		if (sample_position == 0) {
			init_weights_[sample_position] = 1;
		} else {
			for (size_t i = 0; i < sample_position; i++)
				init_weights_[i] *= (1 - learning_rate_);
			init_weights_[sample_position] = learning_rate_;
		}
		// update sample space
		new_sample_id_ = sample_position;
        target_samples_[new_sample_id_] = new_sample; 
		training_sample_num_++;
	}
}

void SampleSpace::updateSampleMatrix(
    Eigen::VectorXcf& gram_vector,
    float new_sample_norm,
    int id1, int id2, float w1, float w2) {

	float alpha1 = w1 / (w1 + w2);
	float alpha2 = 1 - alpha1;

	if (id2 < 0) {
		std::complex<float> norm_id1 = gram_mat_(id1, id1);
        gram_mat_.col(id1) = alpha1 * gram_mat_.col(id1) + alpha2 * gram_vector;
		Eigen::MatrixXcf row_temp = (gram_mat_.col(id1)).transpose();
        gram_mat_.row(id1) = row_temp;
        gram_mat_(id1, id1) = std::pow(alpha1, 2) * norm_id1.real() +
            std::pow(alpha2, 2) * new_sample_norm + 2 * alpha1 * alpha2 * gram_vector[id1].real();

		// Update distance matrix
		Eigen::VectorXcf distance(sample_num_, 1);
		for (size_t i = 0; i < sample_num_; i++) {
			std::complex<float> temp = gram_mat_(id1, id1) +
						 gram_mat_(i, i) - 2.f * gram_mat_(i, id1);
			distance[i] = (temp.real() > 0) ? temp : 0;
		}
        distance_mat_.col(id1) = distance;
		Eigen::MatrixXcf dist_trans_temp = distance.transpose();
        distance_mat_.row(id1) = dist_trans_temp;
        distance_mat_(id1, id1) = std::complex<float>(MAX_DIST, 0);
	} else {
		// Two existing samples are merged and the new sample fills the empty
		std::complex<float> norm_id1 = gram_mat_(id1, id1);
		std::complex<float> norm_id2 = gram_mat_(id2, id2);
		std::complex<float> ip_id1_id2 = gram_mat_(id1, id2);
		// Handle the merge of existing samples
		Eigen::VectorXcf vec_temp = alpha1 * gram_mat_.col(id1) +
					alpha2 * gram_mat_.col(id2);
        gram_mat_.col(id1) = vec_temp;
		Eigen::MatrixXcf gram_id1_trans = vec_temp.transpose();
        gram_mat_.row(id1) = gram_id1_trans;

		gram_mat_(id1, id1) = std::pow(alpha1, 2) * norm_id1.real() +
			std::pow(alpha2, 2) * norm_id2.real() + 2 * alpha1 * alpha2 * ip_id1_id2.real();
		gram_vector(id1) = alpha1 * gram_vector[id1] +
			alpha2 * gram_vector[id2];

		// Handle the new sample
        gram_mat_.col(id2) = gram_vector;
		Eigen::MatrixXcf gram_id2_trans = gram_vector.transpose();
        gram_mat_.row(id2) = gram_id2_trans;
		gram_mat_(id2, id2) = new_sample_norm;

		// Update the distance matrix
		Eigen::VectorXcf distance(sample_num_, 1);
		std::vector<int> id({id1, id2});
		for (size_t i = 0; i < 2; i++) {
			for (size_t j = 0; j < sample_num_; j++) {
				std::complex<float> temp = gram_mat_(id[i], id[i]) +
					gram_mat_(j, j) - 2.f * gram_mat_(j, id[i]);
                distance[j] = (temp.real() > 0) ? temp : 0;
			}
            distance_mat_.col(id[i]) = distance;
			Eigen::MatrixXcf dist_trans = distance.transpose();
            distance_mat_.row(id[i]) = dist_trans;
			distance_mat_(id[i], id[i]) = std::complex<float>(MAX_DIST, 0);
		}
	}
}
} // namespace eco_tracker