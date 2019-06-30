#ifndef _SAMPLE_SPACE_HPP
#define _SAMPLE_SPACE_HPP

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <math.h>
#include <vector>
#include <numeric>

#include "ffttools.hpp"
#include "eco_util.hpp"

#define MAX_DIST 1e10

namespace eco_tracker {

typedef std::vector<std::vector<Eigen::MatrixXcf> > EcoFeats;

// using SampleMat = Eigen::Matrix<std::complex<float>, SAMPLE_NUM, SAMPLE_NUM>;
using SampleMat = Eigen::MatrixXcf;

class SampleSpace {
  public:
    SampleSpace(){};

    void init(
      const std::vector<cv::Size> &filter,
      const std::vector<int> &feature_dim,
      int sample_num, float learning_rate);

    ~SampleSpace(){};

    void updateSampleSpaceModel(const EcoFeats& new_sample);

    void updateSampleMatrix(
      Eigen::VectorXcf& gram_vector,
      float new_sample_norm,
      int id1, int id2, float w1, float w2);


    inline void replaceSample(const EcoFeats &new_sample, const size_t idx) {
      target_samples_[idx] = new_sample;
    }

    inline void setGramMatrix(const int r, const int c, const float val) {
      gram_mat_(r, c) = val;
    }

    std::vector<float> getWeights() const { return init_weights_; }

    std::vector<EcoFeats> getSamples() const { return target_samples_; }
  private:

    inline EcoFeats mergeSamples(const EcoFeats &sample1,
                  const EcoFeats &sample2,
                  const float w1, const float w2) {

      float alpha1 = w1 / (w1 + w2);
      float alpha2 = 1 - alpha1;
      EcoFeats merged_sample = sample1;

          for (size_t i = 0; i < sample1.size(); i++)
              for (size_t j = 0; j < sample1[i].size(); j++)
                  merged_sample[i][j] = alpha1 * sample1[i][j] + alpha2 * sample2[i][j];
      return merged_sample;
    }

    // find the minimum element in prior_weights_;
    inline void findMin(float &min_w, size_t &index) const
    {
      std::vector<float>::const_iterator pos = std::min_element(init_weights_.begin(), init_weights_.end());
      min_w = *pos;
      index = pos - init_weights_.begin();
    }

    int sample_num_;
    float learning_rate_;
    int new_sample_id_;
    int merged_sample_id_;
    int training_sample_num_;

    std::vector<EcoFeats> target_samples_;
    SampleMat distance_mat_;
    SampleMat gram_mat_;
    std::vector<float> init_weights_;
};
} // namespace eco_tracker

#endif