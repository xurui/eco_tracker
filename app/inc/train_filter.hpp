#ifndef _TRAIN_FILTER_HPP_
#define _TRAIN_FILTER_HPP_

#include <iostream>
#include <string>
#include <math.h>
#include <algorithm>
#include <Eigen/Dense>
#include "parameters.hpp"
#include "matrix_operator.hpp"
#include "ffttools.hpp"
#include "eco_util.hpp"

namespace eco_tracker {

typedef std::vector<std::vector<Eigen::MatrixXcf> > EcoFeats;

struct CG_state {
	EcoFeats p, r_prev;
	float rho;
};

void trainFilterJoint(
    const EcoFeats& xlf,
	const EcoFeats& sample_energy,
    const std::vector<Eigen::MatrixXcf>& reg_filter,
	const std::vector<float>& reg_energy,
    const std::vector<Eigen::MatrixXcf>& proj_energy,
    const std::vector<Eigen::MatrixXcf>& yf,
    const EcoParameters& params,
    EcoFeats& hf,
    std::vector<Eigen::MatrixXcf>& projection_matrix);

ECO_Train buildLhsOperationJoint(
    const ECO_Train& f_delta_P,
    const EcoFeats& samples_f,
    const std::vector<Eigen::MatrixXcf>& reg_filter,
    const EcoFeats& init_samplef,
    const std::vector<Eigen::MatrixXcf>& init_samplef_H,
    const EcoFeats &init_hf,
    const float& proj_lambda);

ECO_Train runPCGEcoJoint(
    const ECO_Train& f_delta_P,
    const ECO_Train& rhs_sample,
    const ECO_Train& diag_M,
    const EcoFeats &init_samplef_proj,
    const std::vector<Eigen::MatrixXcf>& reg_filter,
    const EcoFeats& init_samplef,
    const std::vector<Eigen::MatrixXcf>& init_samplef_H,
    const EcoFeats &init_hf,
    const EcoParameters& params);

void trainFilter(
    const std::vector<EcoFeats>& samplesf,
    const std::vector<Eigen::MatrixXcf>& reg_filter,
    const std::vector<float>& sample_weights,
    const EcoFeats& sample_energy,
    const std::vector<float>& reg_energy,
    const std::vector<Eigen::MatrixXcf>& yf,
    const EcoParameters& params,
    EcoFeats& hf);

EcoFeats buildLhsOperation(
    const EcoFeats& hf,
    const std::vector<EcoFeats>& samples_f,
    const std::vector<Eigen::MatrixXcf>& reg_filter,
    const std::vector<float>& sample_weights);

void runPCGEcoFilter(
    const vector<EcoFeats>& samplesf,
    const vector<Eigen::MatrixXcf>& reg_filter,
    const vector<float>& sample_weights,
    const EcoFeats& rhs_samplef,
    const EcoFeats& diag_M,
    const EcoParameters& params,
    EcoFeats& hf);

EcoFeats computeFeatureMutiply2(
    const EcoFeats& a,
    const std::vector<Eigen::MatrixXcf>& b);

std::vector<Eigen::MatrixXcf> computeFeatureMutiply(
    const EcoFeats& a,
    const EcoFeats& b);

ECO_Train EcoFeatureDotDivideJoint(
    const ECO_Train &a,
    const ECO_Train &b);

float getInnerProductJoint(
    const ECO_Train &a,
    const ECO_Train &b);

float getInnerProduct(
    const EcoFeats &a,
    const EcoFeats &b);

void FilterSymmetrize(EcoFeats &hf);

} // namespace eco_tracker
#endif