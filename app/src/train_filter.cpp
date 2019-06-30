#include "train_filter.hpp"

namespace eco_tracker {

void trainFilterJoint(
    const EcoFeats& xlf,
	const EcoFeats& sample_energy,
	const std::vector<Eigen::MatrixXcf>& reg_filter,
	const std::vector<float>& reg_energy,
	const std::vector<Eigen::MatrixXcf>& proj_energy,
    const std::vector<Eigen::MatrixXcf>& yf,
    const EcoParameters& params,
    EcoFeats& hf,
    std::vector<Eigen::MatrixXcf>& projection_matrix) {
	// step 1. Initial filter and projection matrix
	std::vector<int> lf_ind;
	for (size_t i = 0; i < hf.size(); i++) {
		lf_ind.push_back(hf[i][0].rows() * (hf[i][0].cols() - 1));
	}
	EcoFeats init_samplesf = xlf;
	std::vector<Eigen::MatrixXcf> init_samplesf_H;

	for (size_t i = 0; i < xlf.size(); i++) {
		Eigen::MatrixXcf temp(static_cast<int>(xlf[i].size()), xlf[i][0].cols() * xlf[i][0].rows());
		for (size_t j = 0; j < xlf[i].size(); j++) {
			Eigen::MatrixXcf xlf_ij = xlf[i][j];
            Eigen::Map<Eigen::VectorXcf> tmp_v(xlf_ij.data(), xlf[i][j].cols() * xlf[i][j].rows());
            Eigen::MatrixXcf tmp_vt = tmp_v.transpose();
            temp.row(j) = tmp_vt;
		}
		init_samplesf_H.push_back(temp.conjugate());
	}
	// step 2. build preconditioner diag matrix
	EcoFeats diag_M1;
	for (size_t i = 0; i < sample_energy.size(); i++) {
        Eigen::MatrixXcf mean = Eigen::MatrixXcf::Zero(sample_energy[i][0].rows(), sample_energy[i][0].cols());
		for (size_t j = 0; j < sample_energy[i].size(); j++) {
			mean += sample_energy[i][j];
		}
		mean = mean / static_cast<int>(sample_energy[i].size());

		std::vector<Eigen::MatrixXcf> temp_vec;
		for (size_t j = 0; j < sample_energy[i].size(); j++) {
            Eigen::MatrixXcf one_mat;
            one_mat.setOnes(sample_energy[i][0].rows(), sample_energy[i][0].cols());
			Eigen::MatrixXcf temp = (1 - params.precond_data_param) * mean + params.precond_data_param * sample_energy[i][j];
			temp = temp * (1 - params.precond_reg_param) + params.precond_reg_param * reg_energy[i] * one_mat;
			temp_vec.push_back(temp);
		}
		diag_M1.push_back(temp_vec);
	}
	std::vector<Eigen::MatrixXcf> diag_M2;
	for (size_t i = 0; i < proj_energy.size(); i++) {
        Eigen::MatrixXcf one_mat;
        one_mat.setOnes(proj_energy[i].rows(), proj_energy[i].cols());		
		diag_M2.push_back(params.precond_proj_param * (proj_energy[i] + params.projection_reg * one_mat));
	}
    ECO_Train diag_M;
    diag_M.part1 = diag_M1;
    diag_M.part2 = diag_M2;
	// step 3. build right hand part(part 1: A^H * y, part 2: B^H * y - lambda * P)
	for (size_t i = 0; i < (size_t)params.init_GN_iter; i++) {
		EcoFeats init_samplef_proj = projectFeature(init_samplesf, projection_matrix);
		EcoFeats init_hf = hf;
		// build A^H * y
		EcoFeats rhs_samplef1 = computeFeatureMutiply2(init_samplef_proj, yf);
		// build B^H * y - lambda * P
		std::vector<Eigen::MatrixXcf> rhs_samplef2;
		EcoFeats fyf = computeFeatureMutiply2(hf, yf);
		std::vector<Eigen::MatrixXcf> fyf_vec = vectorFeature(fyf);
		for (size_t i = 0; i < init_samplesf_H.size(); i++) {
			Eigen::MatrixXcf fyf_vec_T = fyf_vec[i].transpose();
			Eigen::MatrixXcf l1 = init_samplesf_H[i] * fyf_vec_T;
			Eigen::MatrixXcf col = init_samplesf_H[i].block(0, lf_ind[i],
				init_samplesf_H[i].rows(), init_samplesf_H[i].cols() - lf_ind[i]);
			Eigen::MatrixXcf row = fyf_vec_T.block(lf_ind[i], 0,
				fyf_vec_T.rows() - lf_ind[i], fyf_vec_T.cols());

			Eigen::MatrixXcf l2 = col * row;
			Eigen::MatrixXcf temp = 2.f * (l1 - l2) -
						   params.projection_reg * projection_matrix[i];
			rhs_samplef2.push_back(temp);
		}
		ECO_Train rhs_samplef;
        rhs_samplef.part1 = rhs_samplef1;
        rhs_samplef.part2 = rhs_samplef2;

		std::vector<Eigen::MatrixXcf> deltaP;
		for (size_t i = 0; i < projection_matrix.size(); i++) {
            Eigen::MatrixXcf tmp = Eigen::MatrixXcf::Zero(projection_matrix[i].rows(), projection_matrix[i].cols());
			deltaP.push_back(tmp);
		}
		ECO_Train jointFP;
		jointFP.part1 = hf;
		jointFP.part2 = deltaP;

		// step 4. run PCG algorithm to get result
		ECO_Train outFP = runPCGEcoJoint(jointFP,
									     rhs_samplef,
									     diag_M,
                                         init_samplef_proj,
									     reg_filter,
									     init_samplesf,
									     init_samplesf_H,
									     init_hf,
                                         params);
		FilterSymmetrize(outFP.part1);
		hf = outFP.part1;
		// step 5. update Projection matrix
		projection_matrix = projection_matrix + outFP.part2;
	}
}

ECO_Train buildLhsOperationJoint(const ECO_Train& f_delta_P,
                                 const EcoFeats& samples_f,
                                 const std::vector<Eigen::MatrixXcf>& reg_filter,
                                 const EcoFeats& init_samplef,
                                 const std::vector<Eigen::MatrixXcf>& init_samplef_H,
                                 const EcoFeats &init_hf,
                                 const float& proj_lambda) {

    ECO_Train f_delta_P_out;
    EcoFeats f_delta;
    std::vector<Eigen::MatrixXcf> deltaP;
	// step 1. find the maximum of Kd
	int num_features = f_delta_P.part1.size();
    std::vector<int> arr_sz;
    arr_sz.reserve(num_features);
    for (int n = 0; n < num_features; ++n) {
        arr_sz.push_back(f_delta_P.part1.at(n).at(0).rows());
    }
	auto cmp = [](const int& lhs, const int& rhs) {
		return lhs < rhs;
	};
	std::vector<int>::iterator pos = std::max_element(arr_sz.begin(), arr_sz.end(), cmp);
	size_t k1 = pos - arr_sz.begin(); // index
    int maxK = *pos;

	// step2. build A^H * A * f
    std::vector<Eigen::MatrixXcf> Afn = computeFeatureMutiply(samples_f, f_delta_P.part1);
    Eigen::MatrixXcf Af = Eigen::MatrixXcf::Zero(Afn[k1].rows(), Afn[k1].cols());
    for (int n = 0; n < static_cast<int>(Afn.size()); ++n) {
        int offset = (maxK - Afn[n].rows()) / 2;
        Af.block(offset, offset, Afn[n].rows(), Afn[n].cols()) = Afn[n] + Af.block(offset, offset, Afn[n].rows(), Afn[n].cols());
    }

    for (int n = 0; n < num_features; ++n) {
        std::vector<Eigen::MatrixXcf> tmp;
        for (int m = 0; m < static_cast<int>(f_delta_P.part1[n].size()); ++m) {
            int offset = (maxK - Afn[n].rows()) / 2;
            Eigen::MatrixXcf res = ((Af.block(offset, offset, Afn[n].rows(),
			    Afn[n].cols())).conjugate()).cwiseProduct(samples_f[n][m]);
            tmp.push_back(res.conjugate());
        }
        f_delta.push_back(tmp);
    }

    // step 3. build W^H * W * f
    for (int n = 0; n < num_features; ++n) {
        int reg_pad = std::min(reg_filter[n].cols() - 1, f_delta_P.part1[n][0].cols() - 1);
        for (int m = 0; m < static_cast<int>(f_delta_P.part1[n].size()); ++m) {
            Eigen::MatrixXcf temp_conv;
            if (reg_pad == 0) {
               temp_conv = f_delta_P.part1[n][m];
            } else {
				int r = f_delta_P.part1[n][m].rows();
				int c = f_delta_P.part1[n][m].cols();
                temp_conv = Eigen::MatrixXcf::Zero(r, c + reg_pad);
                Eigen::MatrixXcf temp = f_delta_P.part1[n][m].block(0, c - reg_pad - 1, r, reg_pad);
                Eigen::MatrixXcf temp_flip1 = temp.colwise().reverse().eval();
                Eigen::MatrixXcf temp_flip2 = temp_flip1.rowwise().reverse().eval();
                temp_conv.leftCols(c) = f_delta_P.part1[n][m];
                temp_conv.block(0, c, r, reg_pad) = temp_flip2.conjugate();               
            }
            temp_conv = fft_tools::Convolution2(temp_conv, reg_filter[n], 0);
			temp_conv =
				fft_tools::Convolution2(temp_conv.leftCols(temp_conv.cols() - reg_pad),
							 reg_filter[n], 1);

			// A^H * A * f + W^H * W * f
			f_delta[n][m] += temp_conv;
        }
    }

	// step 4. build A^H * B * dp
    EcoFeats samplef_proj = projectFeature(init_samplef, f_delta_P.part2);
    std::vector<Eigen::MatrixXcf> BPn = computeFeatureMutiply(samplef_proj, init_hf);
    Eigen::MatrixXcf BP = Eigen::MatrixXcf::Zero(BPn[k1].rows(), BPn[k1].cols());
    for (int n = 0; n < static_cast<int>(BPn.size()); ++n) {
        int offset = (maxK - BPn[n].rows()) / 2;
        BP.block(offset, offset, BPn[n].rows(), BPn[n].cols()) = BPn[n] + BP.block(offset, offset, BPn[n].rows(), BPn[n].cols());
    }

    // step 5. build B^H * A * f + B^H * B * dp + lambda * dp
    EcoFeats fH_BP, fH_Af;
    for (int n = 0; n < num_features; ++n) {
        std::vector<Eigen::MatrixXcf> fH_BP_vec, fH_Af_vec;
        for (int m = 0; m < static_cast<int>(f_delta.at(n).size()); ++m) {
            int offset = (maxK - f_delta[n][0].rows()) / 2;
            Eigen::MatrixXcf temp = (BP.block(offset,
                    offset, f_delta[n][0].rows(), f_delta[n][0].cols())).cwiseProduct((samples_f[n][m]).conjugate());
            f_delta[n][m] += temp;
            Eigen::MatrixXcf fH_BP_tmp = (init_hf[n][m].conjugate()).cwiseProduct(
                BP.block(offset, offset, f_delta[n][0].rows(), f_delta[n][0].cols()));
            Eigen::MatrixXcf fH_Af_tmp = (init_hf[n][m].conjugate()).cwiseProduct(
                Af.block(offset, offset, f_delta[n][0].rows(), f_delta[n][0].cols()));
            fH_BP_vec.push_back(fH_BP_tmp);
            fH_Af_vec.push_back(fH_Af_tmp);
        }
        fH_BP.push_back(fH_BP_vec);
        fH_Af.push_back(fH_Af_vec);
    }

    for (int n = 0; n < num_features; ++n) {
		int fi = f_delta[n][0].rows() * (f_delta[n][0].cols() - 1) + 0;
        int end_cols = init_samplef_H.at(n).cols() - fi + 1;
        Eigen::MatrixXcf BH_Af = init_samplef_H[n] * (vectorFeature(fH_Af)[n].transpose()) -
			init_samplef_H[n].rightCols(end_cols) * (vectorFeature(fH_Af)[n].rightCols(end_cols).transpose());
		BH_Af = 2 * BH_Af;

		// build B^H * A * f
        Eigen::MatrixXcf BH_BP = init_samplef_H[n] * (vectorFeature(fH_BP)[n].transpose()) -
			init_samplef_H[n].rightCols(end_cols) * (vectorFeature(fH_BP)[n].rightCols(end_cols).transpose());
		BH_BP = 2 * BH_BP + proj_lambda * f_delta_P.part2[n];
        // build B^H * A * f + B^H * B * dp + lambda * dp
		deltaP.push_back(BH_Af + BH_BP);
    }
    f_delta_P_out.part1 = f_delta;
    f_delta_P_out.part2 = deltaP;

    return f_delta_P_out;
}

ECO_Train runPCGEcoJoint(const ECO_Train& f_delta_P,
                         const ECO_Train& rhs_sample,
                         const ECO_Train& diag_M,
                         const EcoFeats &init_samplef_proj,
                         const std::vector<Eigen::MatrixXcf>& reg_filter,
                         const EcoFeats& init_samplef,
                         const std::vector<Eigen::MatrixXcf>& init_samplef_H,
                         const EcoFeats &init_hf,
                         const EcoParameters& params) {

	int maxit = params.CG_opts.maxit;
	bool existM1 = true;
	if (diag_M.part2.empty()) {
		existM1 = false;
	}
	ECO_Train x = f_delta_P;

	ECO_Train p, r_prev;
	float rho = 1, rho1, alpha, beta;

	ECO_Train Ax = buildLhsOperationJoint(x,
                                          init_samplef_proj,
                                          reg_filter,
                                          init_samplef,
                                          init_samplef_H,
                                          init_hf,
                                          params.projection_reg);
	ECO_Train r = rhs_sample;
	r = r - Ax;

	for (size_t ii = 0; ii < (size_t)maxit; ii++) {
		ECO_Train y, z;
		if (existM1) {
			y = EcoFeatureDotDivideJoint(r, diag_M);
		} else {
			y = r;
		}
		z = y;

		rho1 = rho;
		rho = getInnerProductJoint(r, z);
		if ((rho == 0) || (std::abs(rho) >= INT_MAX) || std::isnan(rho)) {
			break;
		}

		if (ii == 0 && p.part2.empty()) {
			p = z;
		} else {
			// Use Fletcher-Reeves
			if (params.CG_opts.CG_use_FR) {
				beta = rho / rho1;
			} else {
				// Use Polak-Ribiere
				float rho2 = getInnerProductJoint(r_prev, z);
				beta = (rho - rho2) / rho1;
			}
			if ((beta == 0) || (std::abs(beta) >= INT_MAX) || std::isnan(beta)) {
				break;
			}
			beta = cv::max(0.0f, beta);
			p = z + p * beta;
		}

		ECO_Train q = buildLhsOperationJoint(p,
                                             init_samplef_proj,
                                             reg_filter,
                                             init_samplef,
                                             init_samplef_H,
                                             init_hf,
                                             params.projection_reg);

		float pq = getInnerProductJoint(p, q);

		if (pq <= 0 || (std::abs(pq) > INT_MAX) || std::isnan(pq)) {
			break;
			// assert(0 && "error: GC condition is not matched");
		} else {
			if (params.CG_opts.CG_standard_alpha) {
				alpha = rho / pq;
			} else {
				alpha = getInnerProductJoint(p, r) / pq;
			}
		}
		if ((std::abs(alpha) > INT_MAX) || std::isnan(alpha)) {
			assert(0 && "GC condition alpha is not matched");
		}

		// Save old r if not using FR formula for beta
		// Use Polak-Ribiere
		if (!params.CG_opts.CG_use_FR) {
			r_prev = r;
		}
		x = x + p * alpha;

		if (ii < (size_t)maxit){
			r = r - q * alpha;
		}
	}
	return x;
}

void trainFilter(const std::vector<EcoFeats>& samplesf,
                  const std::vector<Eigen::MatrixXcf>& reg_filter,
				  const std::vector<float>& sample_weights,
				  const EcoFeats& sample_energy,
				  const std::vector<float>& reg_energy,
                  const std::vector<Eigen::MatrixXcf>& yf,
                  const EcoParameters& params,
                  EcoFeats& hf) {

	// (A^H * Gamma * A + W^H * W) * f = A^H * Gamma * y
	// step 1. A^H * Gamma * y
	EcoFeats rhs_samplef = samplesf[0] * sample_weights[0];
	for (size_t i = 1; i < samplesf.size(); i++) {
		rhs_samplef = samplesf[i] * sample_weights[i] +
					  rhs_samplef;
	}

    EcoFeats tmp_rhs_samplef;
	for (size_t i = 0; i < rhs_samplef.size(); i++) {
		std::vector<Eigen::MatrixXcf> temp;
		for (size_t j = 0; j < rhs_samplef[i].size(); j++) {
            Eigen::MatrixXcf tmp_mat = (rhs_samplef[i][j].conjugate()).cwiseProduct(yf[i]);
			temp.push_back(tmp_mat);
		}
		tmp_rhs_samplef.push_back(temp);
	}
    rhs_samplef = tmp_rhs_samplef;

	// step 2. build preconditioner diag matrix
	EcoFeats diag_M;
	for (size_t i = 0; i < sample_energy.size(); i++) {
        Eigen::MatrixXcf mean = Eigen::MatrixXcf::Zero(sample_energy[i][0].rows(), sample_energy[i][0].cols());
		for (size_t j = 0; j < sample_energy[i].size(); j++) {
			mean += sample_energy[i][j];
		}
		mean = mean / static_cast<int>(sample_energy[i].size());

		std::vector<Eigen::MatrixXcf> temp_vec;
		for (size_t j = 0; j < sample_energy[i].size(); j++) {
            Eigen::MatrixXcf one_mat;
            one_mat.setOnes(sample_energy[i][0].rows(), sample_energy[i][0].cols());
			Eigen::MatrixXcf temp = (1 - params.precond_data_param) * mean + params.precond_data_param * sample_energy[i][j];
			temp = temp * (1 - params.precond_reg_param) + params.precond_reg_param * reg_energy[i] * one_mat;
			temp_vec.push_back(temp);
		}
		diag_M.push_back(temp_vec);
	}
	// step 3. run PCG algorithm to get result
	runPCGEcoFilter(samplesf,
					reg_filter,
					sample_weights,
					rhs_samplef,
					diag_M,
                    params,
					hf);
}

EcoFeats buildLhsOperation(const EcoFeats& hf,
                           const std::vector<EcoFeats>& samples_f,
                           const std::vector<Eigen::MatrixXcf>& reg_filter,
                           const std::vector<float>& sample_weights) {
	// (A^H * Gamma * A + W^H * W) * f = A^H * Gamma * y
	// step 1. find the maximum of Kd
    int num_features = static_cast<int>(hf.size());
    std::vector<int> arr_Kd;
    arr_Kd.reserve(num_features);
    for (int n = 0; n < num_features; ++n) {
        arr_Kd.push_back(hf.at(n).at(0).rows());
    }
	auto cmp = [](const int& lhs, const int& rhs) {
		return lhs < rhs;
	};

	std::vector<int>::iterator pos = std::max_element(arr_Kd.begin(), arr_Kd.end(), cmp);
	size_t k1 = pos - arr_Kd.begin();
    int maxK = *pos;

    // step 2. build A^H * Gamma * A * f
    std::vector<Eigen::MatrixXcf> Af_vec;
    Af_vec.reserve(static_cast<int>(samples_f.size()));
    for (int j = 0; j < static_cast<int>(samples_f.size()); ++j) {
        std::vector<Eigen::MatrixXcf> Ajfn = computeFeatureMutiply(samples_f[j], hf);
        Eigen::MatrixXcf Ajf = Eigen::MatrixXcf::Zero(Ajfn[k1].rows(), Ajfn[k1].cols());
        for (int n = 0; n < static_cast<int>(Ajfn.size()); ++n) {
            int offset = (maxK - Ajfn[n].rows()) / 2;
            Ajf.block(offset, offset, Ajfn[n].rows(), Ajfn[n].cols()) = Ajfn[n] + Ajf.block(offset, offset, Ajfn[n].rows(), Ajfn[n].cols());
        }
        Af_vec.push_back((Ajf * sample_weights[j]).conjugate());
    }

    EcoFeats AH_gamma_Af;
    for (int n = 0; n < num_features; ++n) {
        std::vector<Eigen::MatrixXcf> tmp;
        for (int m = 0; m < static_cast<int>(hf[n].size()); ++m) {
            int offset = (maxK - hf[n][m].rows()) / 2;
            Eigen::MatrixXcf res = Eigen::MatrixXcf::Zero(hf[n][m].rows(), hf[n][m].cols());
            for (int j = 0; j < static_cast<int>(Af_vec.size()); ++j) {
                res += (Af_vec[j].block(offset, offset, hf[n][m].rows(),
				    hf[n][m].cols())).cwiseProduct(samples_f[j][n][m]);
            }
            tmp.push_back(res.conjugate());
        }
        AH_gamma_Af.push_back(tmp);
    }

    EcoFeats hf_out;
    hf_out = AH_gamma_Af;
    // step 3. build W^H * W * f
    for (int n = 0; n < num_features; ++n) {
        int reg_pad = std::min(reg_filter[n].cols() - 1, hf[n][0].cols() - 1);
        for (int m = 0; m < static_cast<int>(hf[n].size()); ++m) {
            Eigen::MatrixXcf temp_conv;
            if (reg_pad == 0) {
               temp_conv = hf[n][m];
            } else {
				int r = hf[n][m].rows();
				int c = hf[n][m].cols();
                temp_conv = Eigen::MatrixXcf::Zero(r, c + reg_pad);
                Eigen::MatrixXcf temp = hf[n][m].block(0, c - reg_pad - 1, r, reg_pad);
                Eigen::MatrixXcf temp_flip1 = temp.colwise().reverse().eval();
                Eigen::MatrixXcf temp_flip2 = temp_flip1.rowwise().reverse().eval();
                temp_conv.leftCols(c) = hf[n][m];
                temp_conv.block(0, c, r, reg_pad) = temp_flip2.conjugate();               
            }
            temp_conv = fft_tools::Convolution2(temp_conv, reg_filter[n], 0);
			temp_conv =
				fft_tools::Convolution2(temp_conv.leftCols(temp_conv.cols() - reg_pad),
							 reg_filter[n], 1);

			// A^H * Gamma * A * f + W^H * W * f
			hf_out[n][m] += temp_conv;           
        }
    }
    return hf_out;
}

void runPCGEcoFilter(const vector<EcoFeats>& samplesf,
					 const vector<Eigen::MatrixXcf>& reg_filter,
					 const vector<float>& sample_weights,
					 const EcoFeats& rhs_samplef,
					 const EcoFeats& diag_M,
                     const EcoParameters& params,
                     EcoFeats& hf) {

	int maxit = params.CG_opts.maxit;
	bool existM1 = true;
	if (diag_M.empty()) {
		existM1 = false;
	}
	EcoFeats x = hf;

	// Load the CG state
	EcoFeats p, r_prev;
	float rho = 1, rho1, alpha, beta;

    static CG_state state;
	if (!state.p.empty()) {
		p = state.p;
		rho = state.rho / params.CG_opts.init_forget_factor;
		if (!params.CG_opts.CG_use_FR) {
			// Use Polak-Ribiere
			r_prev = state.r_prev;
		}
	}

	EcoFeats Ax = buildLhsOperation(x,
									samplesf,
									reg_filter,
									sample_weights);
	EcoFeats r = rhs_samplef - Ax;

	for (size_t ii = 0; ii < (size_t)maxit; ii++) {
		EcoFeats y, z;
		if (existM1) {
			y = EcoFeatureDotDivide(r, diag_M);
		} else {
			y = r;
		}
		z = y;

		rho1 = rho;
		rho = getInnerProduct(r, z);
		if ((rho == 0) || (std::abs(rho) >= INT_MAX) || std::isnan(rho)) {
			break;
		}

		if (ii == 0 && p.empty()) {
			p = z;
		} else {
			if (params.CG_opts.CG_use_FR) {
				// Use Fletcher-Reeves
				beta = rho / rho1;
			} else {
				// Use Polak-Ribiere
				float rho2 = getInnerProduct(r_prev, z);
				beta = (rho - rho2) / rho1;
			}
			if ((beta == 0) || (std::abs(beta) >= INT_MAX) || std::isnan(beta)) {
				break;
			}
			beta = cv::max(0.0f, beta);
			p = z + p * beta;
		}

		EcoFeats q = buildLhsOperation(p,
									   samplesf,
									   reg_filter,
									   sample_weights);

		float pq = getInnerProduct(p, q);

		if (pq <= 0 || (std::abs(pq) > INT_MAX) || std::isnan(pq)) {
			assert(0 && "error: GC condition is not matched");
		} else {
			if (params.CG_opts.CG_standard_alpha) {
				alpha = rho / pq;
			} else {
				alpha = getInnerProduct(p, r) / pq;
			}
		}
		if ((std::abs(alpha) > INT_MAX) || std::isnan(alpha)) {
			assert(0 && "GC condition alpha is not matched");
		}

		// Save old r if not using FR formula for beta
		// Use Polak-Ribiere
		if (!params.CG_opts.CG_use_FR) {
			r_prev = r;
		}

		x = x + p * alpha;

		if (ii < (size_t)maxit) {
			r = r - q * alpha;
		}
	}

	state.p = p;
	state.rho = rho;
	if (!params.CG_opts.CG_use_FR) {
		// Use Polak-Ribiere
		state.r_prev = r_prev;
	}
	hf = x;
}

EcoFeats computeFeatureMutiply2(const EcoFeats& a,
                                const std::vector<Eigen::MatrixXcf>& b) {
    EcoFeats res;
	for (size_t i = 0; i < a.size(); i++) {
		std::vector<Eigen::MatrixXcf> temp;
		for (size_t j = 0; j < a[i].size(); j++) {
            Eigen::MatrixXcf tmp_mat = (a[i][j].conjugate()).cwiseProduct(b[i]);
			temp.push_back(tmp_mat);
		}
		res.push_back(temp);
	}
    return res;
}

std::vector<Eigen::MatrixXcf> computeFeatureMutiply(const EcoFeats& a,
                                                    const EcoFeats& b) {
    std::vector<Eigen::MatrixXcf> res;
    res.reserve((int)a.size());
    if (a.size() != b.size()) {
        std::clog << "two inputs have unmatched size!" << std::endl;
        assert(false);
    }
    for (size_t i = 0; i < a.size(); ++i) {
        Eigen::MatrixXcf temp = Eigen::MatrixXcf::Zero(a[i][0].rows(), a[i][0].cols());
        for (size_t j = 0; j < a[i].size(); ++j) {
            temp += a[i][j].cwiseProduct(b[i][j]);
        }
        res.push_back(temp);
    }
	return res;
}

ECO_Train EcoFeatureDotDivideJoint(const ECO_Train &a, const ECO_Train &b) {

	ECO_Train res;
	EcoFeats up_rs;
	std::vector<Eigen::MatrixXcf> low_rs;
	for (size_t i = 0; i < a.part1.size(); i++) {
		std::vector<Eigen::MatrixXcf> tmp;
		for (size_t j = 0; j < a.part1[i].size(); j++) {
			tmp.push_back(a.part1[i][j].cwiseQuotient(b.part1[i][j]));
		}
		up_rs.push_back(tmp);
		low_rs.push_back(a.part2[i].cwiseQuotient(b.part2[i]));
	}
	res.part1 = up_rs;
	res.part2 = low_rs;
	return res;
}

float getInnerProductJoint(const ECO_Train &a, const ECO_Train &b) {
	float ip = 0;
	for (size_t i = 0; i < a.part1.size(); i++) {
		for (size_t j = 0; j < a.part1[i].size(); j++) {
			int c = a.part1[i][j].cols();
            Eigen::MatrixXcf a_c = a.part1[i][j].col(c-1);
            Eigen::MatrixXcf b_c = b.part1[i][j].col(c-1);
			ip += 2 * (((a.part1[i][j].conjugate()).cwiseProduct(b.part1[i][j])).sum()).real() -
                (((a_c.conjugate()).cwiseProduct(b_c)).sum()).real();
		}
        ip += (((a.part2[i].conjugate()).cwiseProduct(b.part2[i])).sum()).real();
	}
	return ip;
}

float getInnerProduct(const EcoFeats &a, const EcoFeats &b) {
	float ip = 0;
	for (size_t i = 0; i < a.size(); i++) {
		for (size_t j = 0; j < a[i].size(); j++) {
			int c = a[i][j].cols();
            Eigen::MatrixXcf a_c = a[i][j].col(c-1);
            Eigen::MatrixXcf b_c = b[i][j].col(c-1);
			ip += 2 * (((a[i][j].conjugate()).cwiseProduct(b[i][j])).sum()).real() -
                (((a_c.conjugate()).cwiseProduct(b_c)).sum()).real();
		}
	}
	return ip;
}

void FilterSymmetrize(EcoFeats &hf) {
	for (size_t i = 0; i < hf.size(); i++) {
		int dc_ind = (hf[i][0].rows() + 1) / 2;
		for (size_t j = 0; j < (size_t)hf[i].size(); j++) {
			int c = hf[i][j].cols() - 1;
			for (size_t r = dc_ind; r < (size_t)hf[i][j].rows(); r++) {
				hf[i][j](r, c) = std::conj(hf[i][j](2 * dc_ind - r - 2, c));
			}
		}
	}
}

} // namespace eco_tracker