#ifndef _OPTIMIZE_SCORES_HPP_
#define _OPTIMIZE_SCORES_HPP_

#include <iostream>
#include <string>
#include <math.h>
#include <algorithm>
#include <Eigen/Dense>
#include "ffttools.hpp"

namespace eco_tracker {

  void computeScores(
      const std::vector<Eigen::MatrixXcf>& scores_fs,
      const int& max_iteration,
      int& scale_ind,
      float& opt_score,
      float& opt_pos_x,
      float& opt_pos_y);

} // namespace eco_tracker
#endif
