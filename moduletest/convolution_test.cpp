#include "ffttools.hpp"
#include "parameters.hpp"

int main() {

    int Irows = 4;
    int Icols = 5;
    int krows = 2;
    int kcols = 3;

    Eigen::MatrixXcf In(Irows, Icols);
    In << 1, 2, 3, 4, 5,
          2, 3, 4, 5, 6,
          3, 4, 5, 6, 7,
          4, 5, 6, 7, 8;
    Eigen::MatrixXcf k(krows, kcols);
    k << 1, 2, 1,
         2, 3, 2;

    Eigen::MatrixXcf res_full = fft_tools::Convolution2(In, k, 0);

    Eigen::MatrixXcf res_valid = fft_tools::Convolution2(In, k, 1);

    std::cout << "--res full: \n" << res_full << std::endl;

    std::cout << "--res valid: \n" << res_valid << std::endl;

    return 0;
}