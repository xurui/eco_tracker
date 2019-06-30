#include "ffttools.hpp"

namespace fft_tools {
Eigen::MatrixXcf fft(
    const Eigen::MatrixXcf& timeMat) {

    Eigen::FFT<float> fft;
    Eigen::MatrixXcf freqMat(timeMat.rows(), timeMat.cols());

    for (int k = 0; k < timeMat.rows(); k++) {
        Eigen::VectorXcf tmpOut;
        Eigen::MatrixXcf tmp_r = timeMat.row(k);
        Eigen::Map<Eigen::VectorXcf> timeMat_row(tmp_r.data(), timeMat.cols());
        fft.fwd(tmpOut, timeMat_row);
        Eigen::MatrixXcf tmp_row_k = tmpOut.transpose();
        freqMat.row(k) = tmp_row_k;
    }

    for (int k = 0; k < timeMat.cols(); k++) {
        Eigen::VectorXcf tmpOut;
        Eigen::MatrixXcf tmp_c = freqMat.col(k);
        Eigen::Map<Eigen::VectorXcf> freqMat_col(tmp_c.data(), timeMat.rows());
        fft.fwd(tmpOut, freqMat_col);
        freqMat.col(k) = tmpOut;
    }
	return freqMat;
}

Eigen::MatrixXcf fft(
    const Eigen::MatrixXf& timeMat) {

    Eigen::FFT<float> fft;
    Eigen::MatrixXcf freqMat(timeMat.rows(), timeMat.cols());

    for (int k = 0; k < timeMat.rows(); k++) {
        Eigen::VectorXcf tmpOut;
        Eigen::MatrixXf tmp_r = timeMat.row(k);
        Eigen::Map<Eigen::VectorXf> timeMat_row(tmp_r.data(), timeMat.cols());
        fft.fwd(tmpOut, timeMat_row);
        Eigen::MatrixXcf tmp_row_k = tmpOut.transpose();
        freqMat.row(k) = tmp_row_k;
    }

    for (int k = 0; k < timeMat.cols(); k++) {
        Eigen::VectorXcf tmpOut;
        Eigen::MatrixXcf tmp_c = freqMat.col(k);
        Eigen::Map<Eigen::VectorXcf> freqMat_col(tmp_c.data(), timeMat.rows());
        fft.fwd(tmpOut, freqMat_col);
        freqMat.col(k) = tmpOut;
    }
	return freqMat;
}

Eigen::MatrixXcf ifft(
    const Eigen::MatrixXcf& freqMat) {

    Eigen::FFT<float> fft;
    Eigen::MatrixXcf timeMat(freqMat.rows(), freqMat.cols());

    for (int k = 0; k < freqMat.rows(); k++) {
        Eigen::VectorXcf tmpOut;
        Eigen::MatrixXcf tmp_r = freqMat.row(k);
        Eigen::Map<Eigen::VectorXcf> freqMat_row(tmp_r.data(), freqMat.cols());
        fft.inv(tmpOut, freqMat_row);
        Eigen::MatrixXcf tmp_row_k = tmpOut.transpose();
        timeMat.row(k) = tmp_row_k;
    }

    for (int k = 0; k < freqMat.cols(); k++) {
        Eigen::VectorXcf tmpOut;
        Eigen::MatrixXcf tmp_c = timeMat.col(k);
        Eigen::Map<Eigen::VectorXcf> timeMat_col(tmp_c.data(), freqMat.rows());
        fft.inv(tmpOut, timeMat_col);
        timeMat.col(k) = tmpOut;
    }
	return timeMat;
}
 
Eigen::MatrixXcf fftshift(Eigen::MatrixXcf x, bool inverse_flag) {

	Eigen::MatrixXcf y(x.rows(), x.cols());
	int w = x.cols(), h = x.rows();
	int rshift = inverse_flag ? h - h / 2 : h / 2,
		cshift = inverse_flag ? w - w / 2 : w / 2;

	y = circshift(x, rshift, cshift);
	return y;
}

Eigen::MatrixXcf circshift(Eigen::MatrixXcf data, int shift_r, int shift_c) {

	int row = data.rows();
	int col = data.cols();
	Eigen::MatrixXcf temp_m(row, col);
	Eigen::MatrixXcf res(row, col);
	int r = shift_r%row;
	int c = shift_c%col;
    // down shift
	if (r > 0) {
		temp_m.topRows(r) = data.bottomRows(r);
		temp_m.bottomRows(row - r) = data.topRows(row - r);
    // up shift
	} else if (r < 0) {
		temp_m.topRows(row + r) = data.bottomRows(row + r);
		temp_m.bottomRows(abs(r)) = data.topRows(std::abs(r));
	} else if (r == 0) {
		temp_m.array() = data.array();
	}

    // right shift
	if (c > 0) {
		res.leftCols(c) = temp_m.rightCols(c);
		res.rightCols(col - c) = temp_m.leftCols(col - c);
	// left shift
	} else if (c < 0) {
		res.leftCols(col + c) = temp_m.rightCols(col + c);
		res.rightCols(std::abs(c)) = temp_m.leftCols(std::abs(c)); 
	} else if (c == 0) {
		res = temp_m;
	}
	return res;
}

Eigen::MatrixXcf  Convolution2(
    const Eigen::MatrixXcf &I,
    const Eigen::MatrixXcf &kernel,
    int conv_mode) {

    int rows = I.rows() + kernel.rows() - 1;
    int cols = I.cols() + kernel.cols() - 1;

    Eigen::MatrixXcf temp = Eigen::MatrixXcf::Zero(rows, cols);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            for (int m = 0; m < kernel.rows(); ++m) {
                for (int n = 0; n < kernel.cols(); ++n) {
                    int I_r = r - m;
                    int I_c = c - n;
                    if (I_r < 0 || I_r >= I.rows() ||
                        I_c < 0 || I_c >= I.cols()) {
                        continue;
                    }
                    temp(r, c) += kernel(m, n) * I(I_r, I_c);
                }
            }
        }
    }

    Eigen::MatrixXcf res = temp;
    int kern_center_r = kernel.rows() / 2;
    int kern_center_c = kernel.cols() / 2;
    if (conv_mode == 0) {
        // for (int r = 0; r < rows; ++r) {
        //     for (int c = 0; c < cols; ++c) {
        //         if (r < kern_center_r || r >= rows - kern_center_r ||
        //             c < kern_center_c || c >= cols - kern_center_c) {
        //             res(r, c) = 0;
        //         }
        //     }
        // }
    } else if (conv_mode == 1) {
        rows = I.rows() - kernel.rows() + 1;
        cols = I.cols() - kernel.cols() + 1;
        res = temp.block(kernel.rows() - 1, kernel.cols() - 1, rows, cols);
    }
    return res;
}

std::complex<float> getMatValue(
    const Eigen::MatrixXcf& mat,
    int r, int c, const int& border_type) {
    
	int SizeX = mat.rows();
	int SizeY = mat.cols();
	switch (border_type) {
        case 0:		
            if ((r < 0) || (r >= SizeX) || (c < 0) || (c >= SizeY)) {
                return 0;
            } else {
                return mat(r,c);
            }
        case 1:	
            if (r < 0)
                r = 0;
            else if (r >= SizeX)
                r = SizeX - 1;
            if (c < 0)
                c = 0;
            else if (c >= SizeY)
                c = SizeY - 1;
            return mat(r,c);
        default :
            printf("border_type error!\n");
            assert(false);
	}
}

}