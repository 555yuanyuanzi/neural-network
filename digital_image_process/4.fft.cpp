//频率域理想高通、高斯，巴特沃斯滤波器
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

const double PI = 3.1415926;

void bit_reversal_permutation(vector<complex<double>>& data) {
    int n = data.size();
    if (n <= 1) return;
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) swap(data[i], data[j]);
    }
}


//一维快速傅里叶变换 (FFT)
void fft_1d(vector<complex<double>>& data, bool invert) {
    int n = data.size();
    if (n <= 1) return;

    bit_reversal_permutation(data);

    for (int len = 2; len <= n; len <<= 1) {
        double angle = 2 * PI / len * (invert ? -1 : 1);
        complex<double> wlen(cos(angle), sin(angle));
        for (int i = 0; i < n; i += len) {
            complex<double> w(1);
            for (int j = 0; j < len / 2; j++) {
                complex<double> u = data[i + j];
                complex<double> v = data[i + j + len / 2] * w;
                data[i + j] = u + v;
                data[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }

    if (invert) {
        for (auto& val : data) val /= n;
    }
}

//二维快速傅里叶变换 (2D FFT)
void fft_2d(vector<vector<complex<double>>>& matrix, bool invert) {
    int rows = matrix.size();
    if (rows == 0) return;
    int cols = matrix[0].size();
    if (cols == 0) return;
    if ((rows & (rows - 1)) != 0 || (cols & (cols - 1)) != 0) {
        cerr << "当前尺寸: " << cols << "x" << rows << endl;
    }

    //对每一行进行1D FFT
    for (int i = 0; i < rows; ++i) {
        fft_1d(matrix[i], invert);
    }

    //对每一列进行1D FFT
    vector<vector<complex<double>>> transposed(cols, vector<complex<double>>(rows));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            transposed[j][i] = matrix[i][j];
        }
    }
    for (int i = 0; i < cols; ++i) {
        fft_1d(transposed[i], invert);
    }
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = transposed[j][i];
        }
    }
}

//将频谱的四个象限进行对角交换
void fftshift_2d(vector<vector<complex<double>>>& matrix) {
    int rows = matrix.size();
    int cols = matrix[0].size();
    int cy = rows / 2;
    int cx = cols / 2;

    for (int y = 0; y < cy; ++y) {
        for (int x = 0; x < cx; ++x) {
            //交换第一和第四象限
            swap(matrix[y][x], matrix[y + cy][x + cx]);
            //交换第二和第三象限
            swap(matrix[y][x + cx], matrix[y + cy][x]);
        }
    }
}

//计算并创建用于显示的频谱图
Mat create_magnitude_spectrum_from_vector(const vector<vector<complex<double>>>& fft_result) {
    int rows = fft_result.size();
    int cols = fft_result[0].size();
    Mat mag_spectrum(rows, cols, CV_64F);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double mag = abs(fft_result[i][j]); // abs()对复数自动取模
            mag_spectrum.at<double>(i, j) = 20 * log(mag + 1e-9); // 加一个小数避免log(0)
        }
    }

    //归一化到0-255以便显示
    normalize(mag_spectrum, mag_spectrum, 0, 255, NORM_MINMAX);
    mag_spectrum.convertTo(mag_spectrum, CV_8U);
    return mag_spectrum;
}

//工作流程：数据准备、变换、滤波、逆变换、结果转换
Mat perform_ideal_high_pass_filter(const Mat& img_orig, double d0) {
    cout << "理想高通滤波, D0 = " << d0 << " ---" << endl;

    //1. 准备数据：填充到2的幂次尺寸
    int opt_rows = getOptimalDFTSize(img_orig.rows);
    int opt_cols = getOptimalDFTSize(img_orig.cols);
    Mat img_padded;
    copyMakeBorder(img_orig, img_padded, 0, opt_rows - img_orig.rows, 0, opt_cols - img_orig.cols, BORDER_CONSTANT, Scalar::all(0));
    
    int rows = img_padded.rows;
    int cols = img_padded.cols;

    //2. 将数据复制到我们的复数向量中
    vector<vector<complex<double>>> fft_data(rows, vector<complex<double>>(cols));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            fft_data[i][j] = complex<double>((double)img_padded.at<uchar>(i, j), 0.0);
        }
    }

    //3. FFT -> Shift
    fft_2d(fft_data, false);
    fftshift_2d(fft_data);

    //4. 应用理想高通滤波器 (中心置零)
    int center_row = rows / 2;
    int center_col = cols / 2;
    int filter_size = static_cast<int>(d0);
    for (int i = center_row - filter_size; i < center_row + filter_size; ++i) {
        for (int j = center_col - filter_size; j < center_col + filter_size; ++j) {
            if (i >= 0 && i < rows && j >= 0 && j < cols) {
                fft_data[i][j] = complex<double>(0, 0);
            }
        }
    }

    //5. Shift -> IFFT
    fftshift_2d(fft_data);
    fft_2d(fft_data, true);

    //6. 转换结果并返回
    Mat img_back(rows, cols, CV_64F);
    for (int i = 0; i < rows; ++i) for (int j = 0; j < cols; ++j) img_back.at<double>(i, j) = abs(fft_data[i][j]);
    normalize(img_back, img_back, 0, 255, NORM_MINMAX);
    img_back.convertTo(img_back, CV_8U);
    
    //裁剪回原始尺寸
    return img_back(Rect(0, 0, img_orig.cols, img_orig.rows));
}


Mat perform_gaussian_high_pass_filter(const Mat& img_orig, double d0) {
    cout << "高斯高通滤波, D0 = " << d0 << " ---" << endl;

    int opt_rows = getOptimalDFTSize(img_orig.rows);
    int opt_cols = getOptimalDFTSize(img_orig.cols);
    Mat img_padded;
    copyMakeBorder(img_orig, img_padded, 0, opt_rows - img_orig.rows, 0, opt_cols - img_orig.cols, BORDER_CONSTANT, Scalar::all(0));
    
    int rows = img_padded.rows;
    int cols = img_padded.cols;

    vector<vector<complex<double>>> fft_data(rows, vector<complex<double>>(cols));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            fft_data[i][j] = complex<double>((double)img_padded.at<uchar>(i, j), 0.0);
        }
    }

    fft_2d(fft_data, false);
    fftshift_2d(fft_data);

    //应用高斯高通滤波器 (创建模板并相乘)
    int center_row = rows / 2;
    int center_col = cols / 2;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double dx = i - center_row;
            double dy = j - center_col;
            double distance_sq = dx * dx + dy * dy;
            double h = 1.0 - exp(-distance_sq / (2 * d0 * d0));
            fft_data[i][j] *= h;
        }
    }

    fftshift_2d(fft_data);
    fft_2d(fft_data, true);

    Mat img_back(rows, cols, CV_64F);
    for (int i = 0; i < rows; ++i) for (int j = 0; j < cols; ++j) img_back.at<double>(i, j) = abs(fft_data[i][j]);
    normalize(img_back, img_back, 0, 255, NORM_MINMAX);
    img_back.convertTo(img_back, CV_8U);
    
    return img_back(Rect(0, 0, img_orig.cols, img_orig.rows));
}

Mat perform_butterworth_high_pass_filter(const Mat& img_orig, double d0, int n) {
    cout << "巴特沃斯高通滤波, D0 = " << d0 << ", n = " << n << " ---" << endl;
    int opt_rows = getOptimalDFTSize(img_orig.rows);
    int opt_cols = getOptimalDFTSize(img_orig.cols);
    Mat img_padded;
    copyMakeBorder(img_orig, img_padded, 0, opt_rows - img_orig.rows, 0, opt_cols - img_orig.cols, BORDER_CONSTANT, Scalar::all(0));
    
    int rows = img_padded.rows;
    int cols = img_padded.cols;

    vector<vector<complex<double>>> fft_data(rows, vector<complex<double>>(cols));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            fft_data[i][j] = complex<double>((double)img_padded.at<uchar>(i, j), 0.0);
        }
    }

    fft_2d(fft_data, false);
    fftshift_2d(fft_data);

    //应用巴特沃斯高通滤波器
    int center_row = rows / 2;
    int center_col = cols / 2;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double dx = i - center_row;
            double dy = j - center_col;
            double distance = sqrt(dx * dx + dy * dy);
            // 避免除以0
            if (distance == 0) distance = 1e-9;
            // 计算 H(u,v) = 1 / (1 + (D0 / D(u,v))^(2*n))
            double h = 1.0 / (1.0 + pow(d0 / distance, 2.0 * n));
            
            fft_data[i][j] *= h;
        }
    }

    fftshift_2d(fft_data);
    fft_2d(fft_data, true);

    Mat img_back(rows, cols, CV_64F);
    for (int i = 0; i < rows; ++i) for (int j = 0; j < cols; ++j) img_back.at<double>(i, j) = abs(fft_data[i][j]);
    normalize(img_back, img_back, 0, 255, NORM_MINMAX);
    img_back.convertTo(img_back, CV_8U);
    
    return img_back(Rect(0, 0, img_orig.cols, img_orig.rows));
}
int main() {
    //加载图像
    string image_path = "fft.jpg";
    Mat img_orig = imread(image_path, IMREAD_GRAYSCALE);
    if (img_orig.empty()) {
        cerr << "错误: 无法加载图像 " << image_path << endl;
        return -1;
    }

    //定义滤波器参数
    double d0 = 30.0; //截止频率
    int n = 2;        //巴特沃斯滤波器的阶数

    // --- 调用独立的滤波器函数 ---
    Mat ideal_result = perform_ideal_high_pass_filter(img_orig, d0);
    Mat gaussian_result = perform_gaussian_high_pass_filter(img_orig, d0);
    Mat butterworth_result = perform_butterworth_high_pass_filter(img_orig, d0, n);
    
    imshow("Original Image", img_orig);
    imshow("Ideal High Pass Result (D0=30)", ideal_result);
    imshow("Gaussian High Pass Result (D0=30)", gaussian_result);
    imshow("Butterworth High Pass Result (D0=30, n=2)", butterworth_result);
    
    waitKey(0);

    return 0;
}