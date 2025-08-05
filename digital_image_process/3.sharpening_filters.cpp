//空间锐化滤波器：
//二阶微分拉普拉斯算子:突出图像中的细节和精细纹理，对孤立点和细线非常敏感
//一阶微分梯度算子中的sobel算子:检测并突出图像中特定方向（主要是水平和垂直）的边缘，并计算边缘的强度
#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
//卷积计算
void RealCOV_ZeroPadding(const Mat& src, Mat& dst, const Mat& mask) {
    int width = src.cols;
    int height = src.rows;
    int m_width = mask.cols;
    int m_height = mask.rows;
    int mask_center_h = m_height / 2;
    int mask_center_w = m_width / 2;

    dst = Mat::zeros(src.size(), CV_64F); // 目标Mat使用double保证精度

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            double value = 0.0;
            for (int n = 0; n < m_height; n++) {
                for (int m = 0; m < m_width; m++) {
                    int src_i = i + n - mask_center_h;
                    int src_j = j + m - mask_center_w;

                    if (src_i >= 0 && src_i < height && src_j >= 0 && src_j < width) {
                        value += (double)src.at<uchar>(src_i, src_j) * mask.at<double>(n, m);
                    }
                }
            }
            dst.at<double>(i, j) = value;
        }
    }
}

//拉普拉斯锐化
Mat laplacian_sharpen(const Mat& img_orig) {
    cout << "开始执行拉普拉斯锐化" << endl;
    if (img_orig.type() != CV_8UC1) {
        cerr << "错误: 拉普拉斯函数只支持8位单通道灰度图" << endl;
        return Mat();
    }
    
    // 定义拉普拉斯锐化模板 (中心为5)
    Mat mask = (Mat_<double>(3, 3) << 
                 0, -1,  0,
                -1,  5, -1,
                 0, -1,  0);
    
    Mat cov_result;
    RealCOV_ZeroPadding(img_orig, cov_result, mask);

    // 将结果安全地转换回8位图像
    // convertScaleAbs会取绝对值并缩放到0-255
    Mat result_img;
    convertScaleAbs(cov_result, result_img);

    return result_img;
}

//Sobel边缘检测
Mat sobel_edge(const Mat& img_orig) {
    cout << "开始执行Sobel边缘检测" << endl;
    if (img_orig.type() != CV_8UC1) {
        cerr << "错误: Sobel函数只支持8位单通道灰度图" << endl;
        return Mat();
    }
    //定义Sobel的Gx和Gy模板
    Mat mask_gx = (Mat_<double>(3, 3) << 
                   -1, 0, 1,
                   -2, 0, 2,
                   -1, 0, 1);

    Mat mask_gy = (Mat_<double>(3, 3) << 
                   -1, -2, -1,
                    0,  0,  0,
                    1,  2,  1);

    //分别计算x和y方向的梯度
    Mat gx_result, gy_result;
    RealCOV_ZeroPadding(img_orig, gx_result, mask_gx);
    RealCOV_ZeroPadding(img_orig, gy_result, mask_gy);

    //将Gx和Gy的结果转换为可显示的8位图
    Mat abs_gx, abs_gy;
    convertScaleAbs(gx_result, abs_gx);
    convertScaleAbs(gy_result, abs_gy);
    
    //显示x和y方向的梯度
    // imshow("Sobel X Gradient", abs_gx);
    // imshow("Sobel Y Gradient", abs_gy);

    // 组合梯度 G = |Gx| + |Gy|
    Mat result_img;
    addWeighted(abs_gx, 0.5, abs_gy, 0.5, 0, result_img);
    return result_img;
}


int main() {
    string image_path = "pic/fft.tif";
    Mat img_orig = imread(image_path, IMREAD_GRAYSCALE);
    
    if (img_orig.empty()) {
        cerr << "错误: 无法加载图像 " << image_path << endl;
        return -1;
    }
    
    Mat laplacian_result = laplacian_sharpen(img_orig);
    Mat sobel_result = sobel_edge(img_orig);
    
    imshow("Original Image", img_orig);
    imshow("Laplacian Sharpen Result", laplacian_result);
    imshow("Sobel Edge Detection Result", sobel_result);
    
    waitKey(0);

    return 0;
}