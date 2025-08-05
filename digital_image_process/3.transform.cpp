//灰度变换算法
//图像反转: s = L - 1 - r (对于0-255范围，L=256，即 s = 255 - r)。将亮的变暗，暗的变亮。
//对数变换: s = c * log(1 + r)。拉伸暗部区域的动态范围，压缩亮部区域。主要用于增强图像暗部的细节。
//幂律(伽马)变换: s = c * r^γ
//γ < 1: 提升暗部，图像整体变亮。
//γ > 1: 提升亮部，图像整体变暗。

//8U:8伪无符号整型 C1：单通道
#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

//对输入的8位灰度图进行反转变换
Mat inverse_transform(const Mat& img_orig) {
    cout << "开始执行图像反转" << endl;
    //检查输入图像是否为8位单通道
    if (img_orig.type() != CV_8UC1) {
        cerr << "错误: 图像反转函数仅支持8位单通道灰度图" << endl;
        return Mat(); // 返回一个空Mat
    }
    //创建一个与原图同样大小和类型的新Mat来存储结果
    Mat result_img = Mat::zeros(img_orig.size(), img_orig.type());

    int rows = img_orig.rows;
    int cols = img_orig.cols;

    //遍历每一个像素
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            //获取原始像素值 r
            //uchar 在OpenCV中被广泛用来表示8位图像的像素值
            uchar r = img_orig.at<uchar>(i, j);
            // 应用变换公式 s = 255 - r
            result_img.at<uchar>(i, j) = 255 - r;
            //访问并返回位于 (i, j) 位置的、类型为 uchar 的像素值
        }
    }
    return result_img;
}
//对输入的8位灰度图进行对数变换
Mat log_transform(const Mat& img_orig, double c) {
    cout << "开始执行对数变换, c = " << c << endl;
    if (img_orig.type() != CV_8UC1) {
        cerr << "错误: 对数变换函数只支持8位单通道灰度图" << endl;
        return Mat();
    }
    Mat temp_img = Mat::zeros(img_orig.size(), CV_64F);

    int rows = img_orig.rows;
    int cols = img_orig.cols;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double r = (double)img_orig.at<uchar>(i, j);
            //s = c * log(1 + r)
            temp_img.at<double>(i, j) = c * log(1.0 + r);
        }
    }
    
    //归一化到0-255范围,最小-最大值归一化法。
    normalize(temp_img, temp_img, 0, 255, NORM_MINMAX);
    
    //将归一化后的浮点数Mat转换为8位无符号整型Mat
    Mat result_img;
    //将Mat数据类型转换为另一种指定的类型
    temp_img.convertTo(result_img, CV_8U);
    
    return result_img;
}

//对输入的8位灰度图进行伽马变换
Mat gamma_transform(const Mat& img_orig, double c, double gamma) {
    cout << "开始执行伽马变换, c = " << c << ", gamma = " << gamma << endl;
    if (img_orig.type() != CV_8UC1) {
        cerr << "错误: 伽马变换函数只支持8位单通道灰度图" << endl;
        return Mat();
    }
    Mat temp_img = Mat::zeros(img_orig.size(), CV_64F);
    
    int rows = img_orig.rows;
    int cols = img_orig.cols;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double r = (double)img_orig.at<uchar>(i, j);
            //s = c * r^gamma
            temp_img.at<double>(i, j) = c* pow(r, gamma);
        }
    }
    normalize(temp_img, temp_img, 0, 255, NORM_MINMAX);
    Mat result_img;
    temp_img.convertTo(result_img, CV_8U);
    
    return result_img;
}


int main() {
    // 加载图像
    string path_inverse = "pic/inverse_trans.tif"; //反转
    string path_log = "pic/log_trans.tif"; //对数变换
    string path_gamma = "pic/gamma_trans.tif"; //伽马变换
    
    Mat inverse_orig = imread(path_inverse, IMREAD_GRAYSCALE);
    Mat log_orig = imread(path_log, IMREAD_GRAYSCALE);
    Mat gamma_orig = imread(path_gamma, IMREAD_GRAYSCALE);
    
    if (inverse_orig.empty() || log_orig.empty() || gamma_orig.empty()) {
        cerr << "错误: 无法加载一张或多张图像" << endl;
        return -1;
    }

    Mat inverse_result = inverse_transform(inverse_orig);
    Mat log_result = log_transform(log_orig, 1.0);
    Mat gamma_result_brighten = gamma_transform(gamma_orig, 1.0, 0.4); //gamma < 1, 变亮
    Mat gamma_result_darken = gamma_transform(gamma_orig, 1.0, 4.0);  //gamma > 1, 变暗
    
    //显示结果
    imshow("Original for Inverse", inverse_orig);
    imshow("Inverse Transform Result", inverse_result);
    
    imshow("Original for Log", log_orig);
    imshow("Log Transform Result", log_result);

    imshow("Original for Gamma", gamma_orig);
    imshow("Gamma Transform (gamma=0.4, Brighten)", gamma_result_brighten);
    imshow("Gamma Transform (gamma=4.0, Darken)", gamma_result_darken);
    
    waitKey(0);

    return 0;
}