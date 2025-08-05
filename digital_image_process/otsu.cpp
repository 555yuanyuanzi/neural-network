//图像二值化:自动确定一个最佳阈值，将灰度图像转换为只有黑白两色的二值图像
//设定一个阈值T，将图像中灰度值大于T的像素设为白色（255），小于等于T的像素设为黑色（0）
//目的：简化图像，突出目标轮廓，是许多后续处理,文字识别OCR、物体检测的关键预处理步骤

//大津法:经典的自动阈值算法，核心思想是通过最大化类间方差来找到最佳的分割阈值

#include <iostream>
#include <vector>
#include <numeric> 
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

//大津法来寻找最佳阈值
int otsu_threshold(const Mat& img_gray) {
    if (img_gray.type() != CV_8UC1) {
        cout << "错误: 输入图像必须是8位单通道灰度图" << endl;
        return -1;
    }
    //计算图像的直方图
    //hist[i]存储灰度值为i的像素个数
    vector<int> hist(256, 0);
    for (int i = 0; i < img_gray.rows; ++i) {
        for (int j = 0; j < img_gray.cols; ++j) {
            hist[img_gray.at<uchar>(i, j)]++;
        }
    }
    
    //计算总像素数
    long total_pixels = img_gray.rows * img_gray.cols;
    
    //用于存储遍历过程中的最大类间方差和对应的最佳阈值
    double max_variance = 0.0;
    int best_threshold = 0;

    for (int t = 0; t < 256; ++t) {
        //对每个阈值t,计算类间方差
        long w0_count = 0; //背景像素总数
        long w1_count = 0; //前景像素总数
        double w0_sum = 0;       //背景像素灰度总和
        double w1_sum = 0;       //前景像素灰度总和

        //根据阈值t，将像素分为背景(<=t)和前景(>t)两组
        for (int i = 0; i < 256; ++i) {
            if (i <= t) {
                w0_count += hist[i];
                w0_sum += i * hist[i];
            } else {
                w1_count += hist[i];
                w1_sum += i * hist[i];
            }
        }

        if (w0_count == 0 || w1_count == 0) {
            continue; 
        }
        //计算两组的权重 w0, w1
        double w0 = (double)w0_count / total_pixels;
        double w1 = (double)w1_count / total_pixels;
        //计算两组的平均灰度值 u0, u1
        double u0 = w0_sum / w0_count;
        double u1 = w1_sum / w1_count;
        //计算类间方差 g
        double variance = w0 * w1 * pow(u0 - u1, 2);
        //更新最大方差和最佳阈值
        if (variance > max_variance) {
            max_variance = variance;
            best_threshold = t;
        }
    }
    return best_threshold;
}
