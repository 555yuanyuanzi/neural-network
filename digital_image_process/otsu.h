// otsu.h
#pragma once 

#include <opencv2/opencv.hpp>

//手写实现大津法(Otsu's Method)来寻找最佳阈值
//param img_gray 输入的单通道8位灰度图 (CV_8U)
//return计算出的最佳阈值
int otsu_threshold(const cv::Mat& img_gray);