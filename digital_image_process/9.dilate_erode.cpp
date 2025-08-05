#include <iostream>
#include <vector>
#include <numeric>
#include <opencv2/opencv.hpp>

// 引入std和cv命名空间
using namespace std;
using namespace cv;

//BGR彩色图手动转换为灰度图
Mat bgr_to_gray(const Mat& img_bgr) {
    if (img_bgr.channels() != 3) {
        cerr << "错误: bgr_to_gray 函数需要一个3通道图像" << endl;
        return Mat();
    }
    Mat gray_img = Mat::zeros(img_bgr.size(), CV_8U);
    for (int i = 0; i < img_bgr.rows; ++i) {
        for (int j = 0; j < img_bgr.cols; ++j) {
            Vec3b pixel = img_bgr.at<Vec3b>(i, j); // Vec3b 用于访问BGR像素
            uchar b = pixel[0];
            uchar g = pixel[1];
            uchar r = pixel[2];
            // 应用加权平均公式
            gray_img.at<uchar>(i, j) = static_cast<uchar>(0.2126 * r + 0.7152 * g + 0.0722 * b);
        }
    }
    return gray_img;
}


//Otsu 二值化
Mat otsu_binarization(const Mat& img_gray) {
    // (此处省略了 otsu_threshold 函数的实现，假设它已存在并返回最佳阈值)
    // 为保持代码完整性，这里粘贴一份精简版
    vector<int> hist(256, 0);
    for (int i = 0; i < img_gray.rows; ++i){
        for (int j = 0; j < img_gray.cols; ++j){
            hist[img_gray.at<uchar>(i, j)]++;
        }
    }
    long total_pixels = img_gray.rows * img_gray.cols;
    double max_variance = 0.0;
    int best_threshold = 0;

    for (int t = 0; t < 256; ++t) {
        long w0_count = 0; double w0_sum = 0;
        long w1_count = 0; double w1_sum = 0;
        for (int i = 0; i <= t; ++i) { w0_count += hist[i]; w0_sum += i * hist[i]; }
        for (int i = t + 1; i < 256; ++i) { w1_count += hist[i]; w1_sum += i * hist[i]; }
        if (w0_count == 0 || w1_count == 0) continue;
        
        double w0 = (double)w0_count / total_pixels;
        double w1 = (double)w1_count / total_pixels;
        double u0 = w0_sum / w0_count;
        double u1 = w1_sum / w1_count;
        double variance = w0 * w1 * pow(u0 - u1, 2);
        
        if (variance > max_variance) {
            max_variance = variance;
            best_threshold = t;
        }
    }
    cout << "Otsu 阈值 >> " << best_threshold << endl;
    
    Mat binary_img;
    threshold(img_gray, binary_img, best_threshold, 255, THRESH_BINARY);
    return binary_img;
}


//形态学膨胀
Mat morphology_dilate(const Mat& binary_img, int dilate_times = 1) {
    //传入二值图像，膨胀次数
    if (binary_img.type() != CV_8UC1) {
        cerr << "错误: 膨胀操作只支持8位单通道图像" << endl;
        return Mat();
    }
    //定义十字形结构元，存储邻居相对于中心点的偏移量
    vector<Point> kernel = {Point(0, -1), Point(-1, 0), Point(1, 0), Point(0, 1)};
    Mat current_img = binary_img.clone();
    for (int t = 0; t < dilate_times; ++t) {
        Mat next_img = current_img.clone();
        //扫描像素
        for (int i = 0; i < current_img.rows; ++i) {
            for (int j = 0; j < current_img.cols; ++j) {
                // 如果当前像素是背景(0)，检查其邻域是否有前景(255)
                if (current_img.at<uchar>(i, j) == 0) {
                    for (const auto& offset : kernel) {
                        int ni = i + offset.y;
                        int nj = j + offset.x;
                        // 检查邻域点是否在图像内且为前景
                        if (ni >= 0 && ni < current_img.rows && nj >= 0 && nj < current_img.cols &&
                            current_img.at<uchar>(ni, nj) == 255) {
                            next_img.at<uchar>(i, j) = 255; // 将当前背景点变为前景
                            break; // 只要有一个邻居是前景，就立即膨胀
                        }
                    }
                }
            }
        }
        current_img = next_img;
    }
    return current_img;
}


//形态学腐蚀
Mat morphology_erode(const Mat& binary_img, int erode_times = 1) {
    if (binary_img.type() != CV_8UC1) {
        cerr << "错误: 腐蚀操作只支持8位单通道图像" << endl;
        return Mat();
    }
    
    // 定义十字形结构元
    vector<Point> kernel = {Point(0, 0), Point(0, -1), Point(-1, 0), Point(1, 0), Point(0, 1)};

    Mat current_img = binary_img.clone();

    for (int t = 0; t < erode_times; ++t) {
        Mat next_img = current_img.clone();
        for (int i = 0; i < current_img.rows; ++i) {
            for (int j = 0; j < current_img.cols; ++j) {
                // 如果当前像素是前景(255)，检查结构元是否完全覆盖在前景上
                if (current_img.at<uchar>(i, j) == 255) {
                    bool should_erode = false;
                    for (const auto& offset : kernel) {
                        int ni = i + offset.y;
                        int nj = j + offset.x;
                        // 检查邻域点是否越界或为背景
                        if (!(ni >= 0 && ni < current_img.rows && nj >= 0 && nj < current_img.cols &&
                              current_img.at<uchar>(ni, nj) == 255)) {
                            should_erode = true; // 只要有一个点不满足，就需要腐蚀
                            break;
                        }
                    }
                    if (should_erode) {
                        next_img.at<uchar>(i, j) = 0; // 将当前前景点变为背景
                    }
                }
            }
        }
        current_img = next_img;
    }
    return current_img;
}

int main() {
    // 加载图像
    string image_path = "pic/test.png"; // 替换为你的图片路径
    Mat img_orig = imread(image_path, IMREAD_COLOR);
    if (img_orig.empty()) {
        cerr << "错误: 无法加载图像 " << image_path << endl;
        return -1;
    }
    Mat gray_img = bgr_to_gray(img_orig);
    Mat otsu_img = otsu_binarization(gray_img);
    Mat dilate_result = morphology_dilate(otsu_img, 1);
    Mat erode_result = morphology_erode(otsu_img, 1);

    imshow("Original Image", img_orig);
    imshow("Otsu Binarization", otsu_img);
    imshow("Dilate Result", dilate_result);
    imshow("Erode Result", erode_result);

    waitKey(0);
    return 0;
}