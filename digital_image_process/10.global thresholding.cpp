//全局阈值处理（迭代法）：region_growing.cpp
//使用一个单一的、全局的阈值 T 来对整幅图像进行分割
#include <iostream>
#include <vector>
#include <numeric> 
#include <cmath> 
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

//使用迭代法自动计算全局阈值
//输入的单通道8位灰度图
//return计算出的最佳全局阈值
int iterative_global_threshold(const Mat& img_gray) {
    if (img_gray.type() != CV_8UC1) {
        cerr << "错误: 输入图像必须是8位单通道灰度图" << endl;
        return -1;
    }

    //初始化一个阈值T
    //可以选择图像的平均灰度值作为初始值
    double current_T = mean(img_gray)[0];
    double previous_T = 0;
    double delta_T = 1.0; //阈值变化的收敛条件

    cout << "初始阈值 T = " << current_T << endl;

    //迭代更新阈值，直到T的变化足够小
    int iteration_count = 0;
    while (fabs(current_T - previous_T) > delta_T) {
        previous_T = current_T;
        iteration_count++;

        long sum_g1 = 0, count_g1 = 0; //G1组 (灰度 > T)
        long sum_g2 = 0, count_g2 = 0; //G2组 (灰度 <= T)

        //根据当前阈值T，将像素分为两组
        for (int i = 0; i < img_gray.rows; ++i) {
            for (int j = 0; j < img_gray.cols; ++j) {
                uchar pixel_value = img_gray.at<uchar>(i, j);
                if (pixel_value > current_T) {
                    sum_g1 += pixel_value;
                    count_g1++;
                } else {
                    sum_g2 += pixel_value;
                    count_g2++;
                }
            }
        }

        //计算两组的平均灰度值
        double mu1 = (count_g1 > 0) ? static_cast<double>(sum_g1) / count_g1 : 0;
        double mu2 = (count_g2 > 0) ? static_cast<double>(sum_g2) / count_g2 : 0;

        //更新阈值
        current_T = (mu1 + mu2) / 2.0;

        cout << "迭代 " << iteration_count << ": T = " << current_T << endl;
    }
    
    cout << "迭代结束，最终阈值 = " << static_cast<int>(current_T) << endl;
    return static_cast<int>(current_T);
}

int main() {
    string image_path = "pic/fingerprint.tif"; // 替换为适合全局阈值的图片
    Mat img = imread(image_path, IMREAD_GRAYSCALE);
    if (img.empty()) {
        cerr << "无法加载图像" << endl;
        return -1;
    }

    int best_threshold = iterative_global_threshold(img);
    
    Mat binary_img;
    if (best_threshold != -1) {
        threshold(img, binary_img, best_threshold, 255, THRESH_BINARY);
        imshow("Original Image", img);
        imshow("Iterative Threshold Result", binary_img);
        waitKey(0);
    }
    
    return 0;
}