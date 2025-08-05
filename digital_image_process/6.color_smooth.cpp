//彩色图像平滑
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

//在RGB颜色空间中对每个通道独立进行均值平滑
Mat smooth_in_rgb_space(const Mat& src_img, int kernel_size) {
    cout << "在RGB空间中平滑" << endl;

    if (src_img.channels() != 3) {
        cerr << "错误: 输入图像必须是3通道彩色图" << endl;
        return Mat();
    }

    // 将BGR图像分离成三个独立的通道
    vector<Mat> bgr_channels;
    split(src_img, bgr_channels);
    Mat& blue_channel = bgr_channels[0];
    Mat& green_channel = bgr_channels[1];
    Mat& red_channel = bgr_channels[2];

    // 对每个通道独立地进行均值滤波
    // cv::blur 是均值滤波的函数
    blur(blue_channel, blue_channel, Size(kernel_size, kernel_size));
    blur(green_channel, green_channel, Size(kernel_size, kernel_size));
    blur(red_channel, red_channel, Size(kernel_size, kernel_size));
    
    // 将处理后的三个通道合并回一个BGR图像
    Mat result_img;
    merge(bgr_channels, result_img);

    return result_img;
}

//转换到HSI空间，只对亮度(I)分量进行平滑
Mat smooth_in_hsi_space(const Mat& src_img, int kernel_size) {
    cout << "--- 策略二: 在HSI空间中平滑 ---" << endl;

    if (src_img.channels() != 3) {
        cerr << "错误: 输入图像必须是3通道彩色图" << endl;
        return Mat();
    }

    // 将BGR图像转换到 HSV 颜色空间
    // H(色调), S(饱和度), V(亮度, Value)
    Mat hsv_img;
    cvtColor(src_img, hsv_img, COLOR_BGR2HSV);

    // 将HSV图像分离成三个通道
    vector<Mat> hsv_channels;
    split(hsv_img, hsv_channels);
    Mat& hue_channel = hsv_channels[0];         
    Mat& saturation_channel = hsv_channels[1];  
    Mat& value_channel = hsv_channels[2];     

    // 只对亮度(V)通道进行均值滤波
    blur(value_channel, value_channel, Size(kernel_size, kernel_size));

    // 保持 H 和 S 通道不变，将处理后的三个通道合并回来
    Mat processed_hsv_img;
    merge(hsv_channels, processed_hsv_img);

    // 将处理后的 HSV 图像转换回 BGR 空间以便显示
    Mat result_img;
    cvtColor(processed_hsv_img, result_img, COLOR_HSV2BGR);

    return result_img;
}


int main() {
    string image_path = "pic/color.tif";
    Mat img = imread(image_path, IMREAD_COLOR);

    if (img.empty()) {
        cerr << "错误: 无法加载图像 " << image_path << endl;
        return -1;
    }

    int kernel_size = 5; // 使用5x5的均值核
    Mat result_rgb = smooth_in_rgb_space(img, kernel_size);
    Mat result_hsi = smooth_in_hsi_space(img, kernel_size);
    //计算两种方法的差值
     Mat difference_image;
    absdiff(result_rgb, result_hsi, difference_image);

    imshow("Original Image", img);
    imshow("Smoothed in RGB Space", result_rgb);
    imshow("Smoothed in HSI/HSV Space", result_hsi);
    imshow("Difference Image", difference_image);

    waitKey(0);
    destroyAllWindows();

    return 0;
}