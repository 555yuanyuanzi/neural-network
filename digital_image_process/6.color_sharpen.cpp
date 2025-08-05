//彩色图像锐化
//拉普拉斯锐化:利用二阶微分来突出图像中的细节（边缘、线条等）
//然后将这些细节叠加回原始图像，从而使图像看起来更清晰

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

Mat sharpen_in_rgb_space(const Mat& src_img) {
    cout << "在RGB空间中锐化" << endl;

    if (src_img.channels() != 3) {
        cerr << "错误: 输入图像必须是3通道彩色图" << endl;
        return Mat();
    }

    vector<Mat> bgr_channels;
    split(src_img, bgr_channels);

    // 对每个通道独立地进行拉普拉斯滤波
    for (int i = 0; i < 3; ++i) {
        Mat laplacian_result;
        // 拉普拉斯结果可能有负值，需要用更高位深度的类型(CV_16S)来存储
        Laplacian(bgr_channels[i], laplacian_result, CV_16S);
        
        // 将结果安全地转换回CV_8U，以便进行减法运算
        Mat laplacian_8u;
        convertScaleAbs(laplacian_result, laplacian_8u);

        // 将拉普拉斯结果从原始通道中减去，以实现锐化
        // 锐化图像 = 原始图像 - 拉普拉斯结果
        subtract(bgr_channels[i], laplacian_8u, bgr_channels[i]);
    }
    
    // 将处理后的三个通道合并回一个BGR图像
    Mat result_img;
    merge(bgr_channels, result_img);

    return result_img;
}

Mat sharpen_in_hsv_space(const Mat& src_img) {
    cout << "在HSV空间中锐化 " << endl;

    if (src_img.channels() != 3) {
        cerr << "错误: 输入图像必须是3通道彩色图" << endl;
        return Mat();
    }

    // 将 BGR 图像转换到 HSV 颜色空间
    Mat hsv_img;
    cvtColor(src_img, hsv_img, COLOR_BGR2HSV);

    // 将HSV图像分离成三个通道
    vector<Mat> hsv_channels;
    split(hsv_img, hsv_channels);
    Mat& value_channel = hsv_channels[2]; // V - 亮度通道

    // 只对亮度(V)通道进行拉普拉斯滤波和锐化
    Mat laplacian_result;
    Laplacian(value_channel, laplacian_result, CV_16S);
    Mat laplacian_8u;
    convertScaleAbs(laplacian_result, laplacian_8u);
    subtract(value_channel, laplacian_8u, value_channel);
    Mat processed_hsv_img;
    merge(hsv_channels, processed_hsv_img);

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

    Mat result_rgb = sharpen_in_rgb_space(img);
    Mat result_hsv = sharpen_in_hsv_space(img);

    Mat difference_image;
    absdiff(result_rgb, result_hsv, difference_image);
    Mat enhanced_difference_image;
    difference_image.convertTo(enhanced_difference_image, -1, 10, 0);

    imshow("Original Image", img);
    imshow("Sharpened in RGB Space", result_rgb);
    imshow("Sharpened in HSV Space", result_hsv);
    imshow("Difference between results (x10)", enhanced_difference_image);
    
    waitKey(0);
    destroyAllWindows();

    return 0;
}