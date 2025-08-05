//特征提取之边界特征：
//形态学梯度法 = 膨胀(图像) - 腐蚀(图像)
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

// 引入std和cv命名空间
using namespace std;
using namespace cv;

//使用形态学梯度提取二值图像的边界
Mat extract_boundary_morphological(const Mat& binary_img, int kernel_size = 3) {
    if (binary_img.type() != CV_8U) {
        cerr << "错误: 输入图像必须是8位单通道二值图" << endl;
        return Mat();
    }

    // 创建一个结构元 (Kernel / Structuring Element)
    // MORPH_RECT 表示一个矩形的结构元
    Mat kernel = getStructuringElement(MORPH_RECT, Size(kernel_size, kernel_size));

    // 执行形态学梯度操作
    Mat boundary_img;
    morphologyEx(binary_img, boundary_img, MORPH_GRADIENT, kernel);

    return boundary_img;
}

// 实现形态学梯度
Mat extract_boundary_manual(const Mat& binary_img, int kernel_size = 3) {
    cout << "形态学梯度" << endl;
    Mat kernel = getStructuringElement(MORPH_RECT, Size(kernel_size, kernel_size));
    
    Mat dilated_img, eroded_img, boundary_img;
    
    // a. 膨胀
    dilate(binary_img, dilated_img, kernel);
    // b. 腐蚀
    erode(binary_img, eroded_img, kernel);
    // c. 相减
    subtract(dilated_img, eroded_img, boundary_img);
    
    return boundary_img;
}


int main() {
    // 加载图像
    string image_path = "pic/square.tif"; 
    Mat img_orig = imread(image_path, IMREAD_COLOR);

    if (img_orig.empty()) {
        cerr << "错误: 无法加载图像 " << image_path << endl;
        return -1;
    }

    // 预处理：转灰度 -> 二值化
    Mat gray_img, binary_img;
    cvtColor(img_orig, gray_img, COLOR_BGR2GRAY);
    threshold(gray_img, binary_img, 0, 255, THRESH_BINARY | THRESH_OTSU);
    
    // 如果二值图有噪点，可以先进行开/闭运算清理
    // Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    // morphologyEx(binary_img, binary_img, MORPH_CLOSE, kernel);
    // morphologyEx(binary_img, binary_img, MORPH_OPEN, kernel);

    // 调用形态学梯度函数提取边界
    Mat boundary_result = extract_boundary_morphological(binary_img, 3);

    imshow("Binary Image (src)", binary_img); 
    imshow("Boundary (mask)", boundary_result); 
    
    waitKey(0);
    destroyAllWindows();
    
    return 0;
}