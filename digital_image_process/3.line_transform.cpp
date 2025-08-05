//灰度变换之分段线性变换:
//对比度拉伸: 将图像中实际存在的最低和最高灰度值，线性地拉伸到整个灰度范围（如0-255），以提高整体对比度。
//灰度级分层/切片: 突出图像中某个特定灰度范围内的像素，可以将其设为高亮，也可以保持其他区域不变。
//比特平面分层: 将8位图像看作8个1位的二值图像的叠加，分别显示每个比特位所承载的信息。
#include <iostream>
#include <vector>
#include <algorithm> 
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

//1.对比度拉伸
Mat contrast_stretching(const Mat& img_orig) {
    cout << "开始执行对比度拉伸" << endl;
    
    if (img_orig.type() != CV_8UC1) {
        cerr << "错误: 对比度拉伸函数只支持8位单通道灰度图" << endl;
        return Mat();
    }

    //找到图像中的最小和最大灰度值
    double min_val, max_val;
    cv::minMaxLoc(img_orig, &min_val, &max_val);
    uchar A = static_cast<uchar>(min_val);
    uchar B = static_cast<uchar>(max_val);
    cout << "原始灰度范围 (A, B): (" << (int)A << ", " << (int)B << ")" << endl;
    //如果A和B已经是0和255，则变换无意义，直接返回原图副本
    if (A == 0 && B == 255) {
        return img_orig.clone();
    }
    //创建一个与原图同样大小和类型的新Mat来存储结果
    Mat result_img = Mat::zeros(img_orig.size(), img_orig.type());

    int rows = img_orig.rows;
    int cols = img_orig.cols;

    //遍历每一个像素并应用线性拉伸公式
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            uchar r = img_orig.at<uchar>(i, j);
            //s = 255.0 / (B - A) * (r - A)
            //+0.5进行四舍五入
            uchar s = static_cast<uchar>(255.0 / (B - A) * (r - A) + 0.5);
            result_img.at<uchar>(i, j) = s;
        }
    }
    return result_img;
}


//2. 灰度级分层/切片

Mat gray_level_slicing(const Mat& img_orig, int lower_bound, int upper_bound, int highlight_val, bool keep_background) {
    cout << "开始执行灰度级分层, 范围[" << lower_bound << "," << upper_bound << "]" << endl;

    if (img_orig.type() != CV_8UC1) {
        cerr << "错误: 灰度级分层函数只支持8位单通道灰度图" << endl;
        return Mat();
    }

    Mat result_img = Mat::zeros(img_orig.size(), img_orig.type());
    int rows = img_orig.rows;
    int cols = img_orig.cols;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            uchar r = img_orig.at<uchar>(i, j);
            
            // 检查像素值是否在感兴趣的区间内
            if (r >= lower_bound && r <= upper_bound) {
                result_img.at<uchar>(i, j) = static_cast<uchar>(highlight_val);
            } else {
                if (keep_background) {
                    // 保留背景
                    result_img.at<uchar>(i, j) = r;
                } else {
                    // 将背景设为0 (或任何其他指定值)
                    result_img.at<uchar>(i, j) = 0;
                }
            }
        }
    }
    return result_img;
}


//3. 比特平面分层
Mat bit_slicing(const Mat& img_orig, int bit_plane) {
    cout << "开始提取第 " << bit_plane << " 比特平面" << endl;

    if (img_orig.type() != CV_8UC1) {
        cerr << "错误: 比特平面分层函数只支持8位单通道灰度图" << endl;
        return Mat();
    }
    if (bit_plane < 1 || bit_plane > 8) {
        cerr << "错误: 比特平面必须在1到8之间" << endl;
        return Mat();
    }

    Mat result_img = Mat::zeros(img_orig.size(), img_orig.type());
    int rows = img_orig.rows;
    int cols = img_orig.cols;
    
    // 创建一个掩码 (mask) 来提取特定的位
    // 比如要提取第3位 (bit_plane=3), 掩码是 2^(3-1) = 4, 二进制是 00000100
    uchar mask = 1 << (bit_plane - 1); 

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            uchar r = img_orig.at<uchar>(i, j);
            
            // 使用位与(&)运算检查该位是否为1
            if ((r & mask) != 0) {
                // 如果该位为1，则输出像素设为255 (高亮)
                result_img.at<uchar>(i, j) = 255;
            } else {
                // 如果该位为0，则输出像素设为0
                result_img.at<uchar>(i, j) = 0;
            }
        }
    }
    return result_img;
}


int main() {
    Mat img_contrast = imread("pic/img_contrast.png", IMREAD_GRAYSCALE); // 使用灰度图以简化
    Mat img_slicing = imread("pic/img_slicing.tif", IMREAD_GRAYSCALE);
    Mat img_bit_plane = imread("pic/img_bit_plane.tif", IMREAD_GRAYSCALE);

    if (img_contrast.empty() || img_slicing.empty() || img_bit_plane.empty()) {
        cerr << "错误: 无法加载一张或多张图像" << endl;
        return -1;
    }
    
    //对比度拉伸
    Mat contrast_result = contrast_stretching(img_contrast);
    imshow("Original for Contrast", img_contrast);
    imshow("Contrast Stretching Result", contrast_result);
    
    //灰度级分层
    //高亮感兴趣区域，保留背景
    Mat slicing_result_1 = gray_level_slicing(img_slicing, 190, 230, 255, true);
    //高亮感兴趣区域，背景置零
    Mat slicing_result_2 = gray_level_slicing(img_slicing, 190, 230, 255, false);
    imshow("Original for Slicing", img_slicing);
    imshow("Gray Slicing (Keep BG)", slicing_result_1);
    imshow("Gray Slicing (Zero BG)", slicing_result_2);

    //比特平面分层
    //提取最高位和最低位平面进行对比
    Mat bit_plane_8 = bit_slicing(img_bit_plane, 8); //最高有效位(MSB)
    Mat bit_plane_1 = bit_slicing(img_bit_plane, 1); //最低有效位(LSB)
    imshow("Original for Bit-Plane", img_bit_plane);
    imshow("Bit-Plane 8 (MSB)", bit_plane_8);
    imshow("Bit-Plane 1 (LSB)", bit_plane_1);
    
    waitKey(0);

    return 0;
}