//基于直方图的纹理描述子:texture_analysis
//对每张图的指定区域，计算出一组能够量化描述该区域纹理的统计学数字（均值、标准差、R、三阶矩）
#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

//对图像的指定区域(ROI)进行纹理分析，并打印统计特征
void analyze_texture_in_roi(const Mat& image, const Rect& roi_rect) {
    // 1. 从输入图像中提取ROI
    // 增加一个边界检查，确保ROI在图像内部
    if ((roi_rect & Rect(0, 0, image.cols, image.rows)) != roi_rect) {
        cerr << "错误: 定义的ROI超出了图像边界！" << endl;
        return;
    }
    Mat roi = image(roi_rect);

    // 计算ROI的灰度直方图
    vector<int> hist(256, 0);
    double sum_of_pixels = 0;
    for (int y = 0; y < roi.rows; ++y) {
        for (int x = 0; x < roi.cols; ++x) {
            uchar pixel_value = roi.at<uchar>(y, x);
            hist[pixel_value]++;
            sum_of_pixels += pixel_value;
        }
    }

    // 计算一阶矩 (均值) 
    Scalar mean_scalar = mean(roi);
    double m = mean_scalar[0]; // 对于单通道图像，取第一个分量

    // 计算高阶中心矩 (方差和三阶矩)
    double moment2 = 0; // 二阶中心矩 (方差)
    double moment3 = 0; // 三阶中心矩
    int totalPixels = roi.rows * roi.cols;

    
    // 遍历直方图来计算矩
    for (int i = 0; i < 256; i++) {
        if (hist[i] > 0) { // 只计算有像素的灰度级
            double p_i = (double)hist[i] / totalPixels; // 概率 P(i)
            moment2 += pow(i - m, 2) * p_i;
            moment3 += pow(i - m, 3) * p_i;
        }
    }

     // 计算标准差
    double std_dev = sqrt(moment2);
    double R_smoothness = 1.0 - 1.0 / (1.0 + moment2); // R平滑度度量
    // 计算归一化的三阶矩 (偏度)
    double skewness = 0.0;
    if (std_dev > 1e-6) {
        skewness = moment3 / pow(std_dev, 3);
    } 

    cout << "ROI (" << roi_rect.x << ", " << roi_rect.y << ") - "
         << roi_rect.width << "x" << roi_rect.height << endl;
    cout << " 均值 (Mean): " << m << endl;
    cout << " 标准差 (Std Dev): " << std_dev << endl;
    cout << " R (平滑度): " << R_smoothness << endl;
    cout << " 三阶矩 (偏度): " << skewness << endl;//反应纹理的灰度分布不对称性
    cout << endl;
}

int main() {
    string image_path = "pic/rough_texture.tif"; 
    Mat src = imread(image_path, IMREAD_GRAYSCALE);

    if (src.empty()) {
        cerr << "错误: 无法加载图像 " << image_path << endl;
        return -1;
    }
    vector<Rect> rois_to_analyze = {
        //从点(x , y) 开始，宽和高为x像素
        Rect(0, 0, 75, 75), 
        Rect(155, 155, 75, 75), 
        Rect(10, 10, 75, 75)
    };

    // 创建一个彩色图像副本，用于可视化ROI的位置
    Mat src_display;
    cvtColor(src, src_display, COLOR_GRAY2BGR);
    
    for (size_t i = 0; i < rois_to_analyze.size(); ++i) {
        const auto& r = rois_to_analyze[i];

        rectangle(src_display, r, Scalar(0, 255, 0), 2); 
        putText(src_display, "ROI " + to_string(i+1), Point(r.x, r.y - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
        
        analyze_texture_in_roi(src, r);
    }
    
    imshow("Image with ROIs", src_display);
    waitKey(0);
    destroyAllWindows();
    return 0;
}