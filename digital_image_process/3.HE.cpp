//直方图均衡化HE：
//将集中的灰度值拉伸，让它们均匀地分布在整个[0, 255]的范围内，从而增强图像的整体对比度
#include <iostream>
#include <vector>
#include <cmath> 
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

//全局直方图均衡化
Mat histogram_equal(const Mat& img_orig) {
    cout <<"执行全局直方图均衡化(HE)" << endl;

    if (img_orig.type() != CV_8UC1) {
        cerr << "错误: 直方图均衡化函数只支持8位单通道灰度图" << endl;
        return Mat(); // 返回一个空Mat
    }

    int rows = img_orig.rows;
    int cols = img_orig.cols;
    long total_pixels = rows * cols;
    //计算直方图
    vector<int> hist(256, 0); 
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            //获取原始像素值
            uchar pixel_value = img_orig.at<uchar>(i, j);
            hist[pixel_value]++;
        }
    }

    //计算累积分布函数 (CDF)
    // cdf[i]将存储灰度值小于等于 i 的像素总数
    vector<long> cdf(256, 0);
    cdf[0] = hist[0];
    for (int i = 1; i < 256; ++i) {
        //当前的累积值 = 上一个累积值 + 当前灰度级的像素数
        cdf[i] = cdf[i - 1] + hist[i];
    }

    //创建查找表 (LUT) 并进行映射
    //lut[i] 将存储原始灰度值 i 映射后的新灰度值
    vector<uchar> lut(256);
    double scale_factor = 255.0 / total_pixels; // (L-1) / (M*N)

    for (int i = 0; i < 256; ++i) {
        //s = round( (L-1)/(M*N) * cdf(r) )
        lut[i] = static_cast<uchar>(round(scale_factor * cdf[i]));
    }
    
    Mat result_img = Mat::zeros(img_orig.size(), img_orig.type());

    // 使用查找表进行像素值映射
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            uchar original_value = img_orig.at<uchar>(i, j);
            result_img.at<uchar>(i, j) = lut[original_value];
        }
    }
    return result_img;
}

//绘制直方图的可视化图像
Mat draw_histogram(const Mat& hist, Scalar color = Scalar(255, 255, 255), int hist_size = 256) {
    //Mat传入直方图数据，color为RGB颜色（此处为白色），hist_size为桶数
    //定义画布尺寸
    int hist_w = 512; //宽度
    int hist_h = 400; //高度
    int bin_w = round((double)hist_w / hist_size); //每个桶的像素宽度
    //创建黑色画布
    Mat hist_image(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
    //创建一个直方图数据的副本进行归一化，避免修改原始数据
    Mat hist_normalized;
    normalize(hist, hist_normalized, 0, hist_image.rows, NORM_MINMAX, -1, Mat());
    //绘制线条
    for (int i = 1; i < hist_size; i++) {
        line(hist_image,
             Point(bin_w * (i - 1), hist_h - round(hist_normalized.at<float>(i - 1))),
             Point(bin_w * (i), hist_h - round(hist_normalized.at<float>(i))), color, 2, 8, 0);
    }

    return hist_image;
}

int main() {
    // 加载图像
    string image_path = "pic/bottom_left.tif";
    Mat img_orig = imread(image_path, IMREAD_GRAYSCALE);
    
    if (img_orig.empty()) {
        cerr << "错误: 无法加载图像 " << image_path << endl;
        return -1;
    }

    Mat he_result = histogram_equal(img_orig);
    
    imshow("Original Image", img_orig);
    imshow("Histogram Equal Result", he_result);
    
    //显示直方图对比
    //准备直方图计算的参数
    int hist_size = 256;
    float range[] = { 0, 256 };
    const float* hist_range = { range };
    
    //计算原始图像和HE后图像的直方图
    Mat hist_orig, hist_he;
    calcHist(&img_orig, 1, 0, Mat(), hist_orig, 1, &hist_size, &hist_range);
    calcHist(&he_result, 1, 0, Mat(), hist_he, 1, &hist_size, &hist_range);

    Mat hist_image_orig = draw_histogram(hist_orig, Scalar(255, 0, 0)); // 蓝色
    Mat hist_image_he = draw_histogram(hist_he, Scalar(0, 255, 0));   // 绿色

    //显示绘制好的直方图图像
    imshow("Original Histogram (Blue)", hist_image_orig);
    imshow("Equalized Histogram (Green)", hist_image_he);
    
    waitKey(0);

    return 0;
}