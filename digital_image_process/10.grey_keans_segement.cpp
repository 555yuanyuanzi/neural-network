//灰度图像聚类算法分割
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

int main() {

    // 使用 imread 加载图像，第二个参数 0 表示以灰度模式加载
    Mat src = imread("pic/kmeans.tif", IMREAD_GRAYSCALE);

    // 检查图像是否成功加载
    if (src.empty()) {
        cerr << "错误: 无法加载图像 '14.tif'!" << endl;
        cerr << "请确保图片文件与程序在同一目录下，或提供正确路径。" << endl;
        return -1;
    }
    // 将二维图像矩阵重塑(reshape)为一个单行矩阵
    Mat dataPixels = src.reshape(1, src.rows * src.cols);

    dataPixels.convertTo(dataPixels, CV_32F);

    int numCluster = 3; // 将灰度值聚为3类
    Mat labels;         // 输出参数：每个像素属于哪个聚类的标签(索引)
    Mat centers;        // 输出参数：每个聚类的中心点（即代表性的灰度值）

    // 调用 kmeans 函数
    // dataPixels:     输入数据
    // numCluster:     K值
    // labels:         输出的标签矩阵
    // TermCriteria:   算法终止条件 (迭代10次或精度达到0.1)
    // 3:              尝试次数，算法会运行3次并返回最佳结果
    // KMEANS_PP_CENTERS: 使用K-means++方法来初始化中心点，通常比随机要好
    // centers:        输出的中心点矩阵
    kmeans(dataPixels, numCluster, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 0.1), 3, KMEANS_PP_CENTERS, centers);

    for (int i = 0; i < dataPixels.rows; i++) {
        int k = labels.at<int>(i, 0);
        float center_value = centers.at<float>(k, 0);
        dataPixels.at<float>(i, 0) = center_value;
    }
    dataPixels.convertTo(dataPixels, CV_8U);
    Mat dst = dataPixels.reshape(1, src.rows);

    imshow("Original Grayscale Image", src);
    imshow("K-means Segmented Image", dst);

    waitKey(0);
    destroyAllWindows();

    return 0;
}