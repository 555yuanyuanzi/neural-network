//彩色分割聚类算法
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

Mat preprocess_image(const Mat& image) {
    // 将图像转换为浮点型，并进行归一化到[0.0, 1.0]范围
    // image.convertTo(destination, type, scale_factor)
    Mat normalized_image;
    image.convertTo(normalized_image, CV_32F, 1.0 / 255.0);

    // 调整图像大小
    Mat resized_image;
    resize(normalized_image, resized_image, Size(500, 500));

    // 进行高斯模糊处理，以减少噪音
    Mat blurred_image;
    GaussianBlur(resized_image, blurred_image, Size(5, 5), 0);

    return blurred_image;
}

//使用K-means算法对图像进行颜色分割
Mat kmeans_segmentation(const Mat& image, int num_clusters) {
    // 将图像数据重塑为 (width*height) x 3 的矩阵
    // 每一行代表一个像素的BGR值
    Mat samples(image.rows * image.cols, 3, CV_32F);
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            // Vec3f 用于访问浮点型三通道像素
            samples.at<Vec3f>(y * image.cols + x, 0) = image.at<Vec3f>(y, x);
        }
    }

    Mat labels; // 用于存储每个样本的聚类索引
    Mat centers; // 用于存储聚类中心（即我们的k个代表色）
    
    // 算法终止条件：达到100次迭代或精度达到0.1
    TermCriteria criteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 100, 0.1);
    
    // 3. 运行K-means算法
    // kmeans(数据, k值, 标签, 终止条件, 尝试次数, 初始化方法, 中心点)
    kmeans(samples,            // 输入数据
           num_clusters,       // K值
           labels,             // 输出：每个样本的标签
           criteria,           // 终止条件
           10,                 // 尝试次数 (attempts)，算法会运行10次并返回最佳结果
           KMEANS_RANDOM_CENTERS, // 随机选择初始中心
           centers);           // 输出：聚类中心

    Mat segmented_image(image.size(), image.type());
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            //找到当前像素 (y,x) 对应的聚类索引
            int cluster_idx = labels.at<int>(y * image.cols + x, 0);
            segmented_image.at<Vec3f>(y, x) = centers.at<Vec3f>(cluster_idx, 0);
        }
    }
    return segmented_image;
}

int main() {
    string image_path = "pic/kmeans.tif";
    Mat image = imread(image_path, IMREAD_COLOR);
    if (image.empty()) {
        cerr << "错误: 无法加载图像 " << image_path << endl;
        return -1;
    }

    Mat processed_image = preprocess_image(image);

    int num_clusters = 10; // 设置聚类簇的数量
    Mat segmented_image_float = kmeans_segmentation(processed_image, num_clusters);

    //将结果转换回可显示的8位图像
    Mat segmented_image_uchar;
    segmented_image_float.convertTo(segmented_image_uchar, CV_8UC3, 255.0);

    imshow("Original Image", image);
    imshow("Segmented Image", segmented_image_uchar);
    
    waitKey(0);
    destroyAllWindows();
    return 0;
}
