//细化:去除冗余的前景像素，同时保持物体的基本形状和连通性，最终得到一个单像素宽度的“骨架”
//击中或击不中变换:每次迭代的核心操作,我们使用一组预定义的模板，描述安全删除的“边缘像素”的模式。
#include "otsu.h"
#include <iostream>
#include <vector>
using namespace std;
using namespace cv;

//使用击中或击不中变换进行图像细化
Mat perform_thinning(const Mat& binary_img) {
    cout << "图像细化" << endl;
    if (binary_img.type() != CV_8UC1) {
        cerr << "错误: 细化函数只支持8位单通道二值图" << endl;
        return Mat();
    }
    //定义8个击中或击不中模板
    //1: 必须是前景, -1: 必须是背景, 0: 不关心
    vector<Mat> hmt_kernels;
    hmt_kernels.push_back((Mat_<int>(3, 3) << -1, -1, -1,  0,  1,  0,  1,  1,  1));
    hmt_kernels.push_back((Mat_<int>(3, 3) <<  0, -1, -1,  1,  1, -1,  1,  1,  0));
    hmt_kernels.push_back((Mat_<int>(3, 3) <<  1,  0, -1,  1,  1, -1,  1,  0, -1));
    hmt_kernels.push_back((Mat_<int>(3, 3) <<  1,  1,  0,  1,  1, -1,  0, -1, -1));
    hmt_kernels.push_back((Mat_<int>(3, 3) <<  1,  1,  1,  0,  1,  0, -1, -1, -1));
    hmt_kernels.push_back((Mat_<int>(3, 3) <<  0,  1,  1, -1,  1,  1, -1, -1,  0));
    hmt_kernels.push_back((Mat_<int>(3, 3) << -1,  0,  1, -1,  1,  1, -1,  0,  1));
    hmt_kernels.push_back((Mat_<int>(3, 3) << -1, -1,  0, -1,  1,  1,  0,  1,  1));

    Mat current_img = binary_img.clone();
    int rows = current_img.rows;
    int cols = current_img.cols;

    //开始迭代，直到图像不再变化
    while (true) {
        Mat before_img = current_img.clone();
        vector<Point> points_to_delete; //用于标记待删除的点
        //依次使用8个模板进行扫描
        for (const auto& kernel : hmt_kernels) {
            //对每个像素点，检查其邻域是否匹配当前模板
            for (int i = 1; i < rows - 1; ++i) {
                for (int j = 1; j < cols - 1; ++j) {
                    //只对前景像素进行判断
                    if (current_img.at<uchar>(i, j) != 255) {
                        continue;
                    }
                    bool is_match = true;
                    //遍历3x3模板
                    for (int ki = 0; ki < 3; ++ki) {
                        for (int kj = 0; kj < 3; ++kj) {
                            int kernel_val = kernel.at<int>(ki, kj);
                            if (kernel_val == 0) { // 不关心
                                continue;
                            }
                            
                            uchar pixel_val = current_img.at<uchar>(i + ki - 1, j + kj - 1);
                            
                            //检查匹配条件
                            if (kernel_val == 1 && pixel_val == 0) { //模板要求前景，但图像是背景
                                is_match = false;
                                break;
                            }
                            if (kernel_val == -1 && pixel_val == 255) { //模板要求背景，但图像是前景
                                is_match = false;
                                break;
                            }
                        }
                        if (!is_match) break;
                    }
                    //如果完全匹配，则标记该点
                    if (is_match) {
                        points_to_delete.push_back(Point(j, i));
                    }
                }
            }
        } 

        //一次性删除所有被标记的点
        if (points_to_delete.empty()) {
            break;
        } else {
            for (const auto& pt : points_to_delete) {
                current_img.at<uchar>(pt.y, pt.x) = 0;
            }
        }
    }

    return current_img;
}

int main() {
    string image_path = "pic/car_license.png"; 
    Mat img_orig = imread(image_path, IMREAD_GRAYSCALE);
    //IMREAD_GRAYSCALE返回单通道的灰度图像
    if (img_orig.empty()) {
        cerr << "错误: 无法加载图像 " << image_path << endl;
        return -1;
    }

    cout << "Otsu算法计算阈值" << endl;
    int best_threshold = otsu_threshold(img_orig);

    if (best_threshold == -1) {
        cerr << "Otsu阈值计算失败，请检查输入图像。" << endl;
        return -1;
    }
    cout << "计算出的最佳阈值为: " << best_threshold << endl;
    //计算二值化图像
    Mat binary_img;
    threshold(img_orig, binary_img, best_threshold, 255, THRESH_BINARY);
    
    Mat thin_result = perform_thinning(binary_img);

    imshow("Original Image", img_orig);       // 显示原始图像
    imshow("Binary Image (Otsu)", binary_img); // 显示二值化后的图像
    imshow("Thinned Image", thin_result);      // 显示细化后的图像

    waitKey(0);
    return 0;
}