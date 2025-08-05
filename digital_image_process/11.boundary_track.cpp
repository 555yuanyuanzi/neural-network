//边界检测：
//摩尔邻域跟踪:用于从二值图像中提取单个连通区域的有序边界像素序列
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

//摩尔边界跟踪算法的实现，用于找到边界点的坐标
vector<Point> boundaryTrack(const Mat& binImg) {
    if (binImg.empty() || binImg.type() != CV_8U) {
        cerr << "错误: boundaryTrack需要一个非空的8位单通道二值图。" << endl;
        return {}; // 返回空向量
    }

    Mat image_to_track = binImg.clone(); // 创建一个副本以防万一
    
    // 找到起点
    // 获取指向图像数据起始位置的指针
    uchar* b = image_to_track.ptr<uchar>(0); 
    int istep = image_to_track.cols; // 每行的像素数（作为一维索引的步长）
    int n = 0;
    // 遍历所有像素（看作一维数组），找到第一个值为255的前景像素
    for (; n < image_to_track.rows * image_to_track.cols; n++) {
        if (*b == 255)
            break; // 找到了，跳出循环
        b++; // 移动指针到下一个像素
    }

    // 如果扫描完整个图像都没找到前景像素，则返回空
    if (n == image_to_track.rows * image_to_track.cols) {
        cout << "未在图像中找到前景像素。" << endl;
        return {};
    }

    // 初始化跟踪变量
    int index = n; // 当前点在一维数组中的索引
    int start_index = n; // 保存起点的索引，用于终止判断
    int end = -1;  // 终止标志，初始化为一个与起点不同的值
    vector<Point> boundaryPoints;

    // 添加一个迭代保护，防止因意外情况陷入死循环
    int max_iterations = image_to_track.rows * image_to_track.cols; 
    int iter_count = 0;

    // 边界跟踪循环
    while (end != start_index && iter_count < max_iterations) {
        // 将当前点的1D索引转换为2D坐标并记录
        boundaryPoints.push_back(Point(index % istep, index / istep));
        
        // 定义8个邻域的1D索引偏移量 (N, NE, E, SE, S, SW, W, NW)，并重复一遍
        int neighbor[16] = { -istep , -istep + 1, 1, istep + 1, istep, istep - 1, -1, -istep - 1,
                             -istep , -istep + 1, 1, istep + 1, istep, istep - 1, -1, -istep - 1};
        
        // 寻找下一个边界点
        int i = 0;
        // 从邻域中找到第一个背景点(0)，这标志着“墙壁”的位置
        for (; i < 8; i++) {
            if ((int)b[neighbor[i]] == 0)
                break;
        }

        int j = i;
        // 从“墙壁”开始，逆时针寻找第一个前景点(255)，这就是下一个边界点
        for (; j < i + 8; j++) {
            if ((int)b[neighbor[j]] == 255) {
                // 找到了，更新指针和索引到新位置
                b += neighbor[j];
                index += neighbor[j];
                
                // 更新终止检查变量
                end = index;
                break; // 跳出内层循环
            }
        }
        iter_count++;
    }
    
    if (iter_count >= max_iterations) {
        cout << "警告: 边界跟踪达到最大迭代次数。" << endl;
    }

    return boundaryPoints;
}


int main() {
    string image_path = "pic/square.tif"; 
    Mat img_orig = imread(image_path, IMREAD_COLOR);

    if (img_orig.empty()) {
        cerr << "错误: 无法加载图像 " << image_path << endl;
        return -1;
    }

    Mat gray_img, binary_img;
    cvtColor(img_orig, gray_img, COLOR_BGR2GRAY);
    // 使用Otsu方法自动二值化，得到实心的二值图像 (左图)
    threshold(gray_img, binary_img, 0, 255, THRESH_BINARY | THRESH_OTSU);
    

    // 获取边界点的坐标列表
    vector<Point> boundary_points = boundaryTrack(binary_img); 

    // 关键步骤：根据坐标列表创建边界图像
    // 创建一张和原图一样大小的全黑画布
    Mat boundary_image = Mat::zeros(binary_img.size(), CV_8U);

    if (!boundary_points.empty()) {
        cout << "找到边界，包含 " << boundary_points.size() << " 个点。" << endl;
        
        // 遍历所有找到的边界点坐标
        for (const auto& p : boundary_points) {
            // 在黑色画布上，将这些坐标对应的像素点设为白色
            boundary_image.at<uchar>(p) = 255;
        }
    } else {
        cout << "未能找到边界。" << endl;
    }

    imshow("src", binary_img); 
    imshow("mask", boundary_image); 
    
    waitKey(0);
    destroyAllWindows();
    
    return 0;
}