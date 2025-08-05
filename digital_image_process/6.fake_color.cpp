#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main() {
    //加载图像并强制转为灰度图
    string image_path = "pic/baboon.jpg";
    Mat img_gray = imread(image_path, IMREAD_GRAYSCALE);

    if (img_gray.empty()) {
        cerr << "错误: 无法加载图像 " << image_path << endl;
        return -1;
    }

    //调整输入图像尺寸，使其更适合拼接
    Mat img_resized;
    resize(img_gray, img_resized, Size(200, 200));

    //定义拼接后的大图尺寸和布局
    int tile_width = 200;
    int tile_height = 200;
    int grid_cols = 4; //列数
    int grid_rows = 3; //行数

    //创建一个大的空白彩色图像作为画布（构造函数传入，高度，宽度，类型，BGR三通道）
    Mat result_canvas(tile_height * grid_rows, tile_width * grid_cols, CV_8UC3, Scalar(0, 0, 0));

    //循环遍历所有12种预设的Colormap
    for (int i = 0; i < grid_rows; ++i) {
        for (int j = 0; j < grid_cols; ++j) {
            int colormap_id = i * grid_cols + j;
            if (colormap_id > 11) {
                break;
            }
            cout << "应用颜色映射ID: " << colormap_id << endl;

            //应用伪彩色变换
            Mat img_colored;
            applyColorMap(img_resized, img_colored, colormap_id);

            //将处理后的小图拼接到大画布的正确位置
            Rect roi(j * tile_width, i * tile_height, tile_width, tile_height);
            img_colored.copyTo(result_canvas(roi));
        }
    }

    //显示最终的拼接结果
    imshow("concat result", result_canvas);
    waitKey(0);

    return 0;
}