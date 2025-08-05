//分水岭算法：
//一种基于拓扑理论的数学形态学的分割方法，基本思想是把图像看作测地学上的拓扑地貌，
//将像素点的灰度值视为海拔高度，整个图像就像一张高低起伏的地形图。
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;
int main() {
    Mat src;
    Mat img = imread("pic/water.tif");
    cvtColor(img, src, COLOR_BGR2GRAY);
    imshow("src", src);
    // 阈值处理
    Mat thresh;
    threshold(src, thresh, 0, 255, THRESH_OTSU);
    // 生成确定背景区域
    Mat background;
    Mat ele = getStructuringElement(MORPH_RECT, Size(3, 3));
    dilate(thresh, background, ele, cv::Point(-1, -1), 2);
    bitwise_not(background, background);
 
    // 生成确定前景区域，并利用连通区域标记
    Mat foreground;
    morphologyEx(thresh, foreground, MORPH_OPEN, ele, cv::Point(-1, -1), 2);
    //connectedComponents将包含多个分离物体的二值图像，转换成一张有序的、每个物体都有唯一身份ID的标签图
    int n = connectedComponents(foreground, foreground, 8, CV_32S);// 此时确定前景大于0，其余为0
     
    // 生成标记图
    Mat markers = foreground;
    markers.setTo(255, background);// 将确定背景设为255，其余为0的不动，即为unkown
    Mat markers8u;
    markers.convertTo(markers8u, CV_8U, 10);//灰度值*10使得差异变得明显
    imshow("Markers（输入）", markers8u);
     
    // 分水岭算法标注目标的轮廓,函数watershed实现基于标记的分水岭算法
    watershed(img, markers);// 轮廓由-1表示
    markers.convertTo(markers8u, CV_8U, 10);//灰度值*10使得差异变得明显
    imshow("Markers（输出）", markers8u);
 
    // 后处理（颜色填充）
    Mat mark;
    markers.convertTo(mark, CV_8U);//转换后-1变成0
    bitwise_not(mark, mark);
    vector<Vec3b> colors;
    for (size_t i = 0; i < n; i++)
    {
        int b = theRNG().uniform(0, 256);
        int g = theRNG().uniform(0, 256);
        int r = theRNG().uniform(0, 256);
        colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
    }
    Mat dst = Mat::zeros(markers.size(), CV_8UC3);
    for (int i = 0; i < markers.rows; i++)
    {
        for (int j = 0; j < markers.cols; j++)
        {
            int index = markers.at<int>(i, j);
            if (index > 0 && index <= static_cast<int>(n))
                dst.at<Vec3b>(i, j) = colors[index - 1];
        }
    }
    imshow("dst", dst);
    waitKey(0);
    return 0;
}