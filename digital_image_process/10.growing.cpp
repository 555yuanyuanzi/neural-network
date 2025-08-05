#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

void regionGrow(Mat& src, Mat& mask, vector<Point>& seeds, int thresh, Mat& dst){
    dst = Mat::zeros(src.size(), CV_8UC1);
   
    while(seeds.size()>0){
        Point currentPoint = seeds.back();
       
        dst.at<uchar>(currentPoint.y, currentPoint.x) = 255; 
        seeds.pop_back(); 
        
        Point dps[] = { Point(-1,-1), Point(0,-1), Point(1,-1), Point(1,0), Point(1,1), Point(0,1), Point(-1,1), Point(-1,0) };
        for (size_t i = 0; i < 8; i++)
        {
            Point tmpPoint = currentPoint + dps[i];
            if (!tmpPoint.inside(Rect(0, 0, src.cols, src.rows))) continue;
            
            int gray_diff = src.at<uchar>(currentPoint.y, currentPoint.x) - src.at<uchar>(tmpPoint.y, tmpPoint.x);
            
            if (abs(gray_diff) < thresh && dst.at<uchar>(tmpPoint.y, tmpPoint.x) == 0 && mask.at<uchar>(tmpPoint.y, tmpPoint.x) == 255)
            {
                
                dst.at<uchar>(tmpPoint.y, tmpPoint.x) = 255;
                seeds.push_back(tmpPoint);
            }
        }
    }
}
 
int main() {
    Mat src = imread("pic/defective_weld.tif", 0);
    if (src.empty()) {
        cerr << "错误：无法加载图像" << endl;
        return -1;
    }

    Mat mask, dst;
    
    //自动寻找种子点
    Mat binImg;
    threshold(src, binImg, 254, 255, THRESH_BINARY);
    Mat labels, stats, centroids;
    //connectedComponentsWithStats:在一张二值图像中，自动地找出所有独立的、相互连接的白色区域
    //并为它们编号，同时计算出每个区域的详细统计信息。
    int nccomps = connectedComponentsWithStats(binImg, labels, stats, centroids);
    vector<Point> seeds;
    //从1开始，跳过背景
    for (int i = 1; i < nccomps; i++)
    {
        Point p(cvRound(centroids.at<double>(i,0)), cvRound(centroids.at<double>(i, 1)));
        //原始的筛选条件
        if (src.at<uchar>(p.y, p.x) > 200) {
            seeds.push_back(p);
        }
    }

    //创建生长掩码
    threshold(src, mask, 190, 255, THRESH_BINARY);

    int similarity_thresh = 50; // 原始的阈值
    regionGrow(src, mask, seeds, similarity_thresh, dst);

    cout << "处理完成。正在显示结果" << endl;

    Mat seeds_display;
    cvtColor(src, seeds_display, COLOR_GRAY2BGR);
    for (int i = 1; i < nccomps; i++) {
        Point p(cvRound(centroids.at<double>(i,0)), cvRound(centroids.at<double>(i, 1)));
        if (src.at<uchar>(p.y, p.x) > 200) {
            circle(seeds_display, p, 3, Scalar(0, 255, 0), -1); // 绿色实心圆
        }
    }

    imshow("1. Original Image", src);
    imshow("2. Seeds Found", seeds_display);
    imshow("3. Growing Mask", mask);
    imshow("4. Region Grow Result", dst); 

    waitKey(0);
    destroyAllWindows();
    
    return 0;
}