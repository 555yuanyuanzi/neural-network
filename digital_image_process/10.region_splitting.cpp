#include<opencv2/opencv.hpp>
using namespace std;
using namespace cv;
 
void splitMerge(Mat& src, Mat& dst, int h0, int w0, int h, int w, int maxMean, int minStd, int cell) {
    Rect r = Rect(w0, h0, w, h) & Rect(0, 0, src.cols, src.rows);// 防止出界
    Mat win = src(r);
    Mat mean, stddev;
    meanStdDev(win, mean, stddev);// 计算均值和标准差
    if (mean.at<double>(0, 0) < maxMean && stddev.at<double>(0, 0) > minStd && h < 2 * cell && w<2 * cell)
        // 满足条件，判为目标区域，设为白色
        rectangle(dst, r, Scalar(255), -1);
    else // 不满足条件
        if(h>cell && w>cell){// 继续拆分
            splitMerge(src, dst, h0, w0, (h + 1) / 2, (w + 1) / 2, maxMean, minStd, cell);
            splitMerge(src, dst, h0+ (h + 1) / 2, w0, (h + 1) / 2, (w + 1) / 2, maxMean, minStd, cell);
            splitMerge(src, dst, h0, w0+ (w + 1) / 2, (h + 1) / 2, (w + 1) / 2, maxMean, minStd, cell);
            splitMerge(src, dst, h0 + (h + 1) / 2, w0 + (w + 1) / 2, (h + 1) / 2, (w + 1) / 2, maxMean, minStd, cell);
        }
}
int main() {
    Mat src = imread("pic/cygnusloop.tif", 0);   
    Mat dst32 = Mat::zeros(src.size(), CV_8UC1);
    Mat dst16 = Mat::zeros(src.size(), CV_8UC1);
    Mat dst8 = Mat::zeros(src.size(), CV_8UC1);
    // 均值上界和标准差下界，最小分割区域 cell=32, 16, 8
    int maxMean = 95; int minStd = 10;
    splitMerge(src, dst32, 0, 0, src.cols, src.rows, maxMean, minStd, 32);
    splitMerge(src, dst16, 0, 0, src.cols, src.rows, maxMean, minStd, 16);
    splitMerge(src, dst8, 0, 0, src.cols, src.rows, maxMean, minStd, 8);
 
    imshow("src", src);
    imshow("32x32", dst32);
    imshow("16x16", dst16);
    imshow("8x8", dst8);
    waitKey(0);
    return 0;
}