//频率域的纹理分析
//对输入图像进行傅里叶变换，得到其频谱
//将频谱中心化，便于分析
//计算并提取两种重要的频谱特征
//径向分布 (Radial Distribution): 频谱能量是如何随着离中心的距离（半径）变化的
//角向分布 (Angular Distribution): 频谱能量是如何随着角度变化的

#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
//将傅里叶变换后得到的频谱进行象限交换，把位于四个角落的低频信息移动到图像中心
Mat fftshift_corrected(const Mat& mag_src) {
    Mat mag_shifted = mag_src.clone();
    int cx = mag_shifted.cols / 2;
    int cy = mag_shifted.rows / 2;

    Mat q1(mag_shifted, Rect(0, 0, cx, cy));
    Mat q2(mag_shifted, Rect(cx, 0, cx, cy));
    Mat q3(mag_shifted, Rect(0, cy, cx, cy));
    Mat q4(mag_shifted, Rect(cx, cy, cx, cy));

    Mat temp;
    q1.copyTo(temp);
    q4.copyTo(q1);
    temp.copyTo(q4);

    q2.copyTo(temp);
    q3.copyTo(q2);
    temp.copyTo(q3);
    
    return mag_shifted;
}


int main() {
    Mat src = imread("pic/ordered_matches.tif", IMREAD_GRAYSCALE);
    if (src.empty()) {
        cerr << "Error: Cannot load image!" << endl;
        return -1;
    }

    // 傅里叶变换和幅值谱计算
    Mat padded;
    int m = getOptimalDFTSize(src.rows);
    int n = getOptimalDFTSize(src.cols);
    copyMakeBorder(src, padded, 0, m - src.rows, 0, n - src.cols, BORDER_CONSTANT, Scalar::all(0));

    Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
    Mat complexI;
    merge(planes, 2, complexI);
    dft(complexI, complexI, DFT_COMPLEX_OUTPUT);
    split(complexI, planes);
    Mat magMat;
    magnitude(planes[0], planes[1], magMat);
    magMat += Scalar::all(1);
    log(magMat, magMat);
    
    // 中心化频谱
    Mat mag_shifted = fftshift_corrected(magMat);

    // 准备用于分析的可视化频谱图
    Mat mag_visual;
    normalize(mag_shifted, mag_visual, 0, 255, NORM_MINMAX);
    mag_visual.convertTo(mag_visual, CV_8U);
    
    // 提取频谱特征
    int max_radius = min(mag_visual.rows / 2, mag_visual.cols / 2);
    vector<long> sr(max_radius, 0); // 径向分布
    vector<long> st(180, 0);      // 角向分布

    for (int r = 1; r < max_radius; r++) {
        for (int deg = 0; deg < 360; deg++) { // 扫描整个圆
            double theta = deg * CV_PI / 180;
            int x = cvRound(mag_visual.cols / 2.0 + r * cos(theta));
            int y = cvRound(mag_visual.rows / 2.0 + r * sin(theta));
            if (x >= 0 && x < mag_visual.cols && y >= 0 && y < mag_visual.rows) {
                uchar pixel_val = mag_visual.at<uchar>(y, x);
                sr[r] += pixel_val;
                if (deg < 180) { // 避免重复计算
                    st[deg] += pixel_val;
                }
            }
        }
    }
    
    // 绘制分布曲线 ---
    Mat radial_plot(300, max_radius, CV_8UC3, Scalar(0,0,0));
    Mat angular_plot(300, 180, CV_8UC3, Scalar(0,0,0));
    // 归一化数据以便绘图
    long max_sr = *max_element(sr.begin(), sr.end());
    long max_st = *max_element(st.begin(), st.end());

    // 绘制径向分布
    for (int i = 1; i < max_radius; i++) {
        Point p1(i - 1, radial_plot.rows - (sr[i-1] * (radial_plot.rows - 10) / max_sr));
        Point p2(i, radial_plot.rows - (sr[i] * (radial_plot.rows - 10) / max_sr));
        line(radial_plot, p1, p2, Scalar(0, 255, 0), 1);
    }
    
    // 绘制角向分布
    for (int i = 1; i < 180; i++) {
        Point p1(i - 1, angular_plot.rows - (st[i-1] * (angular_plot.rows - 10) / max_st));
        Point p2(i, angular_plot.rows - (st[i] * (angular_plot.rows - 10) / max_st));
        line(angular_plot, p1, p2, Scalar(0, 255, 255), 1);
    }

    imshow("Original Image", src);
    imshow("Magnitude Spectrum (Shifted)", mag_visual);
    imshow("Radial Distribution", radial_plot);
    imshow("Angular Distribution", angular_plot);

    waitKey(0);
    return 0;
}