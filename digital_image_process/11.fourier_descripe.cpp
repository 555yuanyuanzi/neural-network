//傅里叶描述子:fourier_descripe
//用目标边界曲线的傅里叶变换来描述目标区域的形状，将二维描述问题简化为一维描述问题

#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
 
void FourierShapeDescriptors(vector<Point>& contour, double ratio, vector<Point>& output_contour)
{
    //将轮廓变为复数图像
    Mat pointMat(contour.size(), 1, CV_32FC2);
    Vec2f* p = pointMat.ptr<Vec2f>();
    for (size_t i = 0; i < contour.size(); i++)
    {
        p[i][0] = (float)contour[i].x;
        p[i][1] = (float)contour[i].y;
    }
    //计算离散傅里叶变换的最优尺寸
    Mat paddedMat;
    int m = getOptimalDFTSize(pointMat.rows);
    int n = getOptimalDFTSize(pointMat.cols);
    copyMakeBorder(pointMat, paddedMat, 0, m- pointMat.rows, 0, n - pointMat.cols, BORDER_CONSTANT, Scalar::all(0));
    //中心化
    p = paddedMat.ptr<Vec2f>();
    for (size_t i = 0; i < m; i++)
    {
        if (i % 2 != 0){
            p[i][0] = - p[i][0];
            p[i][1] = - p[i][1];
        }
    }
    //傅里叶变换
    Mat dftMat;
    dft(paddedMat, dftMat, DFT_COMPLEX_INPUT | DFT_COMPLEX_OUTPUT);
     
    //保留重构边界的描述子
    int number = cvRound(m * ratio) & -2; // a & -2 代表最大不超过a的偶数
    int number_remove = (m - number) / 2;
    dftMat.rowRange(0, number_remove) = Scalar::all(0); //删除系数(设置为0)
    dftMat.rowRange(dftMat.rows - number_remove, dftMat.rows) = Scalar::all(0);
    //傅里叶反变换
    Mat idftMat;
    idft(dftMat, idftMat, DFT_COMPLEX_INPUT | DFT_COMPLEX_OUTPUT | DFT_SCALE);
    //去中心化
    p = idftMat.ptr<Vec2f>();
    for (size_t i = 0; i < m; i++)
    {
        if (i % 2 != 0) {
            p[i][0] = -p[i][0];
            p[i][1] = -p[i][1];
        }
    }
    //裁剪为原来大小
    Mat dst = idftMat.rowRange(0, contour.size()).clone();
    dst.convertTo(dst, CV_32SC2);
 
    //由图像Mat变回轮廓点vector<Point>
    Point point;
    for (size_t i = 0; i < contour.size(); i++)
    {
        point.x = dst.ptr<cv::Vec2i>(i)[0][0];
        point.y = dst.ptr<cv::Vec2i>(i)[0][1];
        output_contour.push_back(point);
    }
}
 
int main() {

    Mat src = imread("pic/fouriershape.tif", IMREAD_GRAYSCALE);
    if (src.empty()) {
        cerr << "错误: 无法加载图像!" << endl;
        return -1;
    }

    Mat binImg;
    threshold(src, binImg, 50, 255, THRESH_BINARY);

    vector<vector<Point>> contours;
    //findContours:在一张二-值图像中，找出所有物体的轮廓，并将每个轮廓表示为一个点的序列
    findContours(binImg, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

    if (contours.empty()) {
        cerr << "错误: 未找到任何轮廓!" << endl;
        return -1;
    }

    // 我们只处理最大的那个轮廓
    vector<Point> main_contour = contours[0];
    Mat result_display = Mat::zeros(src.size(), CV_8UC3); // 用彩色图来画不同颜色的轮廓


    // 绘制原始轮廓 (蓝色) 
    vector<vector<Point>> original_contours = { main_contour };
    drawContours(result_display, original_contours, 0, Scalar(255, 0, 0), 1); // Blue


    // 计算和绘制第一个逼近轮廓 (18个描述子)
    vector<Point> output_contour_18;
    //ratio控制形状平滑程度或逼近精度
    double ratio_18 = 0.0063;
    FourierShapeDescriptors(main_contour, ratio_18, output_contour_18);
    vector<vector<Point>> contours_18 = { output_contour_18 };
    drawContours(result_display, contours_18, 0, Scalar(0, 255, 0), 1); // Green
    
    // 计算和绘制第二个逼近轮廓 (8个描述子)
    vector<Point> output_contour_8;
    double ratio_8 = 0.0028;
    FourierShapeDescriptors(main_contour, ratio_8, output_contour_8);
    vector<vector<Point>> contours_8 = { output_contour_8 };
    drawContours(result_display, contours_8, 0, Scalar(0, 0, 255), 1); // Red


    imshow("Original Image", src);
    imshow("Fourier Descriptors Approximation", result_display);
 
    waitKey(0); 

    return 0;
}