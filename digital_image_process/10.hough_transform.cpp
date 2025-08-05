//霍夫变换Hough Transform：
//图像中识别特定几何形状（如直线、圆、椭圆等）的特征提取技术
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

//标准霍夫直线检测与绘制
void detect_and_draw_hough_lines(const Mat& edges, const Mat& original_gray) {
    vector<Vec2f> lines;
    HoughLines(edges, lines, 1, CV_PI / 180, 118);

    Mat result_img;
    cvtColor(original_gray, result_img, COLOR_GRAY2BGR);

    cout << "  - 检测到的直线数量 (标准霍夫): " << lines.size() << endl;
    for (const auto& line : lines) {
        float rho = line[0], theta = line[1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        pt1.x = cvRound(x0 + 1000 * (-b));
        pt1.y = cvRound(y0 + 1000 * (a));
        pt2.x = cvRound(x0 - 1000 * (-b));
        pt2.y = cvRound(y0 - 1000 * (a));
        cv::line(result_img, pt1, pt2, Scalar(0, 0, 255), 1, LINE_AA);
    }
    imshow("Standard Hough Lines Result", result_img);
}

//概率霍夫直线检测与绘制
void detect_and_draw_hough_lines_p(const Mat& edges, const Mat& original_color) {
    vector<Vec4i> lines;
    HoughLinesP(edges, lines, 1, CV_PI / 180, 80, 200, 15);

    Mat result_img = original_color.clone();

    cout << "  - 检测到的线段数量 (概率霍夫): " << lines.size() << endl;
    for (const auto& l : lines) {
        line(result_img, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 255, 0), 2, LINE_AA);
    }
    imshow("Probabilistic Hough Lines Result", result_img);
}

//直线检测:标准和概率霍夫
void run_line_detection() {
    cout << "直线检测" << endl;
    string image_path = "pic/jianzhu.png";
    Mat img_color = imread(image_path, IMREAD_COLOR);
    if (img_color.empty()) {
        cerr << "  错误: 无法加载图像 " << image_path << endl;
        return;
    }

    //公共的预处理步骤
    Mat gray, blurred, edges;
    cvtColor(img_color, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, blurred, Size(3, 3), 0);
    Canny(blurred, edges, 50, 150, 3);
    
    //显示预处理结果
    imshow("Original Image for Lines", img_color);
    imshow("Edges for Lines", edges);

    //调用具体的检测与绘制函数
    detect_and_draw_hough_lines(edges, gray);
    detect_and_draw_hough_lines_p(edges, img_color);

    waitKey(0);
    destroyAllWindows();
}

//圆检测
void run_circle_detection() {
    cout << "圆检测" << endl;
    string image_path = "pic/eye.png";
    Mat img_color = imread(image_path, IMREAD_COLOR);
    if (img_color.empty()) {
        cerr << "  错误: 无法加载图像 " << image_path << endl;
        return;
    }

    //预处理
    Mat gray, blurred;
    cvtColor(img_color, gray, COLOR_BGR2GRAY);
    medianBlur(gray, blurred, 5);

    //检测与绘制
    vector<Vec3f> circles;
    HoughCircles(blurred, circles, HOUGH_GRADIENT, 1, gray.rows / 64.0, 200, 10, 5, 30);

    Mat result_img = img_color.clone();
    cout << "  - 检测到的圆数量: " << circles.size() << endl;
    for (const auto& c : circles) {
        Point center(cvRound(c[0]), cvRound(c[1]));
        int radius = cvRound(c[2]);
        circle(result_img, center, radius, Scalar(0, 255, 0), 2, LINE_AA);
        circle(result_img, center, 2, Scalar(0, 0, 255), 3, LINE_AA);
    }
    
    // 显示
    imshow("Original Image for Circles", img_color);
    imshow("Blurred for Circles", blurred);
    imshow("Hough Circles Result", result_img);
    
    waitKey(0);
    destroyAllWindows();
}

int main() {
    run_line_detection();
    run_circle_detection();
    return 0;
}