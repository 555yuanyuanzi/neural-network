//小波变换:
//使用有限长度、会衰减的“小波 (wavelet)”作为基。通过对小波进行伸缩 (scaling) 和平移 (translation)，
//同时告诉你信号在哪个时间（位置）出现了哪种频率（尺度）的成分
//下文演示了，二维小波一级分解,二维小波多级分解,小波去噪三种方法
#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

//对一行数据执行一维哈尔小波变换
Mat haar_wavelet_transform_1d(const Mat& row_data) {
    int width = row_data.cols;
    Mat transformed_row = Mat::zeros(1, width, CV_32F);
    Mat averages = transformed_row(Rect(0, 0, width / 2, 1));
    Mat differences = transformed_row(Rect(width / 2, 0, width / 2, 1));
    for (int i = 0; i < width / 2; ++i) {
        float a = row_data.at<float>(0, 2 * i);
        float b = row_data.at<float>(0, 2 * i + 1);
        averages.at<float>(0, i) = (a + b) / sqrt(2.0);
        differences.at<float>(0, i) = (a - b) / sqrt(2.0);
    }
    return transformed_row;
}

//对输入图像执行一级二维哈尔小波变换
Mat perform_haar_dwt2_level1(const Mat& img_orig) {
    if (img_orig.type() != CV_32FC1) {
        cerr << "错误: DWT输入图像必须是32位浮点单通道 (CV_32FC1)" << endl;
        return Mat();
    }
    int rows = img_orig.rows;
    int cols = img_orig.cols;
    Mat transformed_img = Mat::zeros(rows, cols, CV_32FC1);

    for (int i = 0; i < rows; ++i) {
        Mat transformed_row = haar_wavelet_transform_1d(img_orig.row(i));
        transformed_row.copyTo(transformed_img.row(i));
    }
    Mat temp_transposed;
    transpose(transformed_img, temp_transposed);
    for (int i = 0; i < cols; ++i) {
        Mat transformed_row = haar_wavelet_transform_1d(temp_transposed.row(i));
        transformed_row.copyTo(temp_transposed.row(i));
    }
    transpose(temp_transposed, transformed_img);
    return transformed_img;
}

//对一行数据执行一维哈尔小波逆变换
Mat haar_wavelet_inverse_transform_1d(const Mat& transformed_row) {
    int width = transformed_row.cols;
    Mat original_row = Mat::zeros(1, width, CV_32F);
    Mat averages = transformed_row(Rect(0, 0, width / 2, 1));
    Mat differences = transformed_row(Rect(width / 2, 0, width / 2, 1));
    for (int i = 0; i < width / 2; ++i) {
        float avg = averages.at<float>(0, i);
        float diff = differences.at<float>(0, i);
        original_row.at<float>(0, 2 * i) = (avg + diff) / sqrt(2.0);
        original_row.at<float>(0, 2 * i + 1) = (avg - diff) / sqrt(2.0);
    }
    return original_row;
}

//对输入图像执行一级二维哈尔小波逆变换
Mat perform_haar_idwt2_level1(const Mat& dwt_img) {
    if (dwt_img.type() != CV_32FC1) {
        cerr << "错误: IDWT输入图像必须是32位浮点单通道 (CV_32FC1)" << endl;
        return Mat();
    }
    int rows = dwt_img.rows;
    int cols = dwt_img.cols;
    Mat reconstructed_img;

    Mat temp_transposed;
    transpose(dwt_img, temp_transposed);
    for (int i = 0; i < cols; ++i) {
        Mat reconstructed_row = haar_wavelet_inverse_transform_1d(temp_transposed.row(i));
        reconstructed_row.copyTo(temp_transposed.row(i));
    }
    transpose(temp_transposed, reconstructed_img);
    for (int i = 0; i < rows; ++i) {
        Mat reconstructed_row = haar_wavelet_inverse_transform_1d(reconstructed_img.row(i));
        reconstructed_row.copyTo(reconstructed_img.row(i));
    }
    return reconstructed_img;
}

//对小波系数进行软阈值处理
Mat soft_threshold(const Mat& coeff_matrix, double threshold) {
    Mat result = Mat::zeros(coeff_matrix.size(), coeff_matrix.type());
    for (int i = 0; i < coeff_matrix.rows; ++i) {
        for (int j = 0; j < coeff_matrix.cols; ++j) {
            float val = coeff_matrix.at<float>(i, j);
            float sign = (val > 0) ? 1.0f : -1.0f;
            float mag = abs(val);
            result.at<float>(i, j) = (mag > threshold) ? sign * (mag - threshold) : 0.0f;
        }
    }
    return result;
}

//一级二维小波变换
void demo_single_level_dwt() {
    string image_path = "pic/cat.png";
    Mat img = imread(image_path, IMREAD_GRAYSCALE);
    if (img.empty()) { cerr << "无法加载图像 " << image_path << endl; return; }

    Mat img_resized;
    resize(img, img_resized, Size(448, 448));
    Mat img_float;
    img_resized.convertTo(img_float, CV_32F);

    Mat dwt_result = perform_haar_dwt2_level1(img_float);
    if (dwt_result.empty()) return;

    //提取子带并独立归一化以便显示
    int h = dwt_result.rows / 2;
    int w = dwt_result.cols / 2;
    Mat cA = dwt_result(Rect(0, 0, w, h));
    Mat cH = dwt_result(Rect(w, 0, w, h));
    Mat cV = dwt_result(Rect(0, h, w, h));
    Mat cD = dwt_result(Rect(w, h, w, h));
    
    normalize(cA, cA, 0, 255, NORM_MINMAX);
    normalize(cH, cH, 0, 255, NORM_MINMAX);
    normalize(cV, cV, 0, 255, NORM_MINMAX);
    normalize(cD, cD, 0, 255, NORM_MINMAX);
    
    Mat top_row, bottom_row, final_display;
    hconcat(cA, cH, top_row);
    hconcat(cV, cD, bottom_row);
    vconcat(top_row, bottom_row, final_display);
    
    final_display.convertTo(final_display, CV_8U);

    imshow("Original (Resized)", img_resized);
    imshow("1-Level DWT Sub-bands", final_display);
    waitKey(0);
}

//多级二维小波变换
void demo_multi_level_dwt() {
    string image_path = "pic/cat.png";
    Mat img = imread(image_path, IMREAD_GRAYSCALE);
    if (img.empty()) { cerr << "无法加载图像 " << image_path << endl; return; }
    
    Mat img_resized;
    resize(img, img_resized, Size(448, 448));
    Mat img_float;
    img_resized.convertTo(img_float, CV_32F);

    //第一级分解
    Mat dwt_level1 = perform_haar_dwt2_level1(img_float);
    Mat cA1 = dwt_level1(Rect(0, 0, dwt_level1.cols / 2, dwt_level1.rows / 2));

    //第二级分解
    Mat dwt_level2 = perform_haar_dwt2_level1(cA1);
    
    int h1 = dwt_level1.rows / 2, w1 = dwt_level1.cols / 2;
    int h2 = dwt_level2.rows / 2, w2 = dwt_level2.cols / 2;

    Mat cA2, cH2, cV2, cD2, cH1, cV1, cD1;
    normalize(dwt_level2(Rect(0, 0, w2, h2)), cA2, 0, 255, NORM_MINMAX);
    normalize(dwt_level2(Rect(w2, 0, w2, h2)), cH2, 0, 255, NORM_MINMAX);
    normalize(dwt_level2(Rect(0, h2, w2, h2)), cV2, 0, 255, NORM_MINMAX);
    normalize(dwt_level2(Rect(w2, h2, w2, h2)), cD2, 0, 255, NORM_MINMAX);
    normalize(dwt_level1(Rect(w1, 0, w1, h1)), cH1, 0, 255, NORM_MINMAX);
    normalize(dwt_level1(Rect(0, h1, w1, h1)), cV1, 0, 255, NORM_MINMAX);
    normalize(dwt_level1(Rect(w1, h1, w1, h1)), cD1, 0, 255, NORM_MINMAX);

    Mat top_row_l2, bottom_row_l2, cA1_display;
    hconcat(cA2, cH2, top_row_l2);
    hconcat(cV2, cD2, bottom_row_l2);
    vconcat(top_row_l2, bottom_row_l2, cA1_display);

    Mat top_row_l1, bottom_row_l1, final_display;
    hconcat(cA1_display, cH1, top_row_l1);
    hconcat(cV1, cD1, bottom_row_l1);
    vconcat(top_row_l1, bottom_row_l1, final_display);
    
    final_display.convertTo(final_display, CV_8U);

    imshow("Original (Resized)", img_resized);
    imshow("2-Level DWT Decomposition", final_display);
    waitKey(0);
}

//小波去噪
void demo_wavelet_denoising() {
    string image_path = "pic/cat.png";
    Mat img_orig = imread(image_path, IMREAD_GRAYSCALE);
    if (img_orig.empty()) { cerr << "无法加载图像 " << image_path << endl; return; }
    
    //小波变换要求尺寸是2的幂，这里用256x256
    resize(img_orig, img_orig, Size(256, 256));
    
    Mat img_float;
    img_orig.convertTo(img_float, CV_32F);

    //添加高斯噪声
    Mat noise = Mat::zeros(img_float.size(), CV_32F);
    randn(noise, 0, 25.0); //均值为0，标准差为25的高斯噪声
    Mat noisy_img = img_float + noise;

    //分解
    Mat dwt_result = perform_haar_dwt2_level1(noisy_img);
    
    //提取子带
    int h = dwt_result.rows / 2, w = dwt_result.cols / 2;
    Mat cA = dwt_result(Rect(0, 0, w, h));
    Mat cH = dwt_result(Rect(w, 0, w, h));
    Mat cV = dwt_result(Rect(0, h, w, h));
    Mat cD = dwt_result(Rect(w, h, w, h));
    
    //对细节系数进行阈值处理
    double threshold_val = 40.0; //需要根据噪声水平调整
    Mat cH_thresh = soft_threshold(cH, threshold_val);
    Mat cV_thresh = soft_threshold(cV, threshold_val);
    Mat cD_thresh = soft_threshold(cD, threshold_val);

    //重新组合
    Mat denoised_dwt = Mat::zeros(dwt_result.size(), dwt_result.type());
    cA.copyTo(denoised_dwt(Rect(0, 0, w, h)));
    cH_thresh.copyTo(denoised_dwt(Rect(w, 0, w, h)));
    cV_thresh.copyTo(denoised_dwt(Rect(0, h, w, h)));
    cD_thresh.copyTo(denoised_dwt(Rect(w, h, w, h)));
    
    Mat denoised_img = perform_haar_idwt2_level1(denoised_dwt);

    Mat noisy_img_8u, denoised_img_8u;
    normalize(noisy_img, noisy_img_8u, 0, 255, NORM_MINMAX);
    noisy_img_8u.convertTo(noisy_img_8u, CV_8U);
    normalize(denoised_img, denoised_img_8u, 0, 255, NORM_MINMAX);
    denoised_img_8u.convertTo(denoised_img_8u, CV_8U);

    imshow("Original Image", img_orig);
    imshow("Noisy Image", noisy_img_8u);
    imshow("Denoised Image (Wavelet)", denoised_img_8u);
    waitKey(0);
}

int main() {
    cout << "一级小波变换演示" << endl;
    demo_single_level_dwt();
    destroyAllWindows();

    cout << "多级小波变换" << endl;
    demo_multi_level_dwt();
    destroyAllWindows();

    cout << "小波去噪" << endl;
    demo_wavelet_denoising();
    destroyAllWindows();

    return 0;
}