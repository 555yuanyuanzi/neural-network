//实现三种常见的空间平滑滤波算法：均值滤波、高斯滤波和中值滤波
#include<iostream>
#include<time.h>
#include<stdlib.h>
#include<math.h>
#include <algorithm>
#include<vector>
#include<opencv2/opencv.hpp>
#define GRAY_LEVEL 8//灰度级
#define WIDTH 10
#define HEIGHT 10//图像尺寸
#define M_WIDTH 3
#define M_HEIGHT 3//滤波器（模板）的尺寸
#define M_PI 3.14159265358979323846
using namespace std;

//二维卷积运算（边界零填充法）
void RealCOV(double *src, double *dst, double *mask,  int width, int height, int m_width, int m_height){
    //src dst指向源图像和目标图像的像素数据，msk指向滤波器模板的数据
    //计算滤波器模板对的中心点坐标
    int mask_center_h = m_height / 2 ;//横坐标
    int mask_center_w = m_width / 2 ;//纵坐标
    //遍历图像中的每一个像素
    for(int i = 0 ; i < height ; i++ ){
        for(int j = 0 ; j < width ; j++ ){//
            double value = 0.0 ;
            //遍历滤波器模板
            for(int n = 0 ; n < m_height ; n++ ){
                for(int m = 0 ; m < m_width ; m++ ){
                    //计算当前模板元素对应的源图像像素坐标
                    int src_i=i + (n-mask_center_h);
                    int src_j=j + (m-mask_center_w);

                    //零填充
                    if(src_i>= 0 && src_i < height && src_j>= 0 &&src_j < width){
                        value += src[src_i*width + src_j] * mask[n*m_width+m] ;  
                    }
                }
            }
            //存储输出图像的一维数组
            dst[i*width+j] = value ;
        }
    }
}

//均值滤波器模板生成
//生成全1的均值滤波器
void MeanMask(double *mask,int width,int height){
    double meanvalue = 1.0/(width * height) ;
    for(int i = 0 ; i < width * height ; i++ ){
        //用一维数组存储矩阵
        mask[i] = meanvalue ;
    }
}
//调用卷积函数实现滤波
void MeanFilter(double *src, double *dst, int width,int height,int m_width,int m_height){
    double *mask = new double[m_width * m_height] ;
    MeanMask(mask, m_width, m_height) ;
    RealCOV(src, dst, mask,  width, height, m_width, m_height) ; 
    delete[] mask;
}


//高斯滤波器模板生成
void GaussianMask(double *mask,int width,int height,double deta){
    //mask存放最终生成的高斯模板，width和height为生成的模板的尺寸，deta为标准差参数
    double deta_2 = deta * deta ;
    //计算模板的中心坐标
    double mask_center_h = (double)height / 2 - 0.5 ; 
    double mask_center_w = (double)width / 2 - 0.5 ;

    double param = 1.0 / (2 * M_PI * deta_2) ;

    double sum = 0.0 ;
    //遍历模板的每一个位置
    for(int i = 0 ; i < height ; i++){
        for(int j = 0 ; j < width ; j++){
            double distance_2 = pow(j-mask_center_w,2)+pow(i-mask_center_h,2);
            mask[i * width + j] = param * exp(-distance_2/(2*deta_2)) ;
            //高斯求和
            sum += mask[i * width + j] ;
        }
    }
    //对模板进行归一化，确保滤波操作不会改变图像的整体平均亮
    for(int i = 0 ; i < height * width ; i++){
        mask[i] /= sum ;
    }
}

void GaussianFilter(double *src, double *dst, int width, int height,int m_width,int m_height, double deta){
    double *mask = new double[width * height] ;
    GaussianMask(mask, m_width, m_height, deta) ;
    RealCOV(src, dst, mask,  width, height, m_width, m_height) ; 
    delete[] mask;
}


//中值滤波器
//用一个像素邻域内所有像素值的中值来代替该像素原来的值）
void MedianFilter(double *src, double *dst, int width, int height, int m_width, int m_height){
    //这里没有mask，只是模拟一个m_width和m_height的矩阵区域
    int mask_size = m_width * m_height;
    int mask_center_h = m_height / 2 ;
    int mask_center_w = m_width / 2 ;

    for(int i = 0 ; i < height ; i++ ){
        for(int j = 0 ; j < width ; j++ ){
            double value = 0.0 ;
            //边界忽略法
            if((i-mask_center_h)>=0 && (j-mask_center_w)>=0 &&(i+mask_center_h)<height && (j+mask_center_w)<width ){
                vector<double> sort_arr;
                sort_arr.reserve(mask_size);
            	for(int n = 0 ; n < m_height ; n++ ){
	                for(int m = 0 ; m < m_width ; m++ ){
						sort_arr.push_back(src[(i+n-mask_center_h)*width + (j+m-mask_center_w)] );
	                }
            	}
                //排序
                sort(sort_arr.begin(), sort_arr.end());
                value = sort_arr[mask_size/2] ;
			}
            dst[i*width+j] = value ;
            //零填充法
            //边界复制法
        }
    }
}
//零填充法
void MedianFilter_Zero(double *src, double *dst, int width, int height, int m_width, int m_height) {
    const int mask_size = m_width * m_height;
    const int mask_center_h = m_height / 2;
    const int mask_center_w = m_width / 2;

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            
            vector<double> sort_arr;
            sort_arr.reserve(mask_size);
            for (int n = 0; n < m_height; n++) {
                for (int m = 0; m < m_width; m++) {
                    int src_i = i + n - mask_center_h;
                    int src_j = j + m - mask_center_w;

                    if (src_i >= 0 && src_i < height && src_j >= 0 && src_j < width) {
                        sort_arr.push_back(src[src_i * width + src_j]);
                    } else {
                        sort_arr.push_back(0.0);
                    }
                }
            }
            sort(sort_arr.begin(), sort_arr.end());
            dst[i * width + j] = sort_arr[mask_size/2];
        }
    }
}
//边界复制法
void MedianFilter_ReplicateBorder(double *src, double *dst, int width, int height, int m_width, int m_height) {
    const int mask_size = m_width * m_height;
    const int mask_center_h = m_height / 2;
    const int mask_center_w = m_width / 2;

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            
            vector<double> sort_arr;
            sort_arr.reserve(mask_size);
            for (int n = 0; n < m_height; n++) {
                for (int m = 0; m < m_width; m++) {
                    int src_i = i + n - mask_center_h;
                    int src_j = j + m - mask_center_w;

                    //将可能越界的坐标“夹”回有效范围内
                    int clamped_i = max(0, min(src_i, height - 1));
                    int clamped_j = max(0, min(src_j, width - 1));
                    sort_arr.push_back(src[clamped_i * width + clamped_j]);
                }
            }
            sort(sort_arr.begin(), sort_arr.end());
            dst[i * width + j] = sort_arr[mask_size/2];
        }
    }
}

int main(){
    srand((unsigned)time(NULL));//初始化伪随机数生成器
    double *src = new double[WIDTH * HEIGHT];
    double *dst = new double[WIDTH * HEIGHT];
    
    cout << "原始图像" << endl;
    for(int i = 0; i < HEIGHT; i++){
        for(int j = 0; j < WIDTH; j++){
            src[i * WIDTH + j] = (int)rand() % GRAY_LEVEL;
            cout << src[i * WIDTH + j] << " ";
        }
        cout << endl;
    }
    cout << endl;

    cout << "均值滤波" << endl;
    MeanFilter(src, dst, WIDTH, HEIGHT, M_WIDTH, M_HEIGHT);
    for(int i = 0; i < HEIGHT; i++){
        for(int j = 0; j < WIDTH; j++){
            cout << dst[i * WIDTH + j] << " ";
        }
        cout << endl;
    }
    cout << endl;

    cout << "高斯滤波" << endl;
    GaussianFilter(src, dst, WIDTH, HEIGHT, M_WIDTH, M_HEIGHT, 1.0); // deta=1.0 更标准
    for(int i = 0; i < HEIGHT; i++){
        for(int j = 0; j < WIDTH; j++){
            cout << dst[i * WIDTH + j] << " ";
        }
        cout << endl;
    }
    cout << endl;

    cout << "中值滤波" << endl;
    MedianFilter(src, dst, WIDTH, HEIGHT, M_WIDTH, M_HEIGHT);
    for(int i = 0; i < HEIGHT; i++){
        for(int j = 0; j < WIDTH; j++){
            cout << dst[i * WIDTH + j] << " ";
        }
        cout << endl;
    }
    cout << endl;

    // 释放内存
    delete[] src;
    delete[] dst;

    return 0;
}