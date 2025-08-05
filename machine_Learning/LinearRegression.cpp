#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <vector>
using namespace std;

/*
机器学习任务体系
1.有监督学习（数据有标签）
   预测任务
      分类（输出离散类别）
      回归（输出连续数值）
   其他（如排序学习）
2.无监督学习（数据无标签）
   聚类（发现数据分组）
   降维（数据维度压缩）
   其他（如异常检测）
*/
// 数组操作函数
class Utils {
public:
    // 计算数组均值
    static double array_mean(double arr[], int m) {
        double sum = 0.0;
        for (int i = 0; i < m; i++) {
            sum += arr[i];
        }
        return sum / m;
    }
    
    // 计算两个数组的协方差
    static double covariance(double x[], double y[], int m) {
        double mean_x = array_mean(x, m);
        double mean_y = array_mean(y, m);
        double cov = 0.0;
        for (int i = 0; i < m; i++) {
            cov += (x[i] - mean_x) * (y[i] - mean_y);
        }
        return cov / (m-1);
    }
    
    // 计算数组方差
    static double variance(double x[], int m) {
        double mean = array_mean(x, m);
        double var = 0.0;
        for (int i = 0; i < m; i++) {
            var += pow(x[i] - mean, 2);
        }
        return var / (m-1);// 样本很大时可近似于m
    }
};

// 线性回归类:最小二乘法 y=kx+b
class LinearRegression {
private:
    double *x;       // 训练数据特征
    double *y;       // 训练数据标签
    int m;           // 样本数量
    double b;        // 偏置项参数
    double k;        // 权重参数

public:
    // 构造函数：初始化训练数据
    LinearRegression(double x[], double y[], int m) {
        this->x = x;
        this->y = y;
        this->m = m;
    }

    // 析构函数：释放资源
    ~LinearRegression() {}

    // 使用最小二乘法训练模型
    void LinearLeastSquares() {
        if (m <= 1) {
            cout << "错误: 样本数量必须大于1" << endl;
            return;
        }
        
        // 计算最小二乘法的参数
        double mean_x = Utils::array_mean(x, m);
        double mean_y = Utils::array_mean(y, m);
        
        // 计算k = cov(x,y) / var(x)
        double cov_xy = Utils::covariance(x, y, m); 
        double var_x = Utils::variance(x, m);       
        k = cov_xy / var_x;
        
        // 计算b = mean_y - k * mean_x
        b = mean_y - k * mean_x;
        
        // 输出训练结果
        cout << "训练完成，最小二乘法参数:" << endl;
        cout << "b = " << b << ", k = " << k << endl;
    }

    // 预测函数
    double predict(double x_value) {
        return b + k * x_value;
    }
};

// 从文件读取数据
double* readFile(string fileName, int& length) {
    ifstream in(fileName);
    if (!in.is_open()) {
        cout << "错误: 无法打开文件 " << fileName << endl;
        return nullptr;
    }
    
    // 计算数据长度
    double temp;
    length = 0;
    while (in >> temp) {
        length++;
    }
    in.close();
    
    // 重新打开文件读取数据
    in.open(fileName);
    double* data = new double[length];
    for (int i = 0; i < length; i++) {
        in >> data[i];
    }
    in.close();
    
    return data;
}
// 读取文件简洁写法
/*
vector<double> readFile(string fileName) {
    ifstream in(fileName);
    if (!in) return {};
    
    vector<double> data;
    double num;
    while (in >> num) {
        data.push_back(num);  // 自动扩容，无需提前统计长度
    }
    return data;
}
*/

int main() {
    double x_predict = 2.1212; // 预测值
    
    // 读取训练数据
    string fileNameX = "data/datax.dat";
    string fileNameY = "data/datay.dat";
    int lengthX = 0, lengthY = 0;
    
    double* X = readFile(fileNameX, lengthX);
    double* Y = readFile(fileNameY, lengthY);
    
    if (X == nullptr || Y == nullptr || lengthX != lengthY) {
        cout << "错误: 数据读取失败或数据长度不匹配!" << endl;
        return 1;
    }
    
    cout << "成功读取 " << lengthX << " 条数据" << endl;
    
    // 创建并训练模型
    LinearRegression lr(X, Y, lengthX);
    lr.LinearLeastSquares();
    
    // 进行预测
    double y_predict = lr.predict(x_predict);
    cout << "预测值: x = " << x_predict << " 时, y = " << y_predict << endl;
    
    // 释放内存
    delete[] X;
    delete[] Y;
    
    return 0;
}
