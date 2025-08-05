#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <cstdlib>
using namespace std;

//数据结构：训练数据
struct DataPoint {
    vector<double> features;  //特征向量(x1,x2,x3,...)
    int label;                //类别标签 (-1 或 1)
};

//支持向量机的参数
class SVM {
private:
    vector<DataPoint> data;    //训练数据
    vector<double> alphas;     //拉格朗日乘子
    double b;                  //偏置
    double c;                  //惩罚因子
    double tolerance = 1e-3;   //容忍度，用于检查KKT条件

    //计算内积（内积核）
    double dotProduct(const DataPoint& x, const DataPoint& y) {
        double result = 0.0;
        for (size_t i = 0; i < x.features.size(); i++) {
            result += x.features[i] * y.features[i];
        }
        return result;
    }

    //计算核函数,此处为线性核
    double kernel(const DataPoint& x, const DataPoint& y) {
        return dotProduct(x, y);
    }

    //计算误差,计算第i个样本的预测值与真实标签之间的误差
    double computeError(int i) {
        double result = b;
        for (size_t j = 0; j < data.size(); ++j) {
            result += alphas[j] * data[j].label * kernel(data[i], data[j]);//预测值公式
        }
        return result - data[i].label;
    }

    //计算目标函数值
    double objectiveFunction() {
        double sum_alpha = 0.0;
        double sum_kernel = 0.0;
        for (size_t i = 0; i < data.size(); i++) {
            sum_alpha += alphas[i];//目标函数公式
            for (size_t j = 0; j < data.size(); j++) {
                sum_kernel += alphas[i] * alphas[j] * data[i].label * data[j].label * kernel(data[i], data[j]);
            }
        }
        return sum_alpha - 0.5 * sum_kernel;
    }

    //检查样本是否满足KKT条件
    bool isKKTSatisfied(int i) {
        double fxi = computeError(i) + data[i].label;
        if (alphas[i] == 0) {
            return data[i].label * fxi >= 1 - tolerance;
        } else if (alphas[i] < c) {
            return fabs(data[i].label * fxi - 1) < tolerance;
        } else {
            return data[i].label * fxi <= 1 + tolerance;
        }
    }

    // 检查所有样本是否满足KKT条件
    bool checkKKT() {
        for (size_t i = 0; i < data.size(); ++i) {
            if (!isKKTSatisfied(i)) {
                return false;
            }
        }
        return true;
    }

    //序列最小优化（SMO）算法
    bool takeStep(int i, int j) {
        if (i == j){
            return false;
        }
        double Ei = computeError(i);
        double Ej = computeError(j);

        double alpha_i = alphas[i];
        double alpha_j = alphas[j];

        double yI = data[i].label;
        double yJ = data[j].label;

        double s = yI * yJ;
        double L, H;

        if (yI == yJ) {
            L = max(0.0, alphas[j] + alphas[i] - c);
            H = min(c, alphas[j] + alphas[i]);
        } else {
            L = max(0.0, alphas[j] - alphas[i]);
            H = min(c, c + alphas[j] - alphas[i]);
        }

        if (L == H) return false;

        double eta = 2.0 * kernel(data[i], data[j]) - kernel(data[i], data[i]) - kernel(data[j], data[j]);
        if (eta >= 0) return false;

        alphas[j] -= yJ * (Ei - Ej) / eta;
        alphas[j] = min(H, max(L, alphas[j]));

        if (fabs(alphas[j] - alpha_j) < 1e-5) return false;

        alphas[i] += s * (alpha_j - alphas[j]);

        double b1 = b - Ei - yI * (alphas[i] - alpha_i) * kernel(data[i], data[i]) - yJ * (alphas[j] - alpha_j) * kernel(data[i], data[j]);
        double b2 = b - Ej - yI * (alphas[i] - alpha_i) * kernel(data[i], data[j]) - yJ * (alphas[j] - alpha_j) * kernel(data[j], data[j]);

        if (0 < alphas[i] && alphas[i] < c) b = b1;
        else if (0 < alphas[j] && alphas[j] < c) b = b2;
        else b = (b1 + b2) / 2.0;

        return true;
    }

public:
    SVM(vector<DataPoint> data, double c = 1.0) : data(data), c(c) {
        alphas.resize(data.size(), 0.0);
        b = 0.0;
    }

    //训练SVM模型
    void train(int maxIter = 100) {
        int iter = 0;
        bool allKKTSatisfied = false;
        while (iter < maxIter &&!allKKTSatisfied) {
            int numchanged = 0;
            for (int i = 0; i < data.size(); i++) {
                if (!isKKTSatisfied(i)) {
                    for (int j = 0; j < data.size(); j++) {
                        if (takeStep(i, j)) {
                            numchanged++;
                        }
                    }
                }
            }
            iter++;
            allKKTSatisfied = checkKKT();
        }
    }

    //预测函数
    int predict(const DataPoint& x) {
        double sum = b;
        for (size_t i = 0; i < data.size(); i++) {
            sum += alphas[i] * data[i].label * kernel(x, data[i]);
        }
        return (sum >= 0)? 1 : -1;
    }

    //打印支持向量
    void printSupportVectors() {
        for (size_t i = 0; i < alphas.size(); i++) {
            if (alphas[i] > 0) {
                cout << "支持向量: ";
                for (auto& feature : data[i].features) {
                    cout << feature << " ";
                }
                cout << "分类标签: " << data[i].label << endl;
            }
        }
    }
};

int main() {
    //创建一些示例数据
    vector<DataPoint> data = {
        {{2.0, 3.0}, 1},
        {{3.0, 3.0}, 1},
        {{4.0, 1.0}, -1},
        {{5.0, 2.0}, -1}
    };

    //创建SVM对象并训练
    SVM svm(data);
    svm.train();

    //打印支持向量
    svm.printSupportVectors();

    //测试预测
    DataPoint testData = {{3.0, 2.0}, 0};
    int prediction = svm.predict(testData);
    cout << "测试数据的预测分类标签为: " << prediction << endl;

    return 0;
}