#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <iomanip> // 用于格式化输出

using namespace std;

// 定义一个结构体来表示数据集中的每一行
struct Sample {
    vector<string> features; // 特征: 帅/性格好/身高/上进
    string label;            // 标签: 嫁/不嫁
};

// 朴素贝叶斯分类器类
class NaiveBayesClassifier {
private:
    // P(C):存储每个类别的先验概率（嫁和不嫁）
    map<string, double> priors;
    // P(X|C):存储P(特征|类别)的值
    //在嫁的条件下，身高为 高 的概率是多少
    //vector里存放嫁/不嫁条件下，所有特征（帅、性格、身高、上进）的概率信息
    map<string, vector<map<string, double>>> likelihoods;
    // 用于拉普拉斯平滑的 alpha 值
    double alpha = 1.0; 
    // 存储每个特征有多少个不同的取值
    vector<int> feature_value_counts;
    // 存储所有唯一的类别标签
    set<string> class_labels;

public:
    // 训练模型
    void train(const vector<Sample>& dataset) {
        if (dataset.empty()) return;

        int num_samples = dataset.size();
        int num_features = dataset[0].features.size();
        feature_value_counts.resize(num_features);

        map<string, int> class_counts;//记录么每个类别出现多少次
        set<string> unique_labels;//存放不重复的类别名称
        vector<set<string>> unique_feature_values(num_features);//第0个set用来存帅/不帅

        for (const auto& sample : dataset) {// 遍历数据集
            class_counts[sample.label]++;
            unique_labels.insert(sample.label);
            for (int i = 0; i < num_features; ++i) {//遍历这一行的所有特征
                unique_feature_values[i].insert(sample.features[i]);
            }
        }
        class_labels = unique_labels;
        for(int i = 0; i < num_features; ++i) {
            feature_value_counts[i] = unique_feature_values[i].size();//每个特征有多少个不同的取值记录下来
        }

        // 计算先验概率P(C)，嫁/不嫁的概率
        for (const auto& pair : class_counts) {
            priors[pair.first] = (double)pair.second / num_samples;
        }

        // 计算条件概率 P(X|C)
        // 统计每个类别下，每个特征值的出现次数
        map<string, vector<map<string, int>>> feature_counts;//feature_counts["不嫁"][2]["矮"]
        for (const auto& label : class_labels) {
            feature_counts[label].resize(num_features);
        }

        for (const auto& sample : dataset) {
            for (int i = 0; i < num_features; ++i) {
                feature_counts[sample.label][i][sample.features[i]]++;
            }
        }
        
        // 使用拉普拉斯平滑计算概率
        for (const auto& label : class_labels) {
            likelihoods[label].resize(num_features);
            for (int i = 0; i < num_features; ++i) {
                int total_class_count = class_counts[label];
                int V = feature_value_counts[i]; // 当前特征的可能取值数量
                // 对该特征下所有出现过的值计算概率
                for (const auto& feature_val : unique_feature_values[i]) {
                    int count = feature_counts[label][i][feature_val];
                    //拉普拉斯平滑公式:(次数 + 1) / (类别总数 + 1 * V)
                    likelihoods[label][i][feature_val] = (count + alpha) / (total_class_count + alpha * V);
                }
            }
        }
    }

    // 预测新样本
    string predict(const vector<string>& features) {
        string best_class;
        double max_prob = -1.0;

        // 对每个可能的类别进行计算
        for (const auto& label : class_labels) {
            // 从先验概率开始
            double current_prob = priors[label];
            // 连乘条件概率 P(f1|C) * P(f2|C) * ...
            for (int i = 0; i < features.size(); ++i) {
                // 检查训练集中是否存在该特征值
                if (likelihoods[label][i].count(features[i])) {
                    current_prob *= likelihoods[label][i][features[i]];
                } else {
                    // 如果这是一个在训练时从未见过的特征值（对于这个类别）
                    // 我们使用平滑后的默认概率
                    int total_class_count = priors[label] * 12;
                    int V = feature_value_counts[i];
                    double prob_for_unseen = alpha / (total_class_count + alpha * V);
                    current_prob *= prob_for_unseen;
                }
            }
            
            // 找到概率最大的类别
            if (current_prob > max_prob) {
                max_prob = current_prob;
                best_class = label;
            }
        }
        cout<<"最大概率为："<<max_prob<<endl;
        return best_class;
    }

    // 打印学习到的概率，便于理解和调试
    void print_p() {
        cout << fixed << setprecision(3);
        cout << "先验概率 P(C)" << endl;
        for (const auto& pair : priors) {
            cout << "P(" << pair.first << ") = " << pair.second << endl;
        }
        cout << "条件概率 P(特征 | C)" << endl;
        vector<string> feature_names = {"帅:", "性格:", "身高:", "上进:"};
        for (const auto& label : class_labels) {
            cout << "类别: " << label << endl;
            for (int i = 0; i < likelihoods[label].size(); ++i) {
                cout << "  " << feature_names[i] << ":" << endl;
                for (const auto& pair : likelihoods[label][i]) {
                    cout << "    P(" << pair.first << " | " << label << ") = " << pair.second << endl;
                }
            }
        }
    }
};

int main() {
    // 1. 根据图片手动录入数据集
    vector<Sample> dataset = {
        {{"帅",   "不好", "矮", "不上进"}, "不嫁"},
        {{"不帅", "好",   "矮", "上进"},   "不嫁"},
        {{"帅",   "好",   "矮", "上进"},   "嫁"},
        {{"不帅", "好",   "高", "上进"},   "嫁"},
        {{"帅",   "不好", "矮", "上进"},   "不嫁"},
        {{"不帅", "不好", "矮", "不上进"}, "不嫁"},
        {{"帅",   "好",   "高", "不上进"}, "嫁"},
        {{"不帅", "好",   "高", "上进"},   "嫁"},
        {{"帅",   "好",   "高", "上进"},   "嫁"},
        {{"不帅", "不好", "高", "上进"},   "嫁"},
        {{"帅",   "好",   "矮", "不上进"}, "不嫁"},
    };
    NaiveBayesClassifier classifier;
    cout<<"开始训练"<<endl;
    classifier.train(dataset);
    // 打印学习到的概率模型
    classifier.print_p();
    cout << "\n进行预测" << endl;
    vector<string> new_sample = {"帅", "不好", "不高", "上进"};
    cout << "待预测: {帅, 不好, 不高, 上进}" << endl;

    string prediction = classifier.predict(new_sample);
    cout << "预测结果: " << prediction << endl;

    return 0;
}