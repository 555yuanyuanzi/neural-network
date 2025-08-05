/*
K均值聚类（原型聚类），密度聚类，层次聚类
*/
#include <iostream>  
#include <vector>  
#include <cmath>  
#include <limits>  
#include <algorithm>  
using namespace std;
struct Point {  
    double x, y;  
    Point(double x = 0, double y = 0) : x(x), y(y) {}  
};  
  
double distance(const Point& a, const Point& b) {  
    return hypot(a.x - b.x, a.y - b.y);//<cmath>库中的hypot计算欧几里得距离  
    //return sqrt(pow(a.x-b.x,2)+pow(a.y-b.y,2));
}  
  
//计算点集的质心
Point centroid(const vector<Point>& cluster) {  
    double sum_x = 0, sum_y = 0;  
    for (const auto& point : cluster) {  
        sum_x += point.x;  
        sum_y += point.y;  
    }  
    return Point(sum_x / cluster.size(), sum_y / cluster.size()); //初始化Point对象
}  

void kmeans(vector<Point>& data, int k, int max_iterations) {  
    if (data.empty() || k <= 0 || k > data.size()) return;  // 边界检查
    vector<Point> centroids(k);  
    vector<vector<Point>> clusters(k);  
    // 初始化质心
    for (int i = 0; i < k; ++i) {  
        centroids[i] = data[rand() % data.size()];  
    }  
    for (int iter = 0; iter < max_iterations; ++iter) {   
        // 清空上一次的簇
        for (auto& cluster : clusters) {  
            cluster.clear();  
        }  
        // 分配点到簇
        for (const auto& point : data) {  
            double min_distance = numeric_limits<double>::max();//初始化为最大值
            int cluster_index = 0;  
            for (int i = 0; i < k; ++i) {  
                double dist = distance(point, centroids[i]);  
                if (dist < min_distance) {  
                    min_distance = dist;  
                    cluster_index = i;  
                }  
            }  
            clusters[cluster_index].push_back(point);  
        } 
        // 计算新质心
        vector<Point> new_centroids(k);  
        for (int i = 0; i < k; ++i) {  
            if (!clusters[i].empty()) {  // 避免空簇
                new_centroids[i] = centroid(clusters[i]);  
            } else {  
                new_centroids[i] = data[rand() % data.size()];  // 处理空簇
            }
        }  
        //判断收敛
        bool converged = true;  
        for (int i = 0; i < k; ++i) {  
            if (distance(centroids[i], new_centroids[i]) > 1e-6) {  
                converged = false;  
                break;  
            }  
        }  
        if (converged) break;  
        centroids = new_centroids;  
    }  
    //输出结果
    for (int i = 0; i < k; ++i) {  
    cout << "分类" << i + 1 << "质心：(" << centroids[i].x << ", " << centroids[i].y << ")" << std::endl;  
    for (const auto& point : clusters[i]) {  
        cout << "(" << point.x << ", " << point.y << ") ";  
    }  
    cout << endl;  
}  
}  
  
int main() {  
    srand(time(nullptr)); //随机数种子，可以使用随机数生成数据集  
    vector<Point> data = {  
        // 数据集 
        {2.0, 10.0}, {2.5, 8.4}, {5.0, 8.0}, {8.0, 8.0}, {1.0, 0.6},  
        {9.0, 11.0}, {8.0, 5.0}, {1.0, 4.0}, {4.0, 6.0}, {6.0, 9.0}  
    };  
    int k = 2; //集群数量 
    int max_iations = 5; //迭代次数
    kmeans(data, k, max_iations);  
    return 0;  
}