#include <iostream>
#include <string>
#include <vector>
#include <set>
#include <algorithm>
#include <map>
#include <cmath>
#include <math.h>
#include <fstream>
#include <sstream>
#include <iomanip>
using namespace std;
 
//决策树模型：基于信息增益（ID3 算法）或信息增益比（C4.5 算法）
#define N 14
#define feature 4
vector< vector<string> > X;

//属性名
vector<string> attributes{"天气", "温度", "湿度", "是否有风"};
//文本文件中读取数据，将其解析为一个二维数组（数据集）
void createDataset() {
    ifstream file("data/decision-data.txt");
    string line;
    while (getline(file, line)) {
        vector<string> row;
        istringstream iss(line);
        string value;
        while (iss >> value) {
            row.push_back(value);
        }
        X.push_back(row);
    }
    file.close();
}

// 打印二维数组 X 的内容
void printDataset() {
    cout << "数据集内容（行数: " << X.size() << ", 列数: " << (X.empty() ? 0 : X[0].size()) << "):" << endl;
    for (const auto& row : X) {
        for (const auto& value : row) {
            
            cout << left <<setw(15) << value;  // 左对齐，宽度15字符
        }
        cout << endl;
    }
}
 
//计算给定数据集的香农信息熵
double calcShanno(const vector< vector<string> > &data) {
	 int n = data.size();
	 map<string, int> classCounts;//map：否：2 是：4
	 int i;
	 int label = data[0].size() - 1;//类别标签的索引
	 for(i=0; i<n; i++)
		classCounts[ data[i][label] ] = 0;
     //统计每个标签类出现的次数
	 for(i=0; i<data.size(); i++)
		classCounts[ data[i][label] ] += 1;
	 //计算香农熵
	 double shanno = 0;
	 map<string, int>::iterator it;
	 for(it = classCounts.begin(); it != classCounts.end(); it++) {
		 double prob = (double)(it->second) / (double)n;
		 shanno -= prob * log2(prob);
	 }
	 return shanno;
}
 
//按指定特征划分数据集 axis ：特征下标 value：特征值
vector< vector<string> > splitDataSet(const vector< vector<string> > data, int axis, string value) {
	vector< vector<string> > result;
	for(int i=0; i<data.size(); i++) {
		if(data[i][axis] == value) {
			vector<string> removed(data[i].begin(), data[i].begin()+axis);
			removed.insert(removed.end(), data[i].begin()+axis+1, data[i].end());
			result.push_back(removed);//保存移除特征列后的数据
		}
	}
	return result;
}
 
//创建特征列表
vector<string> createFeatureList(const vector< vector<string> > &data, int axis) {
	int n = data.size();
	vector<string>featureList;   //特征的所有取值
	set<string> s;
	for(int j=0; j<n; j++)    //寻找该特征的所有可能取值
		s.insert(data[j][axis]);
	set<string>::iterator it;
	for(it = s.begin(); it != s.end(); it++) {
		featureList.push_back(*it);
	}
	return featureList;
}
 
//选择最好的数据集划分方式
int BestFeatureToSplit(const vector< vector<string> > &data) {
	int n = data[0].size()  ; 
	double bestEntropy = calcShanno(data);  //初始香农熵
	double bestInfoGain = 0;   //最大的信息增益
	int bestFeature = 0;       //最好的特征
    //所有特征
	for(int i=0; i<n; i++) {
		double newEntropy = 0;
        //该特征的所有可能取值
		vector<string> featureList = createFeatureList(data, i);  
		for(int j=0; j<featureList.size(); j++) {//对每个取值划分出子集并计算累加加权熵
			vector< vector<string> > subData = splitDataSet(data, i, featureList[j]);
			double prob = (double)subData.size() / (double)data.size();
			newEntropy += prob * calcShanno(subData);   
		}
        //选取较大的信息增益
		double infoGain = bestEntropy - newEntropy;  
		if(infoGain > bestInfoGain) {
			bestInfoGain = infoGain;
			bestFeature = i;
		}
	}
	return bestFeature;
}
 
//返回出现次数最多的分类名称
//如果数据集已处理了所有属性，但类标签依然不是唯一的，采用多数表决的方法定义叶子节点的分类
string majorityCnt(vector<string> &classList) {
	int n = classList.size();
	map<string, int> classCount;
	int i;
	for(i=0; i<n; i++)
		classCount[classList[i]] = 0;
	for(i=0; i<n; i++)//统计各个类别出现的次数
		classCount[classList[i]] += 1;
	int temp = 0;//存储最大的次数
	map<string, int>::iterator it;
	string result = "";
	for(it = classCount.begin(); it != classCount.end(); it++) {
		if(it->second > temp) {//比较当前类别计数与最大值
			temp = it->second;
			result = it->first;
		}
	}
	return result;//返回出现次数最大的类别
}

 //创建决策树结点结构体
struct Node {
	string attribute;//属性
	string val;//特征取值
	bool isLeaf;//是否为叶子
	vector<Node*> childs;
	Node() {
		val = "";
		attribute = "";
		isLeaf = false;
	}
};
Node *root = NULL;
 
//递归构建决策树
Node* createTree(Node *root, const vector< vector<string> > &data, vector<string> &attribute) {
	if(root == NULL)
		root = new Node();
	vector<string> classList;
	set<string> classList1;//存储去重后的类别标签
	int i, j;
	int label = data[0].size() - 1;
	int n = data.size();
	for(i=0; i<n; i++) {
		classList.push_back(data[i][label]);//将第i个样本的类别标签添加到列表中
		classList1.insert(data[i][label]);//将类别标签添加到集合中（自动去重）
	}
    //如果所有实例都属于同一类，停止划分
	if(classList1.size() == 1) {
		if(classList[0] == "适合")
			root->attribute = "适合";
		else
			root->attribute = "不适合";
		root->isLeaf = true;
		return root;
	}
    //遍历完所有特征，返回出现次数最多的类别
	if(data[0].size() == 1) {//表示没有剩余特征可供划分
		root->attribute = majorityCnt(classList);
		return root;
	}
 
	int bestFeatureIndex = BestFeatureToSplit(data);
    //确定当前最优特征的所有可能取值，并记录在决策树节点中
	vector<string> featureList = createFeatureList(data, bestFeatureIndex);  
	string bestFeature = attribute[bestFeatureIndex];
	root->attribute = bestFeature;   
    //对于当前属性的每个可能值，创建新的分支
	for(i=0; i<featureList.size(); i++) {
		vector<string> subAttribute;  
		for(j=0; j<attribute.size(); j++) {
			if(bestFeature != attribute[j])
			    //移除已使用的特征，生成剩余可用特征列表subAttribute
				subAttribute.push_back(attribute[j]);
		}
		Node *newNode = new Node();
		newNode->val = featureList[i];//记录属性的取值
		createTree(newNode, splitDataSet(data, bestFeatureIndex, featureList[i]), subAttribute);
		root->childs.push_back(newNode);
	}
	return root;
}
 
//打印
void print(Node *root, int depth) {
	int i;
	for(i=0; i<depth; i++)
		cout << "\t";
	if(root->val != "") {
		cout << root->val << endl;
		for(i=0; i<depth+1; i++)
			cout << "\t";
	}
	cout << root->attribute << endl;
	vector<Node*>::iterator it;
	for(it = root->childs.begin(); it != root->childs.end(); it++) {
		print(*it, depth+1);
	}
}
 
//预测x
string classify(Node *root, vector<string> &attribute, string *test) {
	string firstFeature = root->attribute;
	int firstFeatureIndex;
	int i;
    //找到根节点是第几个特征
	for(i=0; i<attribute.size(); i++) {
		if(firstFeature == attribute[i]) {
			firstFeatureIndex = i;
			break;
		}
	}
	if(root->isLeaf)  //如果是叶子节点，直接输出结果
		return root->attribute;
	for(i=0; i<root->childs.size(); i++) {
		if(test[firstFeatureIndex] == root->childs[i]->val) {
			return classify(root->childs[i], attribute, test);
		}
	}
}
 
//释放节点
void freeNode(Node *root) {
	if(root == NULL)
		return;
	vector<Node*>::iterator it;
	for(it=root->childs.begin(); it != root->childs.end(); it++)
		freeNode(*it);
	delete root;
}
 
int main() {	
	createDataset();
    printDataset(); 
	root = createTree(root, X, attributes);
	print(root, 0);
	string test[] = {"晴", "温", "中", "是"};
	int i;
	cout << endl << "属性：";
	for(i=0; i<feature; i++)
		cout << attributes[i] << "\t";
	cout << endl << "输入：";
	for(i=0; i<feature; i++)
		cout << test[i] << "\t";
	cout << endl << "预测：";
	cout << classify(root, attributes, test) +"出行" << endl;
	freeNode(root);
	return 0;
}