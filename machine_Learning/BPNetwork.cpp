#include <iostream>      
#include <vector>      
#include <cmath>          
using namespace std;

//实现简单的两层神经网络，用于分类

//损失函数，计算均方误差
double getMSEloss(double x1,double x2){
    return (x1 - x2)*(x1 - x2);
}
//神经网络类
class NNetwork
{
    private:
    int epoches;//训练轮次
    double learning_rate;//学习率
    double w1,w2,w3,w4,w5,w6;//权重系数
    double b1,b2,b3;//偏置系数
    public:
    NNetwork(int es,double lr);//构造函数
    double sigmoid(double x);//激活函数
    double deriv_sigmoid(double x);//求导函数
    double forward(vector<double> data);//向前传播
    void train(vector<vector<double>> data,vector<double> label);
    void predict(vector<vector<double>> test_data,vector<double> test_label);
};
NNetwork::NNetwork(int es,double lr):epoches(es),learning_rate(lr){
    //超参数、参数初始化
    w1=w2=w3=w4=w5=w6=0;
    b1=b2=b3=0;
}
double NNetwork::sigmoid(double x){
    //激活函数
    return 1/(1+exp(-x));
}
double NNetwork::deriv_sigmoid(double x){
    //激活函数求导
    double y = sigmoid(x);
    return y*(1-y);
}
double NNetwork::forward(vector<double> data){
    //前向传播
    double sum_h1 = w1 * data[0] + w2 * data[1] + b1;
    double h1 = sigmoid(sum_h1);
    double sum_h2 = w3 * data[0] + w4 * data[1] + b2;
    double h2 = sigmoid(sum_h2);
    double sum_o1 = w5 * h1 + w6 * h2 + b3;
    return sigmoid(sum_o1);
}
void NNetwork::train(vector<vector<double>> data,vector<double> label){
    for(int epoch=0;epoch<epoches;++epoch){//控制训练轮次
        int total_n = data.size();//存储样本数量
        for(int i=0;i<total_n;++i){
            vector<double> x = data[i];//x存储样本i的所有特征
            double sum_h1 = w1 * x[0] + w2 * x[1] + b1;//每个样本只有两个特征
            double h1 = sigmoid(sum_h1);
            double sum_h2 = w3 * x[0] + w4 * x[1] + b2;
            double h2 = sigmoid(sum_h2);
            double sum_o1 = w5 * h1 + w6 * h2 + b3;
            double o1 = sigmoid(sum_o1);
            double pred = forward(data[i]);
 
            double d_loss_pred = -2 * (label[i] - pred);//计算损失函数
            //输出层权重和偏置的梯度
            double d_pred_w5 = h1 * deriv_sigmoid(sum_o1);
            double d_pred_w6 = h2 * deriv_sigmoid(sum_o1);
            double d_pred_b3 = deriv_sigmoid(sum_o1);
            //隐藏层到输出层的梯度传递
            double d_pred_h1 = w5 * deriv_sigmoid(sum_o1);
            double d_pred_h2 = w6 * deriv_sigmoid(sum_o1);
            //隐藏层权重和偏置的梯度
            double d_h1_w1 = x[0] * deriv_sigmoid(sum_h1);
            double d_h1_w2 = x[1] * deriv_sigmoid(sum_h1);
            double d_h1_b1 = deriv_sigmoid(sum_h1);
            double d_h2_w3 = x[0] * deriv_sigmoid(sum_h2);
            double d_h2_w4 = x[1] * deriv_sigmoid(sum_h2);
            double d_h2_b2 = deriv_sigmoid(sum_h2);
            //SGD梯度下降调整权重和偏置
            w1 -= learning_rate * d_loss_pred * d_pred_h1 * d_h1_w1;
            w2 -= learning_rate * d_loss_pred * d_pred_h1 * d_h1_w2;
            b1 -= learning_rate * d_loss_pred * d_pred_h1 * d_h1_b1;
            w3 -= learning_rate * d_loss_pred * d_pred_h2 * d_h2_w3;
            w4 -= learning_rate * d_loss_pred * d_pred_h2 * d_h2_w4;
            b2 -= learning_rate * d_loss_pred * d_pred_h2 * d_h2_b2;
            w5 -= learning_rate * d_loss_pred * d_pred_w5;
            w6 -= learning_rate * d_loss_pred * d_pred_w6;
            b3 -= learning_rate * d_loss_pred * d_pred_b3; 
        }
        //每10轮计算损失值
        if(epoch%10==0){
            double loss = 0;
            for(int i=0;i<total_n;++i){
                double pred = forward(data[i]);
                loss += getMSEloss(pred,label[i]);
            }
            cout<<"轮次"<<epoch<<"损失值："<<loss<<endl;
        }
    }
}
void NNetwork::predict(vector<vector<double>> test_data,vector<double> test_label){
    int n = test_data.size();
    double cnt = 0;
    for(int i=0;i<n;++i){
        double pred = forward(test_data[i]);
        pred = pred>0.5?1:0;
        cnt += (test_label[i]==pred);
    }
    cout<<"准确率:"<<cnt/n<<endl;//正确率=预测正确的样本数/总样本数
}
int main(){
    vector<vector<double>> data = {{-2,-1},{25,6},{17,4},{-15,-6}};
    vector<double> label = {1,0,0,1};
    NNetwork network = NNetwork(1000,0.1);
    network.train(data,label);
    vector<vector<double>> test_data  = {{-3,-4},{-5,-4},{12,3},{-13,-4},{9,12}};
    vector<double> test_label = {1,1,0,1,0};
    network.predict(test_data,test_label);
    return 0;
}
