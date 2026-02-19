// models.h - 提供示例损失函数和优化器
#pragma once

#include <Eigen/Eigen>
#include <functional>
#include <utility>

using Eigen::VectorXd;
using Eigen::VectorXi;
using Eigen::MatrixXd;
using std::function;
using std::pair;

// 简单的线性回归损失函数
// data 格式：前 n 列是 X，最后一列是 y
class LinearRegressionModel {
public:
    // 计算损失：RSS = sum((y - X*beta)^2) / (2*n)
    static double loss(const VectorXd& beta, const MatrixXd* data);
    
    // 计算损失和梯度
    static pair<double, VectorXd> loss_and_gradient(const VectorXd& beta, const MatrixXd* data);
    
    // 按样本切片
    static MatrixXd* slice_by_sample(const MatrixXd* data, const VectorXi& indices);
};

// 简单的梯度下降优化器
class GradientDescentSolver {
public:
    static pair<double, VectorXd> solve(
        function<double(VectorXd const&, const MatrixXd*)> loss_fn,
        function<pair<double, VectorXd>(const VectorXd&, const MatrixXd*)> value_and_grad,
        const VectorXd& init_para,
        const VectorXi& active_indices,
        const MatrixXd* data,
        int max_iter = 1000,
        double tol = 1e-6,
        double learning_rate = 0.01
    );
};
