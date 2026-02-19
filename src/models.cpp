#include "models.h"
#include <iostream>
#include <cmath>

using namespace Eigen;
using namespace std;

// 线性回归损失函数实现
double LinearRegressionModel::loss(const VectorXd& beta, const MatrixXd* data) {
    if (!data) return 0.0;
    
    int n = data->rows();
    int p = data->cols() - 1;  // 最后一列是 y
    
    MatrixXd X = data->leftCols(p);
    VectorXd y = data->col(p);
    
    // 只使用 beta 的前 p 个元素
    VectorXd beta_used = beta.head(p);
    
    VectorXd residual = y - X * beta_used;
    return residual.squaredNorm() / (2.0 * n);
}

pair<double, VectorXd> LinearRegressionModel::loss_and_gradient(const VectorXd& beta, const MatrixXd* data) {
    if (!data) return std::make_pair(0.0, VectorXd::Zero(beta.size()));
    
    int n = data->rows();
    int p = data->cols() - 1;
    
    MatrixXd X = data->leftCols(p);
    VectorXd y = data->col(p);
    
    VectorXd beta_used = beta.head(p);
    
    VectorXd residual = y - X * beta_used;
    double loss_val = residual.squaredNorm() / (2.0 * n);
    
    VectorXd grad = VectorXd::Zero(beta.size());
    grad.head(p) = -X.transpose() * residual / n;
    
    return std::make_pair(loss_val, grad);
}

MatrixXd* LinearRegressionModel::slice_by_sample(const MatrixXd* data, const VectorXi& indices) {
    if (!data) return nullptr;
    
    MatrixXd* sliced = new MatrixXd(indices.size(), data->cols());
    for (int i = 0; i < indices.size(); ++i) {
        sliced->row(i) = data->row(indices(i));
    }
    return sliced;
}

// 梯度下降优化器实现
pair<double, VectorXd> GradientDescentSolver::solve(
    function<double(VectorXd const&, const MatrixXd*)> loss_fn,
    function<pair<double, VectorXd>(const VectorXd&, const MatrixXd*)> value_and_grad,
    const VectorXd& init_para,
    const VectorXi& active_indices,
    const MatrixXd* data,
    int max_iter,
    double tol,
    double learning_rate
) {
    VectorXd para = init_para;
    VectorXd active_para = para(active_indices);
    
    double loss_val = 0.0;
    VectorXd grad;
    
    for (int iter = 0; iter < max_iter; ++iter) {
        // 构造完整参数
        VectorXd full_para = VectorXd::Zero(para.size());
        full_para(active_indices) = active_para;
        
        // 计算梯度
        std::tie(loss_val, grad) = value_and_grad(full_para, data);
        VectorXd active_grad = grad(active_indices);
        
        // 检查收敛
        if (active_grad.norm() < tol) {
            break;
        }
        
        // 梯度下降更新
        active_para = active_para - learning_rate * active_grad;
    }
    
    // 构造最终结果
    VectorXd result = VectorXd::Zero(para.size());
    result(active_indices) = active_para;
    
    // 用最终参数重新计算 loss，确保 loss 与参数匹配
    loss_val = loss_fn(result, data);
    
    return std::make_pair(loss_val, result);
}
