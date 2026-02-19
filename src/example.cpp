// example.cpp - 线性回归示例，演示 SCOPE 算法的默认行为
#include <Eigen/Eigen>
#include <tuple>
#include <vector>
#include <iostream>
#include <cmath>

#include "UniversalData.h"
#include "utilities.h"
#include "models.h"

using namespace Eigen;
using namespace std;

// 声明 run_scope（定义在 main.cpp 中）
extern tuple<VectorXd, double, double, double>
run_scope(MatrixXd* universal_data, UniversalModel universal_model, ConvexSolver convex_solver,
          int model_size, int sample_size, int aux_para_size, int max_iter,
          int exchange_num, int path_type, bool is_greedy, bool use_hessian,
          bool is_dynamic_exchange_num, bool is_warm_start, int ic_type, double ic_coef,
          int Kfold, VectorXi sequence, VectorXd lambda_seq, int s_min, int s_max,
          int screening_size, VectorXi g_index, VectorXi always_select,
          int thread, int splicing_type, int sub_search, VectorXi cv_fold_id,
          VectorXi A_init, VectorXd beta_init, VectorXd coef0_init);

int main() {
    cout << "=== ScopeCpp 线性回归示例 ===" << endl;

    // 初始化日志
    init_spdlog(2, 2, "scope.log");

    // ==================== 构造测试数据 ====================
    // y = X * beta_true + noise，其中 beta_true 只有3个非零分量
    int n = 500;   // 样本数
    int p = 20;    // 特征数

    // 使用固定种子生成标准正态分布数据，确保结果可复现
    srand(123);
    // Box-Muller 变换生成正态分布矩阵
    auto randn = [](int rows, int cols) -> MatrixXd {
        MatrixXd U1 = (MatrixXd::Random(rows, cols).array() + 1.0) / 2.0;  // [0, 1]
        MatrixXd U2 = (MatrixXd::Random(rows, cols).array() + 1.0) / 2.0;  // [0, 1]
        // 防止 log(0)
        U1 = U1.array().max(1e-10);
        return ((-2.0 * U1.array().log()).sqrt() * (2.0 * M_PI * U2.array()).cos()).matrix();
    };

    MatrixXd X = randn(n, p);

    VectorXd beta_true = VectorXd::Zero(p);
    beta_true(0) = 1.5;
    beta_true(2) = -0.8;
    beta_true(5) = 2.0;

    // 高信噪比：噪声标准差 = 0.1
    VectorXd noise = 0.1 * randn(n, 1);
    VectorXd y = X * beta_true + noise;

    // 合并数据矩阵 [X, y]
    MatrixXd* data = new MatrixXd(n, p + 1);
    data->leftCols(p) = X;
    data->col(p) = y;

    // ==================== 配置模型 ====================
    UniversalModel model;
    model.set_loss_of_model(LinearRegressionModel::loss);
    model.set_gradient_user_defined(LinearRegressionModel::loss_and_gradient);
    model.set_slice_by_sample(LinearRegressionModel::slice_by_sample);

    // ==================== 配置优化器 ====================
    ConvexSolver solver = [](
        function<double(VectorXd const&, const MatrixXd*)> loss_fn,
        function<pair<double, VectorXd>(const VectorXd&, const MatrixXd*)> value_and_grad,
        const VectorXd& init_para,
        const VectorXi& active_indices,
        const MatrixXd* data
    ) -> pair<double, VectorXd> {
        return GradientDescentSolver::solve(loss_fn, value_and_grad, init_para,
                                           active_indices, data, 5000, 1e-8, 0.5);
    };

    // ==================== 运行 SCOPE 算法 ====================
    // 使用 sequential path (path_type=1)，遍历稀疏度从 1 到 p
    // 使用 BIC 信息准则自动选择最佳稀疏度
    // 大部分参数使用默认值

    // 自动生成 support_size 序列：1, 2, ..., p
    VectorXi support_size_list = VectorXi::LinSpaced(p, 1, p);
    VectorXd lambda_seq = VectorXd::Zero(1);  // lambda = 0，无正则化

    // 分组信息：每个变量独立一组
    VectorXi g_index = VectorXi::LinSpaced(p, 0, p - 1);
    VectorXi always_select = VectorXi::Zero(0);
    VectorXi cv_fold_id = VectorXi::Zero(n);
    VectorXi A_init = VectorXi::Zero(0);
    VectorXd beta_init = VectorXd::Zero(p);
    VectorXd coef0_init = VectorXd::Zero(0);

    cout << "运行 SCOPE 算法（BIC + Sequential Path）..." << endl;

    VectorXd result_beta;
    double train_loss, test_loss, ic;

    tie(result_beta, train_loss, test_loss, ic) = run_scope(
        data, model, solver,
        /* model_size */ p,
        /* sample_size */ n,
        /* aux_para_size */ 0,
        /* max_iter */ 30,
        /* exchange_num */ 5,
        /* path_type */ 1,
        /* is_greedy */ true,
        /* use_hessian */ false,
        /* is_dynamic_exchange_num */ true,
        /* is_warm_start */ true,
        /* ic_type */ 2,       // BIC
        /* ic_coef */ 1.0,
        /* Kfold */ 1,
        support_size_list, lambda_seq,
        /* s_min */ 1,
        /* s_max */ p,
        /* screening_size */ -1,
        g_index, always_select,
        /* thread */ 1,
        /* splicing_type */ 0,
        /* sub_search */ 0,
        cv_fold_id, A_init, beta_init, coef0_init
    );

    // ==================== 输出结果 ====================
    cout << "\n=== 结果 ===" << endl;
    cout << "真实 beta（非零项）: [0]=1.5, [2]=-0.8, [5]=2.0" << endl;
    cout << "估计 beta:" << endl;
    for (int i = 0; i < result_beta.size(); i++) {
        if (abs(result_beta(i)) > 1e-6) {
            cout << "  beta[" << i << "] = " << result_beta(i) << endl;
        }
    }
    cout << "训练损失: " << train_loss << endl;
    cout << "BIC: " << ic << endl;

    // 验证结果正确性
    int correct_count = 0;
    bool correct_support = true;
    for (int i = 0; i < p; i++) {
        bool is_true_nonzero = (abs(beta_true(i)) > 1e-6);
        bool is_est_nonzero = (abs(result_beta(i)) > 1e-6);
        if (is_true_nonzero && is_est_nonzero) {
            correct_count++;
        }
        if (is_true_nonzero != is_est_nonzero) {
            correct_support = false;
        }
    }

    cout << "\n=== 验证 ===" << endl;
    cout << "正确识别的非零变量数: " << correct_count << " / 3" << endl;
    if (correct_support) {
        cout << "✓ 支撑集完全正确！" << endl;
    } else {
        cout << "△ 支撑集部分匹配" << endl;
    }

    cout << "\n程序执行完毕！" << endl;

    return 0;
}
