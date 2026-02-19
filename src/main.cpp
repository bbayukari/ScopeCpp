// main.cpp - 纯 C++ 入口，替代 pywrap.cpp
#include <Eigen/Eigen>
#include <tuple>
#include <vector>
#include <iostream>

#include "Algorithm.h"
#include "UniversalData.h"
#include "Data.h"
#include "Metric.h"
#include "OpenMP.h"
#include "path.h"
#include "screening.h"
#include "utilities.h"
#include "models.h"

using namespace Eigen;
using namespace std;

tuple<VectorXd, double, double, double>
run_scope(MatrixXd* universal_data, UniversalModel universal_model, ConvexSolver convex_solver, 
          int model_size, int sample_size, int aux_para_size, int max_iter,
          int exchange_num, int path_type, bool is_greedy, bool use_hessian, 
          bool is_dynamic_exchange_num, bool is_warm_start, int ic_type, double ic_coef, 
          int Kfold, VectorXi sequence, VectorXd lambda_seq, int s_min, int s_max, 
          int screening_size, VectorXi g_index, VectorXi always_select,
          int thread, int splicing_type, int sub_search, VectorXi cv_fold_id, 
          VectorXi A_init, VectorXd beta_init, VectorXd coef0_init)
{
#ifdef _OPENMP
    int max_thread = omp_get_max_threads();
    if (thread == 0 || thread > max_thread)
    {
        thread = max_thread;
    }

    setNbThreads(thread);
    omp_set_num_threads(thread);
#endif

    SPDLOG_DEBUG("SCOPE begin!");
    UniversalData x(model_size, sample_size, universal_data, &universal_model, convex_solver);
    MatrixXd y = MatrixXd::Zero(sample_size, aux_para_size);
    int normalize_type = 0;
    VectorXd weight = VectorXd::Ones(sample_size);
    Parameters parameters(sequence, lambda_seq, s_min, s_max);

    int algorithm_list_size = max(thread, Kfold);
    vector<Algorithm*> algorithm_list(algorithm_list_size);
    for (int i = 0; i < algorithm_list_size; i++)
    {
        algorithm_list[i] = new Algorithm(max_iter, is_warm_start, exchange_num, always_select, 
                                         splicing_type, is_greedy, sub_search, use_hessian, 
                                         is_dynamic_exchange_num);
    }

    bool early_stop = true, sparse_matrix = true;
    int beta_size = model_size;

    Data data(x, y, normalize_type, weight, g_index, sparse_matrix, beta_size);

    VectorXi screening_A;
    if (screening_size >= 0)
    {
        screening_A = screening(data, algorithm_list, screening_size, beta_size,
                               parameters.lambda_list(0), A_init);
    }

    Metric* metric = new Metric(ic_type, ic_coef, Kfold);
    if (Kfold > 1)
    {
        metric->set_cv_train_test_mask(data, data.n, cv_fold_id);
        metric->set_cv_init_fit_arg(beta_size, data.M);
    }

    vector<Result> result_list(Kfold);
    if (path_type == 1)
    {
#pragma omp parallel for
        for (int i = 0; i < Kfold; i++)
        {
            sequential_path_cv(data, algorithm_list[i], metric, parameters, early_stop, i, 
                             A_init, beta_init, coef0_init, result_list[i]);
        }
    }
    else
    {
        gs_path(data, algorithm_list, metric, parameters, A_init, beta_init, coef0_init, result_list);
    }

    int min_loss_index = 0;
    int sequence_size = (parameters.sequence).size();
    Matrix<VectorXd, Dynamic, 1> beta_matrix(sequence_size, 1);
    Matrix<VectorXd, Dynamic, 1> coef0_matrix(sequence_size, 1);
    Matrix<VectorXd, Dynamic, 1> bd_matrix(sequence_size, 1);
    MatrixXd ic_matrix(sequence_size, 1);
    MatrixXd test_loss_sum = MatrixXd::Zero(sequence_size, 1);
    MatrixXd train_loss_matrix(sequence_size, 1);
    MatrixXd effective_number_matrix(sequence_size, 1);

    if (Kfold == 1)
    {
        beta_matrix = result_list[0].beta_matrix;
        coef0_matrix = result_list[0].coef0_matrix;
        ic_matrix = result_list[0].ic_matrix;
        train_loss_matrix = result_list[0].train_loss_matrix;
        effective_number_matrix = result_list[0].effective_number_matrix;
        ic_matrix.col(0).minCoeff(&min_loss_index);
    }
    else
    {
        for (int i = 0; i < Kfold; i++)
        {
            test_loss_sum += result_list[i].test_loss_matrix;
        }
        test_loss_sum /= ((double)Kfold);
        test_loss_sum.col(0).minCoeff(&min_loss_index);

        VectorXi used_algorithm_index = VectorXi::Zero(algorithm_list_size);

#pragma omp parallel for
        for (int ind = 0; ind < sequence_size; ind++)
        {
            int support_size = parameters.sequence(ind).support_size;
            double lambda = parameters.sequence(ind).lambda;

            int algorithm_index = omp_get_thread_num();
            used_algorithm_index(algorithm_index) = 1;

            VectorXd beta_init;
            VectorXd coef0_init;
            VectorXi A_init;
            coef_set_zero(beta_size, data.M, beta_init, coef0_init);
            VectorXd bd_init = VectorXd::Zero(data.g_num);

            for (int j = 0; j < Kfold; j++)
            {
                beta_init = beta_init + result_list[j].beta_matrix(ind) / Kfold;
                coef0_init = coef0_init + result_list[j].coef0_matrix(ind) / Kfold;
                bd_init = bd_init + result_list[j].bd_matrix(ind) / Kfold;
            }

            algorithm_list[algorithm_index]->update_sparsity_level(support_size);
            algorithm_list[algorithm_index]->update_lambda_level(lambda);
            algorithm_list[algorithm_index]->update_beta_init(beta_init);
            algorithm_list[algorithm_index]->update_coef0_init(coef0_init);
            algorithm_list[algorithm_index]->update_bd_init(bd_init);
            algorithm_list[algorithm_index]->update_A_init(A_init, data.g_num);
            algorithm_list[algorithm_index]->fit(data.x, data.y, data.weight, data.g_index, 
                                                data.g_size, data.n, data.p, data.g_num);

            beta_matrix(ind) = algorithm_list[algorithm_index]->get_beta();
            coef0_matrix(ind) = algorithm_list[algorithm_index]->get_coef0();
            train_loss_matrix(ind) = algorithm_list[algorithm_index]->get_train_loss();
            ic_matrix(ind) = metric->ic(data.n, data.M, data.g_num, algorithm_list[algorithm_index]);
            effective_number_matrix(ind) = algorithm_list[algorithm_index]->get_effective_number();
        }
    }

    VectorXd beta;
    if (screening_size < 0)
    {
        beta = beta_matrix(min_loss_index);
    }
    else
    {
        beta = VectorXd::Zero(model_size);
        beta(screening_A) = beta_matrix(min_loss_index);
    }

    delete metric;
    for (int i = 0; i < algorithm_list_size; i++)
    {
        delete algorithm_list[i];
    }
    SPDLOG_DEBUG("SCOPE end!");

    return make_tuple(beta,
                     train_loss_matrix(min_loss_index),
                     test_loss_sum(min_loss_index),
                     ic_matrix(min_loss_index));
}

int main() {
    cout << "=== ScopeCpp Pure C++ Version ===" << endl;
    
    // 初始化日志（提供默认参数）
    init_spdlog(2, 2, "scope.log");
    
    // 创建简单的测试数据: y = X * beta + noise
    int n_samples = 100;
    int n_features = 10;
    
    // 生成随机数据
    MatrixXd X = MatrixXd::Random(n_samples, n_features);
    VectorXd true_beta = VectorXd::Zero(n_features);
    true_beta(0) = 1.5;
    true_beta(2) = -0.8;
    true_beta(5) = 2.0;
    
    VectorXd y = X * true_beta + 0.1 * VectorXd::Random(n_samples);
    
    // 构造数据矩阵 [X, y]
    MatrixXd* data = new MatrixXd(n_samples, n_features + 1);
    data->leftCols(n_features) = X;
    data->col(n_features) = y;
    
    // 设置模型
    UniversalModel model;
    model.set_loss_of_model(LinearRegressionModel::loss);
    model.set_gradient_user_defined(LinearRegressionModel::loss_and_gradient);
    model.set_slice_by_sample(LinearRegressionModel::slice_by_sample);
    // 不设置自定义 deleter，使用默认的
    
    // 设置优化器 - 使用 lambda 包装
    ConvexSolver solver = [](
        function<double(VectorXd const&, const MatrixXd*)> loss_fn,
        function<pair<double, VectorXd>(const VectorXd&, const MatrixXd*)> value_and_grad,
        const VectorXd& init_para,
        const VectorXi& active_indices,
        const MatrixXd* data
    ) -> pair<double, VectorXd> {
        return GradientDescentSolver::solve(loss_fn, value_and_grad, init_para, 
                                           active_indices, data, 1000, 1e-6, 0.01);
    };
    
    // 设置参数
    int model_size = n_features;
    int sample_size = n_samples;
    int aux_para_size = 0;
    int max_iter = 30;
    int exchange_num = 5;
    int path_type = 1;
    bool is_greedy = true;
    bool use_hessian = false;
    bool is_dynamic_exchange_num = true;
    bool is_warm_start = true;
    int ic_type = 1;  // AIC
    double ic_coef = 1.0;
    int Kfold = 1;  // 不使用 CV
    
    // 创建搜索序列
    VectorXi sequence(5);
    VectorXd lambda_seq(5);
    for (int i = 0; i < 5; i++) {
        sequence(i) = 2 * (i + 1);  // 支持集大小：2, 4, 6, 8, 10
        lambda_seq(i) = 0.01 * (5 - i);  // lambda 递减
    }
    
    int s_min = 2, s_max = 10;
    int screening_size = -1;  // 不使用筛选
    
    // 分组信息：每个变量单独一组
    VectorXi g_index = VectorXi::LinSpaced(n_features, 0, n_features - 1);
    VectorXi always_select = VectorXi::Zero(0);
    
    int thread = 1;
    int splicing_type = 0;
    int sub_search = 0;
    VectorXi cv_fold_id = VectorXi::Zero(n_samples);
    VectorXi A_init = VectorXi::Zero(0);
    VectorXd beta_init = VectorXd::Zero(model_size);
    VectorXd coef0_init = VectorXd::Zero(0);
    
    // 运行算法
    cout << "Running SCOPE algorithm..." << endl;
    VectorXd result_beta;
    double train_loss, test_loss, ic;
    
    tie(result_beta, train_loss, test_loss, ic) = run_scope(
        data, model, solver, model_size, sample_size, aux_para_size, max_iter,
        exchange_num, path_type, is_greedy, use_hessian, is_dynamic_exchange_num,
        is_warm_start, ic_type, ic_coef, Kfold, sequence, lambda_seq, s_min, s_max,
        screening_size, g_index, always_select, thread, splicing_type, sub_search,
        cv_fold_id, A_init, beta_init, coef0_init
    );
    
    // 输出结果
    cout << "\n=== Results ===" << endl;
    cout << "True beta (non-zero): [0]=1.5, [2]=-0.8, [5]=2.0" << endl;
    cout << "Estimated beta:" << endl;
    for (int i = 0; i < result_beta.size(); i++) {
        if (abs(result_beta(i)) > 1e-6) {
            cout << "  beta[" << i << "] = " << result_beta(i) << endl;
        }
    }
    cout << "Train loss: " << train_loss << endl;
    cout << "IC: " << ic << endl;
    
    // data 由 UniversalData 内部的 shared_ptr 管理，会自动释放
    
    cout << "\nProgram completed successfully!" << endl;
    
    return 0;
}
