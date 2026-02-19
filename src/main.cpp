// main.cpp - SCOPE 算法核心入口函数
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
