#pragma once

#include <Eigen/Eigen>


#include <algorithm>
#include <cfloat>


#include "Data.h"
#include "utilities.h"

using namespace std;
using namespace Eigen;


Eigen::VectorXi screening(Data &data, std::vector<Algorithm *> algorithm_list,
                          int screening_size, int &beta_size, double lambda, Eigen::VectorXi &A_init) {
    int n = data.n;
    int M = data.M;
    int g_num = data.g_num;

    Eigen::VectorXi g_size = data.g_size;
    Eigen::VectorXi g_index = data.g_index;
    Eigen::VectorXi always_select = algorithm_list[0]->always_select;

    Eigen::VectorXi screening_A(screening_size);
    Eigen::VectorXd coef_norm = Eigen::VectorXd::Zero(g_num);

    Eigen::VectorXd beta_init;
    Eigen::VectorXd coef0_init;
    Eigen::VectorXd bd_init;

    for (int i = 0; i < g_num; i++) {
        int p_tmp = g_size(i);
        Eigen::VectorXi index = Eigen::VectorXi::LinSpaced(p_tmp, g_index(i), g_index(i) + p_tmp - 1);
        UniversalData x_tmp = X_seg(data.x, n, index, algorithm_list[0]->model_type);
        Eigen::VectorXi g_index_tmp = Eigen::VectorXi::LinSpaced(p_tmp, 0, p_tmp - 1);
        Eigen::VectorXi g_size_tmp = Eigen::VectorXi::Ones(p_tmp);
        coef_set_zero(p_tmp, M, beta_init, coef0_init);

        algorithm_list[0]->update_sparsity_level(p_tmp);
        algorithm_list[0]->update_lambda_level(lambda);
        algorithm_list[0]->update_beta_init(beta_init);
        algorithm_list[0]->update_bd_init(bd_init);
        algorithm_list[0]->update_coef0_init(coef0_init);
        algorithm_list[0]->update_A_init(A_init, p_tmp);
        algorithm_list[0]->fit(x_tmp, data.y, data.weight, g_index_tmp, g_size_tmp, n, p_tmp, p_tmp);

        Eigen::VectorXd beta = algorithm_list[0]->beta;
        coef_norm(i) = beta.squaredNorm() / p_tmp;
    }

    // keep always_select in active_set
    slice_assignment(coef_norm, always_select, DBL_MAX);
    screening_A = max_k(coef_norm, screening_size);

    // data after screening
    Eigen::VectorXi new_g_index(screening_size);
    Eigen::VectorXi new_g_size(screening_size);

    int new_p = 0;
    for (int i = 0; i < screening_size; i++) {
        new_p += g_size(screening_A(i));
        new_g_size(i) = g_size(screening_A(i));
    }

    new_g_index(0) = 0;
    for (int i = 0; i < screening_size - 1; i++) {
        new_g_index(i + 1) = new_g_index(i) + g_size(screening_A(i));
    }

    Eigen::VectorXi screening_A_ind = find_ind(screening_A, g_index, g_size, beta_size, g_num); 
    UniversalData x_A = X_seg(data.x, 0, screening_A_ind, 0);

    Eigen::VectorXd new_x_mean, new_x_norm;
    slice(data.x_mean, screening_A_ind, new_x_mean);
    slice(data.x_norm, screening_A_ind, new_x_norm);

    data.x = x_A;
    data.x_mean = new_x_mean;
    data.x_norm = new_x_norm;
    data.p = new_p;
    data.g_num = screening_size;
    data.g_index = new_g_index;
    data.g_size = new_g_size;
    beta_size = new_p;

    if (always_select.size() != 0) {
        Eigen::VectorXi new_always_select(always_select.size());
        int j = 0;
        for (int i = 0; i < always_select.size(); i++) {
            while (always_select(i) != screening_A(j)) j++;
            new_always_select(i) = j;
        }
        int algorithm_list_size = static_cast<int>(algorithm_list.size());
        for (int i = 0; i < algorithm_list_size; i++) {
            algorithm_list[i]->always_select = new_always_select;
        }
    }

    return screening_A_ind;
}


