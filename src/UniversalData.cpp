#include "UniversalData.h"
#include "utilities.h"
#include <iostream>
using namespace std;
using Eigen::Map;
using Eigen::Matrix;

UniversalData::UniversalData(Eigen::Index model_size, Eigen::Index sample_size, pybind11::object &data, UniversalModel *model, ConvexSolver convex_solver)
    : model(model), convex_solver(convex_solver), sample_size(sample_size), model_size(model_size), effective_size(model_size)
{
    this->effective_para_index = VectorXi::LinSpaced(model_size, 0, model_size - 1);
    this->data = shared_ptr<pybind11::object>(new pybind11::object(data));
}

UniversalData UniversalData::slice_by_para(const VectorXi &target_para_index)
{
    UniversalData tem(*this);
    tem.effective_para_index = this->effective_para_index(target_para_index);
    tem.effective_size = target_para_index.size();
    return tem;
}

UniversalData UniversalData::slice_by_sample(const VectorXi &target_sample_index)
{
    UniversalData tem(*this);
    tem.sample_size = target_sample_index.size();
    tem.data = shared_ptr<pybind11::object>(new pybind11::object(model->slice_by_sample(*data, target_sample_index)), model->deleter);
    return tem;
}

Eigen::Index UniversalData::cols() const
{
    return effective_size;
}

Eigen::Index UniversalData::rows() const
{
    return sample_size;
}

const VectorXi &UniversalData::get_effective_para_index() const
{
    return effective_para_index;
}

double UniversalData::loss(const VectorXd &effective_para)
{
    VectorXd complete_para = VectorXd::Zero(this->model_size);
    complete_para(this->effective_para_index) = effective_para;
    return model->loss(complete_para, *this->data);
}

double UniversalData::loss_and_gradient(const VectorXd &effective_para, VectorXd &gradient)
{
    double value = 0.0;
    VectorXd complete_para = VectorXd::Zero(this->model_size);
    complete_para(this->effective_para_index) = effective_para;

    if (model->gradient_user_defined)
    {
        // Note: using complete_para to store gradient isn't a good idea, just for saving memory
        tie(value, complete_para) = model->gradient_user_defined(complete_para, *this->data);
        gradient = complete_para(this->effective_para_index);
    }

    return value;
}


void UniversalData::init_para(VectorXd &effective_para)
{
    if (model->init_para)
    {
        VectorXd complete_para = VectorXd::Zero(this->model_size);
        complete_para(this->effective_para_index) = effective_para;
        complete_para = model->init_para(complete_para, *this->data, this->effective_para_index);
        effective_para = complete_para(this->effective_para_index);
    }
}

double UniversalData::optimize(VectorXd &effective_para)
{
    if (effective_para.size() == 0){
        return model->loss(VectorXd::Zero(this->model_size), *this->data);
    }
    auto value_and_grad = [this](const VectorXd &complete_para, pybind11::object data) -> pair<double, VectorXd>
    {
        if (this->model->gradient_user_defined)
        {
            return this->model->gradient_user_defined(complete_para, data);
        }
    };
    VectorXd complete_para = VectorXd::Zero(this->model_size);
    complete_para(this->effective_para_index) = effective_para;
    double loss;
    tie(loss, complete_para) = this->convex_solver(
        model->loss,
        value_and_grad,
        complete_para,
        this->effective_para_index,
        *this->data);
    effective_para = complete_para(this->effective_para_index);
    return loss;
}

void UniversalModel::set_loss_of_model(function<double(VectorXd const &, pybind11::object const &)> const &f)
{
    loss = f;
}

void UniversalModel::set_gradient_user_defined(function<pair<double, VectorXd>(VectorXd const &, pybind11::object const &)> const &f)
{
    gradient_user_defined = f;
}


void UniversalModel::set_slice_by_sample(function<pybind11::object(pybind11::object const &, VectorXi const &)> const &f)
{
    slice_by_sample = f;
}

void UniversalModel::set_deleter(function<void(pybind11::object const &)> const &f)
{
    if (f)
    {
        deleter = [f](pybind11::object const *p)
        { f(*p); delete p; };
    }
    else
    {
        deleter = [](pybind11::object const *p)
        { delete p; };
    }
}

void UniversalModel::set_init_params_of_sub_optim(function<VectorXd(VectorXd const &, pybind11::object const &, VectorXi const &)> const &f)
{
    init_para = f;
}
