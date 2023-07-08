# pragma once

#include <Eigen/Eigen>
#include <pybind11/pybind11.h>
#include <memory>
#include <utility>


#include <functional>

using Eigen::VectorXd;
using Eigen::VectorXi;
using Eigen::MatrixXd;
using std::function;
using std::pair;

using ConvexSolver = function<pair<double, VectorXd>(
    function <double(VectorXd const&, pybind11::object const&)>, // loss_fn
    function <pair<double, VectorXd>(const VectorXd&, pybind11::object)>, // value_and_grad
    const VectorXd&, // complete_para
    const VectorXi&, // effective_para_index
    pybind11::object const& // data
)>;

class UniversalModel;

class UniversalData {
private:
    UniversalModel* model;
    ConvexSolver convex_solver;
    Eigen::Index sample_size;
    Eigen::Index model_size; // length of complete_para
    VectorXi effective_para_index;// `complete_para[effective_para_index[i]]` is `effective_para[i]`
    Eigen::Index effective_size; // length of effective_para_index
    std::shared_ptr<pybind11::object> data; // statistic data from user 
public:
    UniversalData() = default;
    UniversalData(Eigen::Index model_size, Eigen::Index sample_size, pybind11::object& data, UniversalModel* model, ConvexSolver convex_solver);
    UniversalData slice_by_para(const VectorXi& target_para_index); // used in util func X_seg() and slice()

    Eigen::Index rows() const; // getter of sample_size
    Eigen::Index cols() const; // getter of effective_para
    const VectorXi& get_effective_para_index() const; // getter of effective_para_index, only used for log
    UniversalData slice_by_sample(const VectorXi& target_sample_index);
    double loss(const VectorXd& effective_para); // compute the loss with effective_para
    double loss_and_gradient(const VectorXd& effective_para, Eigen::VectorXd& gradient);          
    void init_para(VectorXd & effective_para);  // initialize para for primary_model_fit, default is not change.                                                                                        
    double optimize(VectorXd& effective_para);                
};

class UniversalModel{
    friend class UniversalData;
private:
    // size of para will be match for data
    function <double(VectorXd const& para, pybind11::object const& data)> loss;
    function <pair<double, VectorXd>(VectorXd const& para, pybind11::object const& data)> gradient_user_defined;
    function <pybind11::object(pybind11::object const& old_data, VectorXi const& target_sample_index)> slice_by_sample;
    function <void(pybind11::object const* p)> deleter = [](pybind11::object const* p) { delete p; };
    function <VectorXd(VectorXd & para, pybind11::object const& data, VectorXi const& active_para_index)> init_para = nullptr;

public:
    // register callback function
    void set_loss_of_model(function <double(VectorXd const&, pybind11::object const&)> const&);
    void set_gradient_user_defined(function <pair<double, VectorXd>(VectorXd const&, pybind11::object const&)> const&);
    void set_slice_by_sample(function <pybind11::object(pybind11::object const&, VectorXi const&)> const&);
    void set_deleter(function <void(pybind11::object const&)> const&);
    void set_init_params_of_sub_optim(function <VectorXd(VectorXd const&, pybind11::object const&, VectorXi const&)> const&);
};
