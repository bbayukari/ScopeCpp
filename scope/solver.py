from sklearn.model_selection import KFold
from .other_solver import BaseSolver
from sklearn.base import BaseEstimator
import numpy as np
import nlopt
from . import _scope
import math



def convex_solver_nlopt(
    loss_fn,
    value_and_grad,
    params,
    optim_variable_set,
    data,
):
    best_loss = math.inf
    best_params = None

    def cache_opt_fn(x, grad):
        nonlocal best_loss, best_params
        params[optim_variable_set] = x  # update the nonlocal variable: params
        if grad.size > 0:
            loss, full_grad = value_and_grad(params, data)
            grad[:] = full_grad[optim_variable_set]
        else:
            loss = loss_fn(params, data)
        if loss < best_loss:
            best_loss = loss
            best_params = np.copy(x)
        return loss

    nlopt_solver = nlopt.opt(nlopt.LD_LBFGS, optim_variable_set.size)
    nlopt_solver.set_min_objective(cache_opt_fn)

    try:
        params[optim_variable_set] = nlopt_solver.optimize(params[optim_variable_set])
        return nlopt_solver.last_optimum_value(), params
    except RuntimeError:
        params[optim_variable_set] = best_params
        return best_loss, params


class ScopeSolver(BaseEstimator):
    def __init__(
        self,
        dimensionality,
        sparsity=None,
        sample_size=1,
        *,
        always_select=[],
        numeric_solver=convex_solver_nlopt,
        max_iter=20,
        ic_type="aic",
        ic_coef=1.0,
        cv=1,
        split_method=None,
        deleter=None,
        cv_fold_id=None,
        group=None,
        warm_start=True,
        important_search=128,
        screening_size=-1,
        max_exchange_num=5,
        is_dynamic_max_exchange_num=True,
        greedy=True,
        splicing_type="halve",
        path_type="seq",
        gs_lower_bound=None,
        gs_upper_bound=None,
        thread=1,
        random_state=None,
        console_log_level="off",
        file_log_level="off",
        log_file_name="logs/scope.log",
    ):
        self.model = _scope.UniversalModel()
        self.dimensionality = dimensionality
        self.sparsity = sparsity
        self.sample_size = sample_size

        self.always_select = always_select
        self.numeric_solver = numeric_solver
        self.max_iter = max_iter
        self.ic_type = ic_type
        self.ic_coef = ic_coef
        self.cv = cv
        self.split_method = split_method
        self.deleter = deleter
        self.cv_fold_id = cv_fold_id
        self.group = group
        self.warm_start = warm_start
        self.important_search = important_search
        self.screening_size = screening_size
        self.max_exchange_num = max_exchange_num
        self.is_dynamic_max_exchange_num = is_dynamic_max_exchange_num
        self.greedy = greedy
        self.splicing_type = splicing_type
        self.path_type = path_type
        self.gs_lower_bound = gs_lower_bound
        self.gs_upper_bound = gs_upper_bound
        self.thread = thread
        self.random_state = random_state
        self.console_log_level = console_log_level
        self.file_log_level = file_log_level
        self.log_file_name = log_file_name

    def get_config(self, deep=True):
        return super().get_params(deep)

    def set_config(self, **params):
        return super().set_params(**params)
    
    def get_estimated_params(self):
        return self.params
    
    def get_support(self):
        return self.support_set

    @staticmethod
    def _set_log_level(console_log_level, file_log_level, log_file_name):
        # log level
        log_level_dict = {
            "off": 6,
            "error": 4,
            "warning": 3,
            "debug": 1,
        }
        console_log_level = console_log_level.lower()
        file_log_level = file_log_level.lower()
        if (
            console_log_level not in log_level_dict
            or file_log_level not in log_level_dict
        ):
            raise ValueError(
                "console_log_level and file_log_level must be in 'off', 'error', 'warning', 'debug'"
            )
        console_log_level = log_level_dict[console_log_level]
        file_log_level = log_level_dict[file_log_level]
        # log file name
        if not isinstance(log_file_name, str):
            raise ValueError("log_file_name must be a string")

        _scope.init_spdlog(console_log_level, file_log_level, log_file_name)

    def solve(
        self,
        objective,
        data=(),
        init_support_set=None,
        init_params=None,
        gradient=None,
    ):
        ScopeSolver._set_log_level(
            self.console_log_level, self.file_log_level, self.log_file_name
        )

        if not isinstance(data, tuple):
            data = (data,)

        p = self.dimensionality
        BaseSolver._check_positive_integer(p, "dimensionality")

        n = self.sample_size
        BaseSolver._check_positive_integer(n, "sample_size")

        BaseSolver._check_non_negative_integer(self.max_iter, "max_iter")

        # max_exchange_num
        BaseSolver._check_positive_integer(self.max_exchange_num, "max_exchange_num")

        # ic_type
        information_criterion_dict = {
            "aic": 1,
            "bic": 2,
            "gic": 3,
            "ebic": 4,
        }
        if self.ic_type not in information_criterion_dict.keys():
            raise ValueError('ic_type should be "aic", "bic", "ebic" or "gic"')
        ic_type = information_criterion_dict[self.ic_type]

        # group
        if self.group is None:
            group = np.arange(p, dtype="int32")
            group_num = p  # len(np.unique(group))
        else:
            group = np.array(self.group)
            if group.ndim > 1:
                raise ValueError("Group should be an 1D array of integers.")
            if group.size != p:
                raise ValueError(
                    "The length of group should be equal to dimensionality."
                )
            group_num = len(np.unique(group))
            if group[0] != 0:
                raise ValueError("Group should start from 0.")
            if any(group[1:] - group[:-1] < 0):
                raise ValueError("Group should be an incremental integer array.")
            if not group_num == max(group) + 1:
                raise ValueError("There is a gap in group.")
            group = np.array(
                [np.where(group == i)[0][0] for i in range(group_num)], dtype="int32"
            )

        # always_select
        always_select = np.unique(np.array(self.always_select, dtype="int32"))
        if always_select.size > 0 and (
            always_select[0] < 0 or always_select[-1] >= group_num
        ):
            raise ValueError("always_select should be between 0 and dimensionality.")

        # default sparsity level
        force_min_sparsity = always_select.size
        default_max_sparsity = max(
            force_min_sparsity,
            group_num
            if group_num <= 5
            else int(group_num / np.log(np.log(group_num)) / np.log(group_num)),
        )

        # path_type
        if self.path_type == "seq":
            path_type = 1
            gs_lower_bound, gs_upper_bound = 0, 0
            if self.sparsity is None:
                sparsity = np.arange(
                    force_min_sparsity,
                    default_max_sparsity + 1,
                    dtype="int32",
                )
            else:
                sparsity = np.unique(np.array(self.sparsity, dtype="int32"))
                if sparsity.size == 0:
                    raise ValueError("sparsity should not be empty.")
                if sparsity[0] < force_min_sparsity or sparsity[-1] > group_num:
                    raise ValueError(
                        "All sparsity should be between 0 (when `always_select` is default) and dimensionality (when `group` is default)."
                    )
        elif self.path_type == "gs":
            path_type = 2
            sparsity = np.array([0], dtype="int32")
            if self.gs_lower_bound is None:
                gs_lower_bound = force_min_sparsity
            else:
                BaseSolver._check_non_negative_integer(
                    self.gs_lower_bound, "gs_lower_bound"
                )
                gs_lower_bound = self.gs_lower_bound

            if self.gs_upper_bound is None:
                gs_upper_bound = default_max_sparsity
            else:
                BaseSolver._check_non_negative_integer(
                    self.gs_upper_bound, "gs_upper_bound"
                )
                gs_upper_bound = self.gs_upper_bound

            if gs_lower_bound < force_min_sparsity or gs_upper_bound > group_num:
                raise ValueError(
                    "gs_lower_bound and gs_upper_bound should be between 0 (when `always_select` is default) and dimensionality (when `group` is default)."
                )
            if gs_lower_bound > gs_upper_bound:
                raise ValueError("gs_upper_bound should be larger than gs_lower_bound.")
        else:
            raise ValueError("path_type should be 'seq' or 'gs'")

        # screening_size
        if self.screening_size == -1:
            screening_size = -1
        elif self.screening_size == 0:
            screening_size = max(sparsity[-1], gs_upper_bound, default_max_sparsity)
        else:
            screening_size = self.screening_size
            if screening_size > group_num or screening_size < max(
                sparsity[-1], gs_upper_bound
            ):
                raise ValueError(
                    "screening_size should be between max(sparsity) and dimensionality."
                )

        # thread
        BaseSolver._check_non_negative_integer(self.thread, "thread")

        # splicing_type
        if self.splicing_type == "halve":
            splicing_type = 0
        elif self.splicing_type == "taper":
            splicing_type = 1
        else:
            raise ValueError('splicing_type should be "halve" or "taper".')

        # important_search
        BaseSolver._check_non_negative_integer(
            self.important_search, "important_search"
        )

        # cv
        BaseSolver._check_positive_integer(self.cv, "cv")
        if self.cv > n:
            raise ValueError("cv should not be greater than sample_size")
        if self.cv > 1:
            if len(data) == 0 and self.split_method is None:
                data = (np.arange(n),)
                if cpp:
                    self.split_method = lambda data, index: index
                else:
                    self.split_method = lambda data, index: (index,)
            if self.split_method is None:
                raise ValueError("split_method should be provided when cv > 1")
            self.model.set_slice_by_sample(self.split_method)
            self.model.set_deleter(self.deleter)
            if self.cv_fold_id is None:
                kf = KFold(
                    n_splits=self.cv, shuffle=True, random_state=self.random_state
                ).split(np.zeros(n))

                self.cv_fold_id = np.zeros(n)
                for i, (_, fold_id) in enumerate(kf):
                    self.cv_fold_id[fold_id] = i
            else:
                self.cv_fold_id = np.array(self.cv_fold_id, dtype="int32")
                if self.cv_fold_id.ndim > 1:
                    raise ValueError("group should be an 1D array of integers.")
                if self.cv_fold_id.size != n:
                    raise ValueError(
                        "The length of group should be equal to X.shape[0]."
                    )
                if len(set(self.cv_fold_id)) != self.cv:
                    raise ValueError(
                        "The number of different masks should be equal to `cv`."
                    )
        else:
            self.cv_fold_id = np.array([], dtype="int32")

        # init_support_set
        if init_support_set is None:
            init_support_set = np.array([], dtype="int32")
        else:
            init_support_set = np.array(init_support_set, dtype="int32")
            if init_support_set.ndim > 1:
                raise ValueError(
                    "The initial active set should be " "an 1D array of integers."
                )
            if init_support_set.min() < 0 or init_support_set.max() >= p:
                raise ValueError("init_support_set contains wrong index.")

        # init_params
        if init_params is None:
            init_params = np.zeros(p, dtype=float)
        else:
            init_params = np.array(init_params, dtype=float)
            if init_params.shape != (p,):
                raise ValueError(
                    "The length of init_params must match `dimensionality`!"
                )

        # set optimization objective
        if len(data) == 1:
            data = data[0]
        loss_fn = self.__set_objective_cpp(objective, gradient)

        result = _scope.pywrap_Universal(
            data,
            self.model,
            self.numeric_solver,
            p,
            n,
            0,
            self.max_iter,
            self.max_exchange_num,
            path_type,
            self.greedy,
            False,
            self.is_dynamic_max_exchange_num,
            self.warm_start,
            ic_type,
            self.ic_coef,
            self.cv,
            sparsity,
            np.array([0.0]),
            gs_lower_bound,
            gs_upper_bound,
            screening_size,
            group,
            always_select,
            self.thread,
            splicing_type,
            self.important_search,
            self.cv_fold_id,
            init_support_set,
            init_params,
            np.zeros(0),
        )

        self.params = np.array(result[0])
        self.support_set = np.sort(np.nonzero(self.params)[0])
        self.cv_train_loss = result[1] if self.cv == 1 else 0.0
        self.cv_test_loss = result[2] if self.cv == 1 else 0.0
        self.information_criterion = result[3]
        self.value_of_objective = loss_fn(self.params, data)

        return self.params

    def get_result(self):
        return {
            "params": self.params,
            "support_set": self.support_set,
            "value_of_objective": self.value_of_objective,
            "cv_train_loss": self.cv_train_loss,
            "cv_test_loss": self.cv_test_loss,
            "information_criterion": self.information_criterion,
        }
    
    def __set_objective_cpp(self, objective, gradient):
        if objective.__code__.co_argcount == 1:
            loss_ = lambda params, data: objective(params)
        else:
            loss_ = lambda params, data: objective(params, data)

        if gradient.__code__.co_argcount == 1:
            grad_ = lambda params, data: (loss_(params, data), gradient(params))
        else:
            grad_ = lambda params, data: (loss_(params, data), gradient(params, data))


        self.model.set_loss_of_model(loss_)
        self.model.set_gradient_user_defined(grad_)
        return loss_

    

