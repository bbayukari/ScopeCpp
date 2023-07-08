from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold
import numpy as np
import math
import nlopt


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




class BaseSolver(BaseEstimator):
    def __init__(
        self,
        dimensionality,
        sparsity=None,
        sample_size=1,
        *,
        always_select=[],
        numeric_solver=convex_solver_nlopt,
        max_iter=100,
        group=None,
        ic_type="aic",
        ic_coef=1.0,
        metric_method=None,
        cv=1,
        cv_fold_id=None,
        split_method=None,
        random_state=None,
    ):
        self.dimensionality = dimensionality
        self.sample_size = sample_size
        self.sparsity = sparsity
        self.always_select = always_select
        self.max_iter = max_iter
        self.group = group
        self.ic_type = ic_type
        self.ic_coef = ic_coef
        self.metric_method = metric_method
        self.cv = cv
        self.cv_fold_id = cv_fold_id
        self.split_method = split_method
        self.random_state = random_state
        self.numeric_solver = numeric_solver

    def get_config(self, deep=True):
        return super().get_params(deep)

    def set_config(self, **params):
        return super().set_params(**params)

    def solve(
        self,
        objective,
        data=(),
        gradient=None,
        init_support_set=None,
        init_params=None,
    ):
        if not isinstance(data, tuple):
            data = (data,)
            
        BaseSolver._check_positive_integer(self.dimensionality, "dimensionality")
        BaseSolver._check_positive_integer(self.sample_size, "sample_size")
        BaseSolver._check_non_negative_integer(self.max_iter, "max_iter")
        
        # group
        if self.group is None:
            self.group = np.arange(self.dimensionality, dtype="int32")
            group_num = self.dimensionality  # len(np.unique(group))
        else:
            self.group = np.array(self.group)
            if self.group.ndim > 1:
                raise ValueError("Group should be an 1D array of integers.")
            if self.group.size != self.dimensionality:
                raise ValueError(
                    "The length of group should be equal to dimensionality."
                )
            group_num = len(np.unique(self.group))
            if self.group[0] != 0:
                raise ValueError("Group should start from 0.")
            if any(self.group[1:] - self.group[:-1] < 0):
                raise ValueError("Group should be an incremental integer array.")
            if not group_num == max(self.group) + 1:
                raise ValueError("There is a gap in group.")
            

        # always_select
        self.always_select = np.unique(np.array(self.always_select, dtype="int32"))
        if self.always_select.size > 0 and (
            self.always_select[0] < 0 or self.always_select[-1] >= group_num
        ):
            raise ValueError("always_select should be between 0 and dimensionality.")

        # default sparsity level
        force_min_sparsity = self.always_select.size
        default_max_sparsity = max(
            force_min_sparsity,
            group_num
            if group_num <= 5
            else int(
                group_num
                / np.log(np.log(group_num))
                / np.log(group_num)
            ),
        )

        # sparsity
        if self.sparsity is None:
            self.sparsity = np.arange(
                force_min_sparsity,
                default_max_sparsity + 1,
                dtype="int32",
            )
        else:
            self.sparsity = np.unique(np.array(self.sparsity, dtype="int32"))
            if (
                self.sparsity[0] < force_min_sparsity
                or self.sparsity[-1] > group_num
            ):
                raise ValueError(
                    "All sparsity should be between 0 (when `always_select` is default) and dimensionality."
                )

        BaseSolver._check_positive_integer(self.cv, "cv")
        if self.cv == 1:
            if self.ic_type not in ["aic", "bic", "gic", "ebic"]:
                raise ValueError(
                    "ic_type should be one of ['aic', 'bic', 'gic','ebic']."
                )
            if self.ic_coef <= 0:
                raise ValueError("ic_coef should be positive.")
        else:
            if self.cv > self.sample_size:
                raise ValueError("cv should not be greater than sample_size")
            if len(data) == 0 and self.split_method is None:
                data = (np.arange(self.sample_size),)
                self.split_method = lambda data, index: (index,)
            if self.split_method is None:
                raise ValueError("split_method should be provided when cv > 1")
            if self.cv_fold_id is None:
                kf = KFold(
                    n_splits=self.cv, shuffle=True, random_state=self.random_state
                ).split(np.zeros(self.sample_size))

                self.cv_fold_id = np.zeros(self.sample_size)
                for i, (_, fold_id) in enumerate(kf):
                    self.cv_fold_id[fold_id] = i
            else:
                self.cv_fold_id = np.array(self.cv_fold_id, dtype="int32")
                if self.cv_fold_id.ndim > 1:
                    raise ValueError("group should be an 1D array of integers.")
                if self.cv_fold_id.size != self.sample_size:
                    raise ValueError(
                        "The length of group should be equal to X.shape[0]."
                    )
                if len(set(self.cv_fold_id)) != self.cv:
                    raise ValueError(
                        "The number of different masks should be equal to `cv`."
                    )

        if init_support_set is None:
            init_support_set = np.array([], dtype="int32")
        else:
            init_support_set = np.array(init_support_set, dtype="int32")
            if init_support_set.ndim > 1:
                raise ValueError(
                    "The initial active set should be " "an 1D array of integers."
                )
            if (
                init_support_set.min() < 0
                or init_support_set.max() >= self.dimensionality
            ):
                raise ValueError("init_support_set contains wrong index.")

        if init_params is None:
            init_params = np.zeros(self.dimensionality, dtype=float)
        else:
            init_params = np.array(init_params, dtype=float)
            if init_params.shape != (self.dimensionality,):
                raise ValueError(
                    "The length of init_params must match `dimensionality`!"
                )

        loss_, grad_ = BaseSolver._set_objective(objective, gradient)
        loss_fn = lambda params, data: loss_(params, data).item()

        def value_and_grad(params, data):
            value, grad = grad_(params, data)
            return value.item(), np.array(grad)

        if self.cv == 1:
            is_first_loop: bool = True
            for s in self.sparsity:
                init_params, init_support_set = self._solve(
                    s, loss_fn, value_and_grad, init_support_set, init_params, data
                )  ## warm start: use results of previous sparsity as initial value
                value_of_objective = loss_fn(init_params, data)
                eval = self._metric(
                    value_of_objective,
                    self.ic_type,
                    s,
                    self.sample_size,
                )
                if is_first_loop or eval < self.eval_objective:
                    is_first_loop = False
                    self.params = init_params
                    self.support_set = init_support_set
                    self.value_of_objective = value_of_objective
                    self.eval_objective = eval

        else:  # self.cv > 1
            cv_eval = {s: 0.0 for s in self.sparsity}
            cache_init_support_set = {}
            cache_init_params = {}
            for s in self.sparsity:
                for i in range(self.cv):
                    train_index = np.where(self.cv_fold_id != i)[0]
                    test_index = np.where(self.cv_fold_id == i)[0]
                    init_params, init_support_set = self._solve(
                        s,
                        loss_fn,
                        value_and_grad,
                        init_support_set,
                        init_params,
                        self.split_method(data, train_index),
                    )  ## warm start: use results of previous sparsity as initial value
                    cv_eval[s] += loss_fn(
                        init_params, self.split_method(data, test_index)
                    )
                cache_init_support_set[s] = init_support_set
                cache_init_params[s] = init_params
            best_sparsity = min(cv_eval, key=cv_eval.get)
            self.params, self.support_set = self._solve(
                best_sparsity,
                loss_fn,
                value_and_grad,
                cache_init_support_set[best_sparsity],
                cache_init_params[best_sparsity],
                data,
            )
            self.value_of_objective = loss_fn(self.params, data)
            self.eval_objective = cv_eval[best_sparsity]

        self.support_set = np.sort(self.support_set)
        return self.params

    @staticmethod
    def _set_objective(objective, gradient):
        if objective.__code__.co_argcount == 1:
            loss_ = lambda params, data: objective(params)
        else:
            loss_ = lambda params, data: objective(params, *data)

        if gradient.__code__.co_argcount == 1:
            grad_ = lambda params, data: (loss_(params, data), gradient(params))
        else:
            grad_ = lambda params, data: (loss_(params, data), gradient(params, *data))

        return loss_, grad_

    def _metric(
        self,
        value_of_objective: float,
        method: str,
        effective_params_num: int,
        train_size: int,
    ) -> float:
        if self.metric_method is not None:
            return self.metric_method(
                value_of_objective,
                self.dimensionality,
                effective_params_num,
                train_size,
            )

        if method == "aic":
            return 2 * value_of_objective + 2 * effective_params_num
        elif method == "bic":
            return (
                value_of_objective
                if train_size <= 1.0
                else 2 * value_of_objective
                + self.ic_coef * effective_params_num * np.log(train_size)
            )
        elif method == "gic":
            return (
                value_of_objective
                if train_size <= 1.0
                else 2 * value_of_objective
                + self.ic_coef
                * effective_params_num
                * np.log(np.log(train_size))
                * np.log(self.dimensionality)
            )
        elif method == "ebic":
            return 2 * value_of_objective + self.ic_coef * effective_params_num * (
                np.log(train_size) + 2 * np.log(self.dimensionality)
            )
        else:
            return value_of_objective

    def _solve(
        self,
        sparsity,
        loss_fn,
        value_and_grad,
        init_support_set,
        init_params,
        data,
    ):
        if sparsity == 0:
            return np.zeros(self.dimensionality), np.array([], dtype=int)
        if sparsity < self.always_select.size:
            raise ValueError(
                "The number of always selected variables is larger than the sparsity."
            )
        group_num = len(np.unique(self.group))
        group_indices = [np.where(self.group == i)[0] for i in range(group_num)]

        if (
            math.comb(
                group_num - self.always_select.size,
                sparsity - self.always_select.size,
            )
            > self.max_iter
        ):
            raise ValueError(
                "The number of subsets is too large, please reduce the sparsity, dimensionality or increase max_iter."
            )

        def all_subsets(p: int, s: int, always_select: np.ndarray = np.zeros(0)):
            universal_set = np.setdiff1d(np.arange(group_num), always_select)
            p = p - always_select.size
            s = s - always_select.size

            def helper(start: int, s: str, curr_selection: np.ndarray):
                if s == 0:
                    yield curr_selection
                else:
                    for i in range(start, p - s + 1):
                        yield from helper(
                            i + 1, s - 1, np.append(curr_selection, universal_set[i])
                        )

            yield from helper(0, s, always_select)

        result = {"params": None, "support_set": None, "value_of_objective": math.inf}
        params = init_params.copy()
        for support_set_group in all_subsets(
            group_num, sparsity, self.always_select
        ):
            support_set = np.concatenate([group_indices[i] for i in support_set_group])
            inactive_set = np.ones_like(init_params, dtype=bool)
            inactive_set[support_set] = False
            params[inactive_set] = 0.0
            params[support_set] = init_params[support_set]
            loss, params = self._numeric_solver(
                loss_fn, value_and_grad, params, support_set, data
            )
            if loss < result["value_of_objective"]:
                result["params"] = params.copy()
                result["support_set"] = support_set
                result["value_of_objective"] = loss

        return result["params"], result["support_set"]

    def _numeric_solver(
        self,
        loss_fn,
        value_and_grad,
        params,
        optim_variable_set,
        data,
    ):
        if not isinstance(params, np.ndarray) or params.ndim != 1:
            raise ValueError("params should be a 1D np.ndarray.")
        if (
            not isinstance(optim_variable_set, np.ndarray)
            or optim_variable_set.ndim != 1
        ):
            raise ValueError("optim_variable_set should be a 1D np.ndarray.")

        if optim_variable_set.size == 0:
            return loss_fn(params, data)

        return self.numeric_solver(
            loss_fn, value_and_grad, params, optim_variable_set, data
        )

    def get_result(self):
        return {
            "params": self.params,
            "support_set": self.support_set,
            "value_of_objective": self.value_of_objective,
            "eval_objective": self.eval_objective,
        }
    
    def get_estimated_params(self):
        return self.params
    
    def get_support(self):
        return self.support_set

    @staticmethod
    def _check_positive_integer(var, name: str):
        if not isinstance(var, int) or var <= 0:
            raise ValueError("{} should be an positive integer.".format(name))

    @staticmethod
    def _check_non_negative_integer(var, name: str):
        if not isinstance(var, int) or var < 0:
            raise ValueError("{} should be an non-negative integer.".format(name))


class HTPSolver(BaseSolver):
    def __init__(
        self,
        dimensionality,
        sparsity=None,
        sample_size=1,
        *,
        always_select=[],
        step_size=0.005,
        numeric_solver=convex_solver_nlopt,
        max_iter=100,
        group=None,
        ic_type="aic",
        ic_coef=1.0,
        metric_method=None,
        cv=1,
        cv_fold_id=None,
        split_method=None,
        random_state=None,
    ):
        super().__init__(
            dimensionality=dimensionality,
            sparsity=sparsity,
            sample_size=sample_size,
            always_select=always_select,
            numeric_solver=numeric_solver,
            max_iter=max_iter,
            group=group,
            ic_type=ic_type,
            ic_coef=ic_coef,
            metric_method=metric_method,
            cv=cv,
            cv_fold_id=cv_fold_id,
            split_method=split_method,
            random_state=random_state,
        )
        self.step_size = step_size

    def _solve(
        self,
        sparsity,
        loss_fn,
        value_and_grad,
        init_support_set,
        init_params,
        data,
    ):
        if sparsity <= self.always_select.size:
            return super()._solve(
                sparsity,
                loss_fn,
                value_and_grad,
                init_support_set,
                init_params,
                data,
            )
        # init
        params = init_params
        old_suppport_group_tuple = None
        group_num = len(np.unique(self.group))
        group_indices = [np.where(self.group == i)[0] for i in range(group_num)]

        for n_iters in range(self.max_iter):
            # S1: gradient descent
            params_bias = params - self.step_size * value_and_grad(params, data)[1]
            # S2: Gradient Hard Thresholding
            score = np.array([np.sum(np.square(params_bias[group_indices[i]])) for i in range(group_num)])
            score[self.always_select] = np.inf
            support_new_group = np.argpartition(score, -sparsity)[-sparsity:]
            support_new_group_tuple = tuple(np.sort(support_new_group))
            # terminating condition
            if support_new_group_tuple == old_suppport_group_tuple:
                break
            # S3: debise
            params = np.zeros(self.dimensionality)
            support_new = np.concatenate([group_indices[i] for i in support_new_group])
            params[support_new] = params_bias[support_new]
            loss, params = self._numeric_solver(
                loss_fn, value_and_grad, params, support_new, data
            )
            old_suppport_group_tuple = support_new_group_tuple

        self.n_iters = n_iters       
        return params, np.concatenate([group_indices[i] for i in support_new_group_tuple])


class GraspSolver(BaseSolver):
    def _solve(
        self,
        sparsity,
        loss_fn,
        value_and_grad,
        init_support_set,
        init_params,
        data,
    ):
        if sparsity <= self.always_select.size:
            return super()._solve(
                sparsity,
                loss_fn,
                value_and_grad,
                init_support_set,
                init_params,
                data,
            )
        # init
        params = init_params
        old_suppport_tuple = None
        group_num = len(np.unique(self.group))
        group_indices = [np.where(self.group == i)[0] for i in range(group_num)]

        for n_iters in range(self.max_iter):
            # compute local gradient
            grad_values = value_and_grad(params, data)[1]
            score = np.array([np.sum(np.square(grad_values[group_indices[i]])) for i in range(group_num)])
            score[self.always_select] = np.inf

            # identify directions
            if 2 * sparsity < group_num:
                Omega = [
                    idx
                    for idx in np.argpartition(score, -2 * sparsity)[-2 * sparsity :]
                    if score[idx] != 0.0
                ]  # supp of top 2k largest absolute values of gradient
            else:
                Omega = np.nonzero(score)[0]  # supp(z)

            # merge supports
            support_new = np.concatenate([group_indices[i] for i in Omega])
            support_new = np.unique(np.append(support_new, params.nonzero()[0]))
            suppport_tuple = tuple(np.sort(support_new))
            
            # terminating condition
            if suppport_tuple == old_suppport_tuple:
                break
            
            # minimize
            params_bias = np.zeros(self.dimensionality)
            params_bias[support_new] = params[support_new]
            _, params_bias = self._numeric_solver(
                loss_fn, value_and_grad, params_bias, support_new, data
            )

            # prune estimate
            score = np.array([np.sum(np.square(params_bias[group_indices[i]])) for i in range(group_num)])
            score[self.always_select] = np.inf
            support_set_group = np.argpartition(score, -sparsity)[-sparsity:]
            support_set = np.concatenate([group_indices[i] for i in support_set_group])
            params = np.zeros(self.dimensionality)
            params[support_set] = params_bias[support_set]

            old_suppport_tuple = suppport_tuple

        self.n_iters = n_iters
        return params, support_set
