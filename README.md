# ScopeCpp - Pure C++ Implementation

A pure C++ implementation of the SCOPE (sparsity-constraint optimization via splicing iteration) algorithm for sparse variable selection in high-dimensional statistical models.

## Overview

ScopeCpp provides efficient algorithms for:
- **Sparse variable selection** using splicing techniques
- **Cross-validation** for model selection
- **Information criteria** (AIC, BIC, etc.) for optimal sparsity level

## Features

- **High Performance**: Optimized C++ implementation with OpenMP parallelization
- **Flexible Model Interface**: Easy to extend with custom loss functions and optimizers
- **Memory Safe**: Smart pointer-based memory management
- **Example Included**: Linear regression example demonstrating usage

## Dependencies

### Required
- **C++ Compiler**: GCC 4.8+ or Clang 3.4+ with C++11 support

### Optional
- **OpenMP**: For parallel computation (optional but recommended)

## Building

### Using Make (Recommended)

```bash
# Clone the repository
git clone https://github.com/bbayukari/ScopeCpp.git
cd ScopeCpp

# Build the project
make

# Run the example
make run ## or ./scope
```

### Manual Compilation

```bash
g++ -std=c++11 -O3 -fopenmp -I./include \
    src/Algorithm.cpp \
    src/UniversalData.cpp \
    src/utilities.cpp \
    src/models.cpp \
    src/main.cpp \
    -o scope
```

### Clean Build

```bash
make clean      # Remove object files and executable
make distclean  # Remove all generated files
```

## Usage

### Quick Start

The provided example demonstrates linear regression with variable selection:

```cpp
#include "models.h"
#include "UniversalData.h"
#include "Algorithm.h"

// Create data (X: n x p, y: n x 1)
MatrixXd X = MatrixXd::Random(100, 10);
VectorXd y = VectorXd::Random(100);

// Combine into data matrix [X, y]
MatrixXd* data = new MatrixXd(100, 11);
data->leftCols(10) = X;
data->col(10) = y;

// Set up model
UniversalModel model;
model.set_loss_of_model(LinearRegressionModel::loss);
model.set_gradient_user_defined(LinearRegressionModel::loss_and_gradient);
model.set_slice_by_sample(LinearRegressionModel::slice_by_sample);

// Set up optimizer
ConvexSolver solver = [](/*...*/) {
    return GradientDescentSolver::solve(/*...*/);
};

// Run SCOPE algorithm
VectorXd beta;
double train_loss, test_loss, ic;
tie(beta, train_loss, test_loss, ic) = run_scope(/*...*/);
```

See `src/main.cpp` for a complete working example.

### Extending with Custom Models

To implement your own model, create functions matching these signatures:

```cpp
// Loss function
double my_loss(const VectorXd& beta, const MatrixXd* data) {
    // Compute loss
    return loss_value;
}

// Loss and gradient
pair<double, VectorXd> my_loss_and_gradient(
    const VectorXd& beta, 
    const MatrixXd* data) {
    // Compute loss and gradient
    return make_pair(loss_value, gradient);
}

// Sample slicing for cross-validation
MatrixXd* my_slice_by_sample(
    const MatrixXd* data, 
    const VectorXi& indices) {
    // Extract subset of samples
    return sliced_data;
}
```

Then register them with the model:

```cpp
UniversalModel model;
model.set_loss_of_model(my_loss);
model.set_gradient_user_defined(my_loss_and_gradient);
model.set_slice_by_sample(my_slice_by_sample);
```

## Project Structure

```
ScopeCpp/
├── include/              # External dependencies (Eigen, spdlog, etc.)
├── src/                  # Source code
│   ├── Algorithm.cpp/h   # Core SCOPE algorithm
│   ├── UniversalData.cpp/h  # Data and model interface
│   ├── models.cpp/h      # Example models (LinearRegression)
│   ├── utilities.cpp/h   # Utility functions
│   ├── main.cpp          # Example usage
│   ├── Data.h            # Data wrapper
│   ├── Metric.h          # Information criteria and CV
│   ├── path.h            # Path search algorithms
│   └── screening.h       # Variable screening
├── Makefile              # Build configuration
├── README.md             # This file
└── LICENSE               # MIT License
```

## Algorithm Parameters

Key parameters for `run_scope()`:

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_size` | int | Number of variables (features) |
| `sample_size` | int | Number of observations |
| `max_iter` | int | Maximum iterations per fit |
| `exchange_num` | int | Number of variables to exchange in splicing |
| `path_type` | int | Search strategy (1: sequential, 2: golden section) |
| `ic_type` | int | Information criterion (1: AIC, 2: BIC, 3: GIC) |
| `Kfold` | int | Number of CV folds (1 = no CV) |
| `s_min`, `s_max` | int | Minimum/maximum sparsity levels |
| `screening_size` | int | Pre-screening size (-1 = no screening) |

## Performance

- **Parallel Computing**: Enable OpenMP for multi-threaded execution
- **Screening**: Use variable screening for ultra-high dimensional data
- **Warm Start**: Enabled by default for faster path-following

## Examples

### Linear Regression

```bash
./scope
```

Output:
```
=== ScopeCpp 线性回归示例 ===
运行 SCOPE 算法（BIC + Sequential Path）...

=== 结果 ===
真实 beta（非零项）: [0]=1.5, [2]=-0.8, [5]=2.0
估计 beta:
  beta[0] = 1.49789
  beta[2] = -0.799777
  beta[5] = 2.00108
训练损失: 0.00466457
BIC: -2318.66

=== 验证 ===
正确识别的非零变量数: 3 / 3
✓ 支撑集完全正确！

程序执行完毕！
```

## Citation

If you use this software, please cite:

```bibtex
@article{article,
  title={title},
  author={author},
  journal={journal},
  year={year}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

- **Original Project**: https://github.com/bbayukari/ScopeCpp
- **Issues**: https://github.com/bbayukari/ScopeCpp/issues

## Acknowledgments

- Eigen library for efficient linear algebra
- spdlog for logging functionality
