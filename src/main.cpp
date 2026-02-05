#include <iostream>
#include <cstddef>
#include <memory>
#include "matrix.h"
#include "linear_regression.h"
#include "loss.h"
#include "random_gen.h"

void example_linear_regression()
{
    tinyml::Matrix X(4, 1);
    tinyml::Matrix y(4, 1);

    const float x_vals[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    for (std::size_t i = 0; i < 4; ++i)
    {
        X(i, 0) = x_vals[i];
        y(i, 0) = 2.0f * x_vals[i] + 1.0f;
    }

    tinyml::LinearRegression model(0.01f, 500, std::make_unique<tinyml::MSELoss>());
    std::cout << "weights:\n"
              << model.weights().size() << "\n";
    std::cout << "bias:\n"
              << model.bias().size() << "\n";
    model.fit(X, y);

    tinyml::Matrix preds = model.predict(X);

    std::cout << "Linear regression example\n";
    std::cout << "X:\n"
              << X << "\n";
    std::cout << "y:\n"
              << y << "\n";
    std::cout << "preds:\n"
              << preds << "\n";
    std::cout << "weights:\n"
              << model.weights() << "\n";
    std::cout << "bias:\n"
              << model.bias() << "\n";
}

int main()
{
    tinyml::Matrix m(3, 3);
    std::cout << "The rows of the Matrix are " << m.rows() << " The cols are " << m.cols() << "\n";
    auto shape_m = m.shape();
    std::cout << "The shape of the Matrix is " << shape_m.rows << " " << shape_m.cols << "\n";
    std::cout << "The first element of the matrix is : " << m.get(0, 0) << "\n";
    std::cout << "Setting the firstelemtn to : 100 \n";
    m.set(0, 0, 100);
    std::cout << "The first element of the matrix is : " << m(0, 0) << "\n";
    std::cout << "Testing THE RANDOM GEN FILE \n";

    tinyml::RNG rng(123); // or std::random_device{}()
    tinyml::Matrix n = tinyml::random_normal(3, 2, 0.0, 1.0, rng);
    std::cout << "The matrix n is \n"
              << n << "\n";

    std::cout << "The first element of the new matrix `n` is : " << n(1, 1) << "\n";

    n(1, 1) = 50;
    std::cout << "The first element (after changing it) of the new matrix `n` is : " << n(1, 1) << "\n";

    tinyml::Matrix c = m * n;
    std::cout << "Multiply matrices m x n  " << "\n";
    std::cout << "The matrix c is \n"
              << c << "\n";

    std::cout << "Multiply matrices (1X3)(3X1) " << "\n";
    tinyml::Matrix m_13 = tinyml::random_uniform(1, 3, 0, 1, rng);
    tinyml::Matrix m_31 = tinyml::random_uniform(3, 1, 0, 1, rng);
    tinyml::Matrix scalar = m_13 * m_31;
    for (std::size_t i = 0; i < scalar.rows(); ++i)
    {
        for (std::size_t j = 0; j < scalar.cols(); ++j)
        {
            std::cout << scalar(i, j) << (j + 1 == scalar.cols() ? '\n' : ' ');
        }
    }

    std::cout << "And the length of the scalar object is: " << scalar.size() << "\n";

    std::cout << " Addition of matrices (1X3)(3X1) " << "\n";
    tinyml::Matrix matrix_add = m_13 + m_13;
    std::cout << "The reuslt of the addition of matrices is" << "\n";
    std::cout << "The matrix_add n is \n"
              << matrix_add << "\n";

    std::cout << " Subtraction of matrices (1X3)(3X1) " << "\n";
    tinyml::Matrix matrix_sub = m_13 * 4 - 2 * m_13;
    std::cout << "The matrix_sub n is \n"
              << matrix_add << "\n";

    example_linear_regression();

    return 0;
}
