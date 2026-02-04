#include "matrix.h"
#include "loss.h"
#include <cmath>
#include <stdexcept>
#include <string>

namespace tinyml
{
    using value_type = Matrix::value_type;

    namespace
    {
        std::size_t checked_elements(const Matrix &y_hat, const Matrix &y, const char *name)
        {
            if (y_hat.rows() != y.rows() || y_hat.cols() != y.cols())
            {
                throw std::invalid_argument(std::string(name) + ": shape mismatch!");
            }

            const std::size_t elements = y_hat.rows() * y_hat.cols();
            if (elements == 0)
            {
                throw std::invalid_argument(std::string(name) + ": empty input");
            }
            return elements;
        }
    }

    value_type MSELoss::operator()(const Matrix &y_hat, const Matrix &y)
    {
        last_y_hat_ = y_hat;
        last_y_ = y;
        return mse(y_hat, y);
    }

    Matrix MSELoss::backward()
    {
        return mse_grad(last_y_hat_, last_y_);
    }

    value_type mse(const Matrix &y_hat, const Matrix &y)
    {

        const std::size_t rows = y_hat.rows();
        const std::size_t cols = y_hat.cols();
        const std::size_t elements = checked_elements(y_hat, y, "mse");

        value_type summ = value_type(0); // could have also been a simple `0` but this is the most robust way
        for (std::size_t i = 0; i < rows; i++)
        {
            for (std::size_t j = 0; j < cols; j++)
            {
                const value_type diff = y_hat(i, j) - y(i, j);

                summ += diff * diff;
            }
        }
        return summ / static_cast<value_type>(elements); // could have also been `Matrix::value_type(elements)`
    }

    Matrix mse_grad(const Matrix &y_hat, const Matrix &y)
    {

        const std::size_t elements = checked_elements(y_hat, y, "mse_grad");
        const value_type scale = value_type(2) / static_cast<value_type>(elements);
        return scale * (y_hat - y);
    }

    // float rmse(const Matrix &y_hat, const Matrix &y)
    // {
    //     return sqrt(mse(y_hat, y));
    // }
}
