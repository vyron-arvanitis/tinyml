#include "matrix.h"
#include "loss.h"
#include <cmath>
#include <stdexcept>

namespace tinyml
{

    float mse(const Matrix &y_hat, const Matrix &y)
    {

        if (y_hat.rows() != y.rows() || y_hat.cols() != y.cols())
        {
            throw std::invalid_argument("mse: shape mismatch!");
        }

        std::size_t elements = y_hat.rows() * y_hat.cols();

        if (elements == 0)
        {
            throw std::invalid_argument("mse: empty input");
        }
        Matrix::value_type summ = Matrix::value_type(0); // could have also been a simple `0` but this is the most robust way
        for (std::size_t i = 0; i < y_hat.rows(); i++)
        {
            for (std::size_t j = 0; j < y_hat.cols(); j++)
            {
                const Matrix::value_type diff = y_hat.get(i, j) - y.get(i, j);

                summ += diff * diff;
            }
        }
        return summ / static_cast<Matrix::value_type>(elements); // could have also been `Matrix::value_type(elements)`
    }

    float rmse(const Matrix &y_hat, const Matrix &y)
    {
        return sqrt(mse(y_hat, y));
    }
}