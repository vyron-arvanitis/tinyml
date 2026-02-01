#include "matrix.h"
#include <cstddef>
#include <stdexcept>

namespace tinyml
{
    Matrix::Matrix(std::size_t rows, std::size_t cols)
        : rows_(rows), // size_t(rows), think of it like float(5) sth like htis
          cols_(cols),
          data_(rows * cols, value_type{}) // call vector(count, value), bec value_ype is float
    {
    }

    std::size_t Matrix::rows() const
    {
        return rows_;
    }

    std::size_t Matrix::cols() const { return cols_; }

    void Matrix::set(std::size_t row, std::size_t col, value_type value)
    {
        data_[row * cols_ + col] = value;
    }

    Matrix::value_type Matrix::get(std::size_t row, std::size_t col) const
    {

        return data_[row * cols_ + col];
    }

    Matrix Matrix::mat_mul(const Matrix &other) const
    {

        if (cols_ != other.rows_)
        {
            throw std::invalid_argument("mat_mul: shape mismatch");
        }
        const std::size_t m = rows_;
        const std::size_t k = cols_;
        const std::size_t n = other.cols_;

        Matrix out(m, n);

        for (std::size_t i = 0; i < m; ++i)
        {
            for (std::size_t t = 0; t < k; ++t)
            {
                const value_type a_it = data_[index(i, t)];
                for (std::size_t j = 0; j < n; ++j)
                {
                    out.data_[out.index(i, j)] += a_it * other.data_[other.index(t, j)];
                }
            }
        }

        return out;
    }
}