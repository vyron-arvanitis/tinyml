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

    std::size_t Matrix::cols() const
    {
        return cols_;
    }

    void Matrix::set(std::size_t row, std::size_t col, value_type value)
    {
        (*this)(row, col) = value; // exactly equivalent to `this->(i,j) = value`;
    }

    Matrix::value_type Matrix::get(std::size_t row, std::size_t col) const
    {

        return (*this)(row, col);
    }

     Matrix::value_type &Matrix::operator()(std::size_t row, std::size_t col)
    {
        return data_[index(row, col)];
    }

    const Matrix::value_type &Matrix::operator()(std::size_t row, std::size_t col) const
    {
        return data_[index(row, col)];
    }

    Matrix Matrix::operator*(const Matrix& other) const{
        return (*this).mat_mul(other);
    }

    Matrix Matrix::transpose() const
    {

        Matrix out(rows_, cols_);
        for (std::size_t i = 0; i < rows_; ++i)
        {
            for (std::size_t j = 0; j < cols_; ++j)
            {
                out(j, i) = (*this)(i, j); // exactly equivalent to `this->(i,j)`
            }
        }
        return out;
    }
}