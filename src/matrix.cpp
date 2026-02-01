#include "matrix.h"
#include <cstddef>

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

    Matrix::value_type Matrix::get(std::size_t row, std::size_t col) const
    {

        return data_[row * cols_ + col];
    }

}