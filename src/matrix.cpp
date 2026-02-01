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

    void Matrix::set(std::size_t row, std::size_t col, value_type value)
    {
        data_[row * cols_ + col] = value;
    }

    Matrix::value_type Matrix::get(std::size_t row, std::size_t col) const
    {

        return data_[row * cols_ + col];
    }

    Matrix Matrix::mat_mul(const Matrix &other_matrix) const{
        std::size_t new_rows = this->rows_;
        std::size_t new_cols = other_matrix.cols();

        Matrix new_matrix = Matrix(new_rows, new_cols);
        for(std::size_t i=0; i<this->rows_; i++){
            for(std::size_t j=0; j<other_matrix.cols(); j++){
                for(std::size_t k=0; k<cols_; k++){
                    new_matrix.data_[i*new_cols+j] += this->data_[i*cols_+k]*other_matrix.data_[k*other_matrix.cols_ +j];
                }
            }
        }
        return new_matrix;

    }
}