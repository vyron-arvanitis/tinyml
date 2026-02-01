#include <matrix.h>
#include <cstddef>


namespace tinyml{
    Matrix::Matrix(std::size_t rows, std::size_t cols)
    : rows_(rows), cols_(cols) {}

    std::size_t Matrix::rows() const {
        return rows_;
    }

    std::size_t Matrix::cols() const {return cols_;}

}