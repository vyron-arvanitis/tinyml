#pragma once
#include <cstddef>
#include <vector>
namespace tinyml
{

    class Matrix
    {
    public:
        using value_type = float;
        Matrix(std::size_t rows, std::size_t cols);

        std::size_t rows() const; // the const means that this function is not allowed tomodify the object
        std::size_t cols() const;

        // Method that sets the Matrix elements
        void set(std::size_t row, std::size_t col, value_type value);

        // Method that gets the Matrix elements
        value_type get(std::size_t row, std::size_t col) const;

        // Method that fills the matrix
        void fill(value_type value = 0);

        // Method that multiplies matrices
        Matrix mat_mul(const Matrix &other_matrix) const;

    private:
        // Convert (row, col) to flat index (row-major).
        std::size_t index(std::size_t row, std::size_t col) const {return row* cols_ + col;}

        std::size_t rows_;
        std::size_t cols_;
        std::vector<value_type> data_;
    };

}
