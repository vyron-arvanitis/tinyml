#pragma once
#include <cstddef>
#include <vector>
#include <stdexcept>

namespace tinyml {
    class Matrix {
    public:
        using value_type = float;

        Matrix(std::size_t rows, std::size_t cols);

        std::size_t rows() const; // the const means that this function is not allowed tomodify the object
        std::size_t cols() const;

        // Method that sets the Matrix elements
        void set(std::size_t row, std::size_t col, value_type value);

        // Method that gets the Matrix elements
        value_type get(std::size_t row, std::size_t col) const;

        // Element access (row, col).
        // Returns a reference so callers can both read and write elements:
        //
        //   A(i,j) = 1.0f;     // write
        //   float x = A(i,j);  // read
        //
        // Reference validity: the returned reference points into the internal
        // contiguous storage (std::vector). It remains valid as long as the Matrix
        // object is alive and its storage is not reallocated. In this class, the
        // storage size is fixed after construction (no push_back/resize), so
        // references are stable.
        value_type &operator()(std::size_t row, std::size_t col);

        const value_type &operator()(std::size_t row, std::size_t col) const;

        // Method that transposes the matrix
        Matrix transpose() const;

        Matrix operator*(const Matrix &other) const; // Matrix * Matrix
        Matrix operator*(value_type scalar) const; // Matrix* scaler returns new object
        Matrix &operator*=(value_type scalar); // Matrix* scaler returns mutated object


        Matrix operator+(const Matrix &other) const;

        Matrix operator-(const Matrix &other) const;


        std::size_t len() const;

    private:
        // Convert (row, col) to flat index (row-major).
        std::size_t index(std::size_t row, std::size_t col) const { return row * cols_ + col; }

        // Private Method that multiplies matrices used by the operator*()
        Matrix mat_mul_(const Matrix &other) const {
            if (cols_ != other.rows_) {
                throw std::invalid_argument("mat_mul: shape mismatch");
            }
            const std::size_t m = rows_;
            const std::size_t k = cols_;
            const std::size_t n = other.cols_;

            Matrix out(m, n);

            for (std::size_t i = 0; i < m; ++i) {
                for (std::size_t t = 0; t < k; ++t) {
                    const value_type a_it = (*this)(i, t); // exactly equivalent to `this->(i,t)`
                    for (std::size_t j = 0; j < n; ++j) {
                        out(i, j) += a_it * other(t, j);
                    }
                }
            }

            return out;
        }

        Matrix mat_addition_(const Matrix &other) const {
            if (rows_ != other.rows_ || cols_ != other.cols_) {
                throw std::invalid_argument("mat_addition: shape mismatch");
            }
            Matrix out(rows_, cols_);

            for (std::size_t i = 0; i < rows_; ++i) {
                for (std::size_t j = 0; j < cols_; ++j) {
                    out(i, j) = (*this)(i, j) + other(i, j);
                }
            }

            return out;
        }

        Matrix mat_subtraction_(const Matrix &other) const {
            if (rows_ != other.rows_ || cols_ != other.cols_) {
                throw std::invalid_argument("mat_subtraction_: shape mismatch");
            }
            Matrix out(rows_, cols_);

            for (std::size_t i = 0; i < rows_; ++i) {
                for (std::size_t j = 0; j < cols_; ++j) {
                    out(i, j) = (*this)(i, j) - other(i, j);
                }
            }

            return out;
        }

        std::size_t rows_;
        std::size_t cols_;
        std::vector<value_type> data_;
    };

    Matrix operator*(Matrix::value_type scalar, const Matrix &m);
}
