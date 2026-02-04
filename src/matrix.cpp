#include "matrix.h"
#include <cstddef>
#include <stdexcept>
#include <ostream>

namespace tinyml
{

    // Printing the matrix
    std::ostream &operator<<(std::ostream &os, const Matrix &matrix)
    {
        for (std::size_t i = 0; i < matrix.rows(); ++i)
        {
            for (std::size_t j = 0; j < matrix.cols(); ++j)
            {
                os << matrix(i, j);
                if (j + 1 < matrix.cols())
                    os << ' ';
            }
            if (i + 1 < matrix.rows())
                os << '\n';
        }
        return os;
    }

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

    void Matrix::check_bounds_(std::size_t row, std::size_t col) const
    {
        if (row >= rows_ || col >= cols_)
        {
            throw std::out_of_range("Matrix: index out of range");
        }
    }

    void Matrix::check_same_shape_(const Matrix &other) const
    {
        if (rows_ != other.rows_ || cols_ != other.cols_)
        {
            throw std::invalid_argument("Matrix: shape mismatch");
        }
    }

    void Matrix::set(std::size_t row, std::size_t col, value_type value)
    {
        (*this)(row, col) = value; // exactly equivalent to `this->operator()(i, j) = value`;
    }

    Matrix::value_type Matrix::get(std::size_t row, std::size_t col) const
    {
        return (*this)(row, col);
    }

    // Operators that fetch data
    Matrix::value_type &Matrix::operator()(std::size_t row, std::size_t col)
    {
        check_bounds_(row, col);
        return data_[index(row, col)];
    }

    const Matrix::value_type &Matrix::operator()(std::size_t row, std::size_t col) const
    {
        check_bounds_(row, col);
        return data_[index(row, col)];
    }

    Matrix Matrix::mat_mul_(const Matrix &other) const
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
                const value_type a_it = (*this)(i, t); // exactly equivalent to `this->(i,t)`
                for (std::size_t j = 0; j < n; ++j)
                {
                    out(i, j) += a_it * other(t, j);
                }
            }
        }

        return out;
    }

    // Operator for Matrix multiplication
    Matrix Matrix::operator*(const Matrix &other) const
    {
        return (*this).mat_mul_(other);
    }

    // Operator for Matrix-scalar multiplication
    Matrix &Matrix::operator*=(value_type scalar)
    {
        for (std::size_t i = 0; i < rows_; ++i)
        {
            for (std::size_t j = 0; j < cols_; ++j)
            {
                (*this)(i, j) *= scalar;
            }
        }
        return *this;
    }

    /*This defines the Matrix * scalar*/
    Matrix Matrix::operator*(value_type scalar) const
    {
        Matrix out = *this; // copy
        out *= scalar;      // calls operator*=, modifies out in-place
        return out;         // returns the modified copy
    }

    /*This defines the scalar * Matrix*/
    Matrix operator*(Matrix::value_type scalar, const Matrix &m)
    {
        return m * scalar;
    }

    // Operators for Matrix addition
    Matrix &Matrix::operator+=(const Matrix &other)
    {
        check_same_shape_(other);
        for (std::size_t i = 0; i < rows_; ++i)
        {
            for (std::size_t j = 0; j < cols_; ++j)
            {
                (*this)(i, j) += other(i, j);
            }
        }
        return *this;
    }

    Matrix Matrix::operator+(const Matrix &other) const
    {
        Matrix out = *this; // copy
        out += other;       // reuse logic (calls operator+=)
        return out;         // return by value
    }

    // Operators for Matrix subtraction
    Matrix &Matrix::operator-=(const Matrix &other)
    {
        check_same_shape_(other);
        for (std::size_t i = 0; i < rows_; ++i)
        {
            for (std::size_t j = 0; j < cols_; ++j)
            {
                (*this)(i, j) -= other(i, j);
            }
        }
        return *this;
    }

    Matrix Matrix::operator-(const Matrix &other) const
    {
        Matrix out = *this; // copy
        out -= other;       // reuse logic (calls operator-=)
        return out;         // return by value
    }

    Matrix Matrix::transpose() const
    {
        Matrix out(cols_, rows_);
        for (std::size_t i = 0; i < rows_; ++i)
        {
            for (std::size_t j = 0; j < cols_; ++j)
            {
                out(j, i) = (*this)(i, j); // (*this)(i, j)  ==  this->operator()(i, j)
            }
        }
        return out;
    }

    std::size_t Matrix::size() const
    {
        return rows_ * cols_;
    }

    Matrix::Shape Matrix::shape()const{
        return {rows_, cols_};
    }

}
