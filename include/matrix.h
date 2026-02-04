#pragma once
#include <cstddef>
#include <vector>
#include <stdexcept>
#include <iosfwd>   

namespace tinyml
{      
    class Matrix;   // forward declaration (promises Matrix exists later)

    //Printing the matrix
    std::ostream& operator<<(std::ostream& os, const Matrix& matrix);
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
        Matrix &operator*=(value_type scalar);       // Matrix* scaler returns mutated object
        Matrix operator*(value_type scalar) const;   // Matrix* scaler returns new object

        Matrix &operator+=(const Matrix &other); 
        Matrix operator+(const Matrix &other) const;

        Matrix &operator-=(const Matrix &other); 
        Matrix operator-(const Matrix &other) const;

        std::size_t size() const;

    private:
        // Convert (row, col) to flat index (row-major).
        std::size_t index(std::size_t row, std::size_t col) const { return row * cols_ + col; }

        // Private Method that multiplies matrices used by the operator*()
        Matrix mat_mul_(const Matrix &other) const;
        
        void check_bounds_(std::size_t row, std::size_t col) const;
        void check_same_shape_(const Matrix& other) const;

        std::size_t rows_;
        std::size_t cols_;
        std::vector<value_type> data_;
    };

    Matrix operator*(Matrix::value_type scalar, const Matrix &m);
}
