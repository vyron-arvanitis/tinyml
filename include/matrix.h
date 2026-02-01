#pragma once
#include <cstddef>

#include <vector>
namespace tinyml
{

    class Matrix
    {
    public:
        Matrix(std::size_t rows, std::size_t cols);

        std::size_t rows() const; // the const means that this function is not allowed tomodify the object
        std::size_t cols() const;

    private:
        std::size_t rows_;
        std::size_t cols_;
    };

} 
