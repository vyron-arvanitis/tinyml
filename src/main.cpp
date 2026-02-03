#include <iostream>
#include <cstddef>
#include "matrix.h"
#include "random_gen.h"

int main()
{
    tinyml::Matrix m(3, 3);
    std::cout << "The rows of the Matrix are " << m.rows() << " The cols are " << m.cols() << "\n";
    std::cout << "The first element of the matrix is : " << m.get(0, 0) << "\n";
    std::cout << "Setting the firstelemtn to : 100 \n";
    m.set(0, 0, 100);
    std::cout << "The first element of the matrix is : " << m(0, 0) << "\n";
    std::cout << "Testing THE RANDOM GEN FILE \n";

    tinyml::RNG rng(123); // or std::random_device{}()
    tinyml::Matrix n = tinyml::random_normal(3, 2, 0.0, 1.0, rng);
    for (std::size_t i = 0; i < n.rows(); ++i)
    {
        for (std::size_t j = 0; j < n.cols(); ++j)
        {
            std::cout << n(i, j) << (j + 1 == n.cols() ? '\n' : ' ');
        }
    }

    std::cout << "The first element of the new matrix `n` is : " << n(1, 1) << "\n";

    n(1, 1) = 50;
    std::cout << "The first element (after changing it) of the new matrix `n` is : " << n(1, 1) << "\n";

    tinyml::Matrix c = m*n;
    std::cout << "Multiply matrices m x n  " << "\n";
    for (std::size_t i = 0; i < n.rows(); ++i)
    {
        for (std::size_t j = 0; j < n.cols(); ++j)
        {
            std::cout << n(i, j) << (j + 1 == n.cols() ? '\n' : ' ');
        }
    }



    return 0;
}
