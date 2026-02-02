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
    std::cout << "The first element of the matrix is : " << m.get(0, 0) <<"\n";
    std::cout << "Testing THE RANDOM GEN FILE \n";
    
    tinyml::RNG rng(123); // or std::random_device{}()
    tinyml::Matrix n = tinyml::random_normal(3, 3, 0.0, 1.0, rng);
    for (std::size_t i = 0; i < n.rows(); ++i) {
        for (std::size_t j = 0; j < n.cols(); ++j) {
            std::cout << n.get(i, j) << (j + 1 == n.cols() ? '\n' : ' ');
        }
    }

    std::cout << "The first element of the matrix is : " << n.get(0, 0) <<"\n";


    return 0;
}
