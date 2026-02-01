#include "animal.h"
#include <iostream>
#include "matrix.h"

int main()
{
    tinyml::Animal a("cat", 3);
    a.speak();
    std::cout << "The animal age is " << a.age()<<"\n";
    tinyml::Matrix m(3,3);
    std::cout << "The rows of the Matrix are " << m.rows() << " The cols are " << m.cols();
    return 0;
}
