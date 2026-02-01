#include <iostream>
#include <vector>
#include <typeinfo>
#include <vector>

int main()
{
    int x = 5;

    std::cout << "x value: " << x << "\n";
    std::cout << "x address: " << &x << "\n";

    int *p = &x;
    std::cout << "p value (address): " << p << "\n";
    std::cout << "x address: " << &x << "\n";
    std::cout << "*p value: " << *p << "\n";

    *p = 10;
    std::cout << "x after *p=10: " << x << "\n";

    std::vector<std::string> cars = {"Volvo", "BMW", "Ford", "Mazda"};

    std::cout << "\n";
    std::cout << "The cars vector containts: " << "\n";
    for (std::string car : cars)
    {
        std::cout << car << "\n";
    }

    std::cout << "The first car is " << cars[0] << "\n";
    std::cout << "The first car is " << cars.at(0) << "\n";

    cars[0] = "Ferrari";
    std::cout << "Change the first car into " << cars.at(0) << "\n";

    std::cout << "The last car is " << cars.back() << "\n";
    cars.push_back("Bugatti");

    std::cout << "After adding one element the last car is " << cars.back() << "\n";

    cars.pop_back();

    std::cout << "After deletting the last element last car is " << cars.back() << "\n";

    std::cout << "The size of the car vector is " << cars.size() << "\n";

    std::cout << "\n";

    std::cout << "Playing with the pointers " << "\n";

    std::cout << "Pointer to the cars first element:" << "\n";

    std::string *pCars_0 = &cars[0];
    std::cout << "Memory of pCars" << pCars_0 << "\n";
    std::cout << "Value of pCars " << *pCars_0 << "\n";
    // std::cout << "Memory of pCars type " << typeid(pCars_0).name() << "\n";
    // std::cout << "Value of pCars type  " << typeid(*pCars_0).name() << "\n";
    // std::cout << "The type fo the cars varivable is " << typeid(cars).name() << "\n";
    std::cout << "----------------------------\n";
    std::vector<std::string> *pCars = &cars;

    std::cout << "Pointer to a std::vector<std::string>\n\n";

    // 1) Addresses: vector object vs pointer variable
    std::cout << "Address of vector object (&cars):           " << &cars << "\n";
    std::cout << "Pointer value (pCars == &cars):             " << pCars << "\n";
    std::cout << "Address of pointer variable (&pCars):       " << &pCars << "\n\n";

    // 2) What pCars[0] and pCars[1] really mean (array-style access on a pointer)
    // pCars[0] == *pCars (OK because pCars points to exactly one vector object)
    std::cout << "pCars[0] is the vector object (*pCars). Its address: " << &pCars[0] << "\n";

    // pCars[1] would mean "the next vector object in memory" (DO NOT dereference/use).
    std::cout << "pCars + 1 (address where a 'next vector' would be):  " << (pCars + 1) << "\n\n";

    // 3) Accessing elements of the vector THROUGH the pointer
    std::cout << "First element via (*pCars)[0]:               " << (*pCars)[0] << "\n";
    std::cout << "Second element via (*pCars)[1]:              " << (*pCars)[1] << "\n";
    std::cout << "First element via pCars->at(0) (checked):    " << pCars->at(0) << "\n\n";

    // 4) Addresses related to the vector's internal storage (elements live in a buffer)
    std::cout << "Address of internal element buffer (cars.data()): "
              << static_cast<const void *>(cars.data()) << "\n";
    std::cout << "Address of std::string object cars[0]:       " << &cars[0] << "\n";
    std::cout << "Address of std::string object cars[1]:       " << &cars[1] << "\n";

    std::cout << "----------------------------\n";
    std::cout << "Another wayt o reate a vector!!!!!!!!!!!\n";
    std::vector<int> v(5, 42);           // [42,42,42,42,42]
    std::vector<std::string> s(3, "hi"); // ["hi","hi","hi"]
    std::cout << "The v[0] vector is " << v[0] << "\n";
    std::cout << "The v[1] vector is " << v[1] << "\n";
    std::cout << "The v[2] vector is " << v[2] << "\n";
    std::cout << "The v[3] vector is " << v[3] << "\n";

    std::cout << "The s[1] vector is " << s[1] << "\n";
    std::cout << "Example of the brace initialization float  " << float{} << "\n";
    std::cout << "Example of the brace initialization double " << double{} << "\n";
    std::cout << "Example of the brace initialization int " << int{} << "\n";
    std::cout << "Example of the brace initialization string " << std::string{} << "\n";
    std::cout << "vector<int>{5}.size() = " << std::vector<int>{5}.size() << "\n";
    std::cout << "vector<int>{5, 1, 1}.size() = " << std::vector<int>{5, 1, 1}.size() << "\n";

    std::cout <<"\nTesting out the std::size_t" << std::size_t(5) << " Size of this is " << sizeof(std::size_t(5));
    std::cout <<"\nTesting out the float" << float(5) ;
    std::cout <<"\nTesting out the int" << int(5) ;

    return 0;
}
