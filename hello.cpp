#include <iostream>
#include <vector>
#include <typeinfo>

int main()
{
    int x = 5;

    std::cout << "x value: " << x << "\n";
    std::cout << "x address: " << &x << "\n";

    int *p = &x;
    std::cout << "p value (address): " << p << "\n";
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
    std::cout << "Pointer to the cars vector:" << "\n";
    std::vector<std::string>* pCars = &cars;

    std::cout << "pCars (points to cars): " << pCars << "\n";        // address of cars
    std::cout << "&pCars[0] (same): " << &pCars[0] << "\n";          // same as pCars
    std::cout << "&pCars[1] (??): " << &pCars[1] << "\n";     
    std::cout << "(*pCars)[1]  " << (*pCars)[1] << "should be BMW" << "\n";         
    

    std::cout << "&pCars (pointer variable): " << &pCars << "\n";    // different

    std::cout << "Acces the Value of the first element of  pCars " << (*pCars)[0] << "\n";
    std::cout << "Acces the Value of the first element of  pCars (with arrow operator `->`) " << pCars->at(0) << "\n";


    return 0;
}
