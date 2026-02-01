#include "animal.h"

#include <iostream>
#include <utility> // std::move

namespace tinyml
{

    Animal::Animal(std::string name, int age)
        : name_(std::move(name)), age_(age) {}

    const std::string &Animal::name() const { return name_; }
    int Animal::age() const { return age_; }

    void Animal::speak() const
    {
        std::cout << "Animal " << name_ << " says: hello\n";
    }

} // namespace tinyml
