#pragma once

#include <string>

namespace tinyml
{

    class Animal
    {
    public:
        Animal(std::string name, int age);

        const std::string &name() const;
        int age() const;

        void speak() const;

    private:
        std::string name_;
        int age_;
    };

} // namespace tinyml
