#pragma once
#include <random>
#include <cstddef>
#include "matrix.h"

namespace tinyml
{

    using RNG = std::mt19937_64;

    // Generate a matrix with random uniform numbers
    Matrix random_uniform(std::size_t rows, std::size_t cols,
                          float low, float high,
                          RNG &rng);

    // Generate a matri iwrh random normalnumbers
    Matrix random_normal(std::size_t rows, std::size_t cols,
                         float mean, float stddev,
                         RNG &rng);
}