#include "random_gen.h"
#include "matrix.h"

namespace tinyml
{
    Matrix random_uniform(std::size_t rows, std::size_t cols,
                          float low, float high,
                          RNG &rng)
    {

        Matrix m(rows, cols);
        std::uniform_real_distribution<float> dist(low, high);

        for (std::size_t i = 0; i < rows; i++)
        {
            for (std::size_t j = 0; j < cols; j++)
            {
                m.set(i, j, dist(rng));
            }
        }

        return m;
    }

    Matrix random_normal(std::size_t rows, std::size_t cols,
                         float mean, float stddev,
                         RNG &rng)
    {

        Matrix m(rows, cols);
        std::normal_distribution<float> dist(mean, stddev);

        for (std::size_t i = 0; i < rows; i++)
        {
            for (std::size_t j = 0; j < cols; j++)
            {
                m.set(i, j, dist(rng));
            }
        }

        return m;
    }
}