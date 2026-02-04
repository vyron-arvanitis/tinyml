#include "matrix.h"
#include <iostream>
#include "linear_regression.h"
#include "loss.h"
namespace tinyml
{
    LinearRegression::LinearRegression(Options opts)
        : learning_rate_(opts.learning_rate),
          epochs_(opts.epochs),
          weights_(),
          bias_(),
          fitted_(false)
    {
    }

    void LinearRegression::fit(Matrix &X, Matrix &y)
    {
        for (std::size_t epoch = 0; epoch <= epochs_; epoch++)
        {
            Matrix predictions = (*this).predict(X);
        }

        fitted_ = true;
    }

    Matrix LinearRegression::predict(Matrix &X) const
    {
        return X * weights_ + bias_; // x (N_data, features) * (features, 1) + (features, 1)
    }
}
