#include "matrix.h"
#include <iostream>
#include "linear_regression.h"
#include "loss.h"
namespace tinyml
{
    LinearRegression::LinearRegression(float lr, std::size_t epochs, std::unique_ptr<Loss> loss)
        : learning_rate_(lr),
          epochs_(epochs),
          loss_(std::move(loss)),
          weights_(), //TODO: figure out how to do that
          bias_(), //TODO: figure out how to do that
          fitted_(false)
    {
    }

    void LinearRegression::fit(Matrix &X, Matrix &y)
    {
        for (std::size_t epoch = 0; epoch <= epochs_; epoch++)
        {
            Matrix predictions = (*this).predict(X);
            auto loss_val = (*loss_)(predictions, y);
            Matrix dl_dyhay = loss_->backward();
        }

        fitted_ = true;
    }

    Matrix LinearRegression::predict(Matrix &X) const
    {
        return X * weights_ + bias_; // x (N_data, features) * (features, 1) + (features, 1)
    }
}
