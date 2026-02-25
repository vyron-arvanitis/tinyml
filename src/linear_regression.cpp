#include "matrix.h"
#include <iostream>
#include "linear_regression.h"
#include "loss.h"
#include "random_gen.h"
#include <stdexcept>

namespace tinyml
{
    LinearRegression::LinearRegression(float lr, std::size_t epochs, std::unique_ptr<Loss> loss)
        : learning_rate_(lr),
          epochs_(epochs),
          loss_(std::move(loss)),
          weights_(0, 0),
          bias_(0, 0),
          loss_history_(0, 0),
          fitted_(false)
    {
    }

    void LinearRegression::fit(const Matrix &X, const Matrix &y)
    {
        tinyml::RNG rng(123); // or std::random_device{}()
        weights_ = random_normal(X.cols(), y.cols(), 0, 1, rng);
        bias_ = random_normal(1, y.cols(), 0, 1, rng);
        Matrix X_T = X.transpose();
        loss_history_ = Matrix(epochs_, 1);

        for (std::size_t epoch = 0; epoch < epochs_; epoch++)
        {   
            Matrix predictions = X * weights_ + bias_;
            loss_history_(epoch, 0) = (*loss_)(predictions, y);
            Matrix dl_dy_hat = loss_->backward();  // (N, y.cols)
            Matrix grad_w = X_T * dl_dy_hat; // (F, N) *(N, y.cols)
            Matrix grad_b = Matrix::ones(1, dl_dy_hat.rows())*dl_dy_hat; 
            weights_ -= learning_rate_ * grad_w;
            bias_ -= learning_rate_ * grad_b;
        }

        fitted_ = true;
    }

    Matrix LinearRegression::predict(const Matrix &X) const
    {
        if (!fitted_)
        {
            throw std::logic_error("LinearRegression::predict() called before fit()");
        }

        return X * weights_ + bias_; // x (N_data, features) * (features, y.cols) +(broadcasting!) (1, y.cols)
    }

    const Matrix &LinearRegression::weights() const
    {
        return weights_;
    }

    const Matrix &LinearRegression::bias() const
    {
        return bias_;
    }

    const Matrix &LinearRegression::loss_history() const{
        return  loss_history_;
    }
}
