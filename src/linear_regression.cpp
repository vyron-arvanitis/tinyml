#include "matrix.h"
#include <iostream>
#include "linear_regression.h"


namespace tinyml {
    LinearRegression::LinearRegression(Options opts)
        : learning_rate_(opts.learning_rate),
          epochs_(opts.epochs),
          weights_(),
          bias_(),
          fitted_(false) {
    }

    void LinearRegression::fit(Matrix &X, Matrix &y) {

        fitted_ = true;
    }

    Matrix LinearRegression::predict(Matrix &X) const {
        return weights_* X + bias_;
    }
}
