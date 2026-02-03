#pragma once
#include "matrix.h"

namespace tinyml{
    float mse(const Matrix& y_hat, const Matrix& y);
    float rmse(const Matrix& y_hat, const Matrix& y);

}