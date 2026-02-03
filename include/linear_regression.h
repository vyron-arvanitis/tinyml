#pragma once
#include "matrix.h"

namespace tinyml
{
    class LinearRegression
    {
    public:
        void fit(Matrix &X, Matrix &b);

        Matrix predict(Matrix &X) const;

        const Matrix &weights() const;
        const Matrix &bias() const;

    private:
        Matrix weights_;
        Matrix bias_;
        bool fitted_ = false;
    };
}