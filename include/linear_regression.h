#pragma once
#include "matrix.h"

namespace tinyml
{
    class LinearRegression
    {
    public:
        struct Options
        {
            float learning_rate = 0.01f;
            std::size_t epochs = 1000;
        };

        LinearRegression(Options opts = {});
        
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