#pragma once
#include "matrix.h"
#include "loss.h"
#include <memory>
namespace tinyml
{
    class LinearRegression
    {
    public:
        LinearRegression(float lr, std::size_t epochs, std::unique_ptr<Loss> loss);

        void fit(const Matrix &X, const Matrix &y);

        Matrix predict(const Matrix &X) const;

        const Matrix &weights() const;
        const Matrix &bias() const;
        const Matrix &loss_history() const;

    private:
        float learning_rate_;
        std::size_t epochs_;
        std::unique_ptr<Loss> loss_;
        Matrix weights_;
        Matrix bias_;
        bool fitted_ = false;
        Matrix loss_history_;
    };
}