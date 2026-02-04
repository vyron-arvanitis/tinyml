#pragma once
#include "matrix.h"

namespace tinyml
{

    class Loss
    {
    public:
        using value_type = Matrix::value_type;

        // Virtual destructor so deleting via a Loss* calls the derived destructor.
        virtual ~Loss() = default;

        // "callable" forward
        virtual value_type operator()(const Matrix &y_hat, const Matrix &y) = 0;

        // gradient w.r.t. y_hat
        virtual Matrix backward() = 0;

    protected:
        Matrix last_y_hat_; // could have also been const Matrix last_y_hat = nullptr; this raises issue of lifetime!
        Matrix last_y_; //TODO[LATER]: consider implementing the above idea later!
    };

    class MSELoss final : public Loss
    {
    public:
        value_type operator()(const Matrix &y_hat, const Matrix &y) override; // override is a sfaty net aht wuld raise an error if the function had different signature than the parent
        Matrix backward() override;
    };
    // float mse(const Matrix& y_hat, const Matrix& y);
    // float rmse(const Matrix& y_hat, const Matrix& y);

}
