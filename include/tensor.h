//
// Created by Vyron Arvanitis on 30.04.26.
//
#pragma once

#ifndef TINYML_TENSOR_H
#define TINYML_TENSOR_H
#include <vector>

namespace tinyml {
    template<typename T>
    class Tensor {
    public:
        struct Shape {
            std::vector<size_t> dims_;

            // Needed because we define another constructor below.
            // Without this, the compiler would NOT generate a default constructor,
            // so `Shape s;` and default-initialization of members (e.g., in Tensor) would fail.
            Shape() = default;

            // Enables brace-initialization: Shape s{2,3,4};
            // The values are passed as std::initializer_list and copied into dims
            Shape(const std::initializer_list<size_t> dims)
                : dims_(dims) {
            }

            size_t ndim() const { return dims_.size(); }

            size_t operator[](const size_t i) const { return dims_[i]; }
        };

        explicit Tensor(const Shape &shape);

        Tensor(const Shape &shape, std::vector<T> &data);

        Tensor(const Shape &shape, std::vector<T> &data, bool requires_grad_);

        // Return reference to underlying data (no copy).
        // const → caller cannot modify the tensor through this access.
        const T &operator()(std::initializer_list<size_t> indices) const;

    private:
        std::vector<T> data_;
        std::vector<T> grad_;
        Shape shape_;
        std::vector<size_t> strides_;
        bool requires_grad_ = false;

        void compute_strides();

        size_t offset(std::initializer_list<size_t> indices) const;
    };
}

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! IDEA TO MAYBE USE LATER !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// template<int D, typename T>
// struct Vec : public std::vector<Vec<D - 1, T>> {
//     static_assert(D >= 1, "Vector dimension must be greater than zero!");
//     template<typename... Args>
//     Vec(int n = 0, Args... args) : std::vector<Vec<D - 1, T>>(n, Vec<D - 1, T>(args...)) {
//     }
// };
// template<typename T>
// struct Vec<1, T> : public std::vector<T> {
//     Vec(int n = 0, const T& val = T()) : std::vector<T>(n, val) {
//     }
// };


/* Implemetnation of the temmplate class must be in the same file*/

namespace tinyml {
    template<typename T>
    Tensor<T>::Tensor(const Shape &shape)
        : shape_(shape) { compute_strides(); }

    template<typename T>
    Tensor<T>::Tensor(const Shape &shape, std::vector<T> &data)
        : data_(data),
          shape_(shape) { compute_strides(); }

    template<typename T>
    Tensor<T>::Tensor(const Shape &shape, std::vector<T> &data, bool requires_grad_)
        : data_(data), shape_(shape), requires_grad_(requires_grad_) { compute_strides(); }


    template<typename T>
    void Tensor<T>::compute_strides() {
        const size_t ndim = shape_.ndim();

        strides_.resize(ndim);

        if (ndim == 0) {
            return;
        }

        strides_[ndim - 1] = 1;

        for (size_t i = ndim - 1; i > 0; --i) {
            strides_[i - 1] = strides_[i] * shape_[i];
        }
    }

    template<typename T>
    size_t Tensor<T>::offset(std::initializer_list<size_t> indices) const {
        size_t offset = 0;
        size_t dim = 0;
        for (const auto idx_val : indices) {
            offset+=strides_[dim] * idx_val;
            ++dim;
        }
        return offset;

    }
    template<typename T>
    const T &Tensor<T>::operator()(std::initializer_list<size_t> indices) const {
        return data_[offset(indices)];
    }
}


#endif //TINYML_TENSOR_H
