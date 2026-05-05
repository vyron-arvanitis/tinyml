//
// Created by Vyron Arvanitis on 30.04.26.
//
#pragma once

#ifndef TINYML_TENSOR_H
#define TINYML_TENSOR_H
#include <iterator>
#include <vector>
#include <__ranges/data.h>
#include<algorithm>

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

            size_t numel() const {
                size_t total = 1;
                for (const size_t d: dims_) total *= d;
                return total;
            }

            bool operator==(const Shape &other) const {
                return dims_ == other.dims_;
            }

            size_t operator[](const size_t i) const { return dims_[i]; }
        };

        explicit Tensor(const Shape &shape);

        Tensor(const Shape &shape, const std::vector<T> &data);

        Tensor(const Shape &shape, const std::vector<T> &data, bool requires_grad_);

        //--------------------------//
        /*REGION DEFINING OF OPERATORS BETWEEN TENSORS*/
        // Return reference to underlying data (no copy).
        // const → caller cannot modify the tensor through this access.
        const T &operator()(std::initializer_list<size_t> indices) const;

        Tensor &operator+=(const Tensor &other);

        Tensor operator+(const Tensor &other) const;

        Tensor &operator-=(const Tensor &other);

        Tensor operator-(const Tensor &other) const;

        Tensor &operator*=(const Tensor &other);

        Tensor operator*(const Tensor &other) const;

        Tensor &operator/=(const Tensor &other);

        Tensor operator/(const Tensor &other) const;

        //--------------------------//
        /*REGION DEFINING OF OPERATORS BETWEEN TENSORS AND SCALARS*/
        Tensor &operator+=(const T &scalar);

        Tensor operator+(const T &scalar) const;

        Tensor &operator-=(const T &scalar);

        Tensor operator-(const T &scalar) const;

        Tensor &operator*=(const T &scalar);

        Tensor operator*(const T &scalar) const;

        Tensor &operator/=(const T &scalar);

        Tensor operator/(const T &scalar) const;


        //--------------------------//
        /*REGION DEFINE PUBLIC METHODS*/
        const std::vector<T> &data() const;

        const std::vector<T> &grad() const;

        std::vector<T> &data();

        std::vector<T> &grad();

        void zero_grad();

        const Shape &shape() const;

    private:
        std::vector<T> data_;
        std::vector<T> grad_;
        Shape shape_;
        std::vector<size_t> strides_;
        bool requires_grad_ = false;

        void compute_strides();

        void validate_size() const;

        size_t offset(std::initializer_list<size_t> indices) const;
    };

    template<typename T>
    Tensor<T> operator+(const T &scalar, const Tensor<T> &other);

    template<typename T>
    Tensor<T> operator-(const T &scalar, const Tensor<T> &other);

    template<typename T>
    Tensor<T> operator*(const T &scalar, const Tensor<T> &other);
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


/* Implementation of the template class must be in the same file*/

namespace tinyml {
    template<typename T>
    Tensor<T>::Tensor(const Shape &shape)
        : data_(shape.numel(), T{}),
          grad_(shape.numel(), T{}),
          shape_(shape) {
        validate_size(); //same as (*this).validate_size()
        compute_strides();
    }

    template<typename T>
    Tensor<T>::Tensor(const Shape &shape, const std::vector<T> &data)
        : data_(data),
          grad_(shape.numel(), T{}),
          shape_(shape) {
        validate_size();
        compute_strides();
    }

    template<typename T>
    Tensor<T>::Tensor(const Shape &shape, const std::vector<T> &data, const bool requires_grad_)
        : data_(data),
          grad_(shape.numel(), T{}),
          shape_(shape),
          requires_grad_(requires_grad_) {
        validate_size();
        compute_strides();
    }

    //--------------------------//
    /*PRIVATE METHODS*/
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
    void Tensor<T>::validate_size() const {
        if (data_.size() != shape_.numel())
            throw std::invalid_argument("Data size does not match Tensor shape");
    }

    template<typename T>
    size_t Tensor<T>::offset(std::initializer_list<size_t> indices) const {
        if (indices.size() != shape_.ndim()) {
            throw std::invalid_argument("Indices size does not match Tensor shape");
        }
        size_t offset = 0;
        size_t dim = 0;
        for (const auto idx_val: indices) {
            if (idx_val >= shape_[dim]) {
                throw std::invalid_argument("Index out of bounds");
            }
            offset += strides_[dim] * idx_val;
            ++dim;
        }
        return offset;
    }

    //--------------------------//
    /*REGION IMPLEMENTATION OF OPERATORS BETWEEN TENSORS*/

    template<typename T>
    Tensor<T> &Tensor<T>::operator+=(const T &scalar) {
        for (size_t i = 0; i < data_.size(); i++) {
            data_[i] += scalar;
        }
        return *this;
    }

    template<typename T>
    Tensor<T> Tensor<T>::operator+(const T &scalar) const {
        Tensor<T> out = *this;
        out += scalar;
        return out;
    }

    template<typename T>
    Tensor<T> operator+(const T &scalar, const Tensor<T> &other) {
        return other + scalar;
    }

    template<typename T>
    Tensor<T> &Tensor<T>::operator-=(const T &scalar) {
        for (size_t i = 0; i < data_.size(); i++) {
            data_[i] -= scalar;
        }
        return *this;
    }

    template<typename T>
    Tensor<T> Tensor<T>::operator-(const T &scalar) const {
        Tensor<T> out = *this;
        out -= scalar;
        return out;
    }

    template<typename T>
    Tensor<T> operator-(const T &scalar, const Tensor<T> &other) {
        Tensor<T> out = other;

        for (auto &data: out.data()) {
            data = scalar - data;
        }

        return out;
    }

    template<typename T>
    Tensor<T> &Tensor<T>::operator*=(const T &scalar) {
        for (size_t i = 0; i < data_.size(); i++) {
            data_[i] *= scalar;
        }
        return *this;
    }

    template<typename T>
    Tensor<T> Tensor<T>::operator*(const T &scalar) const {
        Tensor<T> out = *this;
        out *= scalar;
        return out;
    }

    template<typename T>
    Tensor<T> operator*(const T &scalar, const Tensor<T> &other) {
        return other * scalar;
    }

    template<typename T>
    Tensor<T> &Tensor<T>::operator/=(const T &scalar) {
        for (size_t i = 0; i < data_.size(); i++) {
            data_[i] /= scalar;
        }
        return *this;
    }

    template<typename T>
    Tensor<T> Tensor<T>::operator/(const T &scalar) const {
        Tensor<T> out = *this;
        out /= scalar;
        return out;
    }

    template<typename T>
    const T &Tensor<T>::operator()(std::initializer_list<size_t> indices) const {
        return data_[offset(indices)];
    }


    template<typename T>
    Tensor<T> &Tensor<T>::operator+=(const Tensor &other) {
        if (shape_ != other.shape_) {
            throw std::invalid_argument("Tensor shape mismatch");
        }

        for (size_t i = 0; i < data_.size(); ++i) {
            data_[i] += other.data_[i];
        }
        return *this;
    }

    template<typename T>
    Tensor<T> Tensor<T>::operator+(const Tensor &other) const {
        Tensor<T> out = *this;
        out += other;
        return out;
    }

    template<typename T>
    Tensor<T> &Tensor<T>::operator-=(const Tensor &other) {
        if (shape_ != other.shape_) {
            throw std::invalid_argument("Tensor shape mismatch");
        }

        for (size_t i = 0; i < data_.size(); ++i) {
            data_[i] -= other.data_[i];
        }
        return *this;
    }

    template<typename T>
    Tensor<T> Tensor<T>::operator-(const Tensor &other) const {
        Tensor<T> out = *this;
        out -= other;
        return out;
    }

    template<typename T>
    Tensor<T> &Tensor<T>::operator*=(const Tensor &other) {
        if (shape_ != other.shape_) {
            throw std::invalid_argument("Tensor shape mismatch");
        }
        for (size_t i = 0; i < data_.size(); ++i) {
            data_[i] *= other.data_[i];
        }
        return *this;
    }

    template<typename T>
    Tensor<T> Tensor<T>::operator*(const Tensor &other) const {
        Tensor<T> out = *this;
        out *= other;
        return out;
    }

    template<typename T>
    Tensor<T> &Tensor<T>::operator/=(const Tensor &other) {
        if (shape_ != other.shape_) {
            throw std::invalid_argument("Tensor shape mismatch");
        }
        for (size_t i = 0; i < data_.size(); ++i) {
            data_[i] /= other.data_[i];
        }
        return *this;
    }

    template<typename T>
    Tensor<T> Tensor<T>::operator/(const Tensor &other) const {
        Tensor<T> out = *this;
        out /= other;
        return out;
    }


    //--------------------------//
    /*REGION IMPLEMENT PUBLIC METHODS*/
    template<typename T>
    const std::vector<T> &Tensor<T>::data() const {
        return data_;
    }

    template<typename T>
    const std::vector<T> &Tensor<T>::grad() const {
        return grad_;
    }

    template<typename T>
    std::vector<T> &Tensor<T>::data() {
        return data_;
    }


    template<typename T>
    std::vector<T> &Tensor<T>::grad() {
        return grad_;
    }

    template<typename T>
    void Tensor<T>::zero_grad() {
        std::fill(grad_.begin(), grad_.end(), T{0});
    }

    template<typename T>
    const typename Tensor<T>::Shape &Tensor<T>::shape() const {
        return shape_;
    }
}


#endif //TINYML_TENSOR_H
