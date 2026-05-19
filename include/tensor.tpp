//
// Created by Vyron Arvanitis on 30.04.26.
//
#pragma once

#ifndef TINYML_TENSOR_H
#define TINYML_TENSOR_H
#include <vector>
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

            // Enables brace-initialization:
            //
            //     Shape s{2, 3, 4};
            //
            // std::initializer_list is used specifically for brace syntax.
            // The values are passed as std::initializer_list and copied into dims
            Shape(const std::initializer_list<size_t> dims)
                : dims_(dims) {
            }

            // Common modern C++ idiom:
            //
            // Accept by value, then move into the member.
            //
            // If caller passes a lvalue:
            //     std::vector<size_t> v{2,3,4};
            //     Shape s(v);
            // one copy happens into the parameter.
            //
            // If caller passes a temporary/rvalue:
            //     Shape s(std::vector<size_t>{2,3,4});
            // the vector can be moved efficiently.
            //
            // std::move transfers the vector's internal buffer into dims_
            // instead of copying all elements.
            //
            // explicit prevents accidental automatic conversions:
            //     Shape s = std::vector<size_t>{2,3,4}; // ERROR
            //     Shape s(std::vector<size_t>{2,3,4});  // OK
            explicit Shape(std::vector<size_t> dims)
                : dims_(std::move(dims)) {
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

            bool operator!=(const Shape &other) const {
                // return dims_ != other.dims_;
                return !(*this == other);
            }

            size_t operator[](const size_t i) const { return dims_[i]; }

            Shape broadcast_shape(const Shape &a, const Shape &b) {
                size_t ndim = std::max(a.ndim(), b.ndim());

                std::vector<size_t> result(ndim);
                for (size_t i = 0; i < ndim; ++i) {
                    size_t a_dim = (i < ndim - a.ndim()) ? 1 : a.dims_[i - (ndim - a.ndim())];
                    size_t b_dim = (i < ndim - b.ndim()) ? 1 : b.dims_[i - (ndim - b.ndim())];

                    if (a_dim == b_dim) {
                        result[i] = a_dim;
                    } else if (a_dim == 1) {
                        result[i] = b_dim;
                    } else if (b_dim == 1) {
                        result[i] = a_dim;
                    } else {
                        throw std::invalid_argument("Shapes are not broadcastable");
                    }
                }
                return Shape(result);
            }

            // std::vector<size_t> broadcast_index(
            //       std::initializer_list<size_t> out_index,
            //       const Shape &input_shape,
            //       const Shape &output_shape) {
            //
            //     std::vector<size_t> result;
            //     return result;
            // }
        };

        explicit Tensor(const Shape &shape);

        Tensor(const Shape &shape, const std::vector<T> &data);

        Tensor(const Shape &shape, const std::vector<T> &data, bool requires_grad_);

        //--------------------------//
        /*REGION DEFINING OF OPERATORS BETWEEN TENSORS*/
        // Return reference to underlying data (no copy).
        // const → caller cannot modify the tensor through this access.
        const T &operator()(std::initializer_list<size_t> indices) const;

        T &operator()(std::initializer_list<size_t> indices);

        const T &operator()(const std::vector<size_t> &indices) const;

        T &operator()(const std::vector<size_t> &indices);

        bool operator==(const Tensor &other) const;

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

        T sum() const;

        T mean() const;

    private:
        std::vector<T> data_;
        std::vector<T> grad_;
        Shape shape_;
        std::vector<size_t> strides_;
        bool requires_grad_ = false;

        static std::vector<size_t> _compute_strides(const Shape &shape);

        void compute_strides();

        void validate_size() const;

        size_t offset(std::initializer_list<size_t> indices) const;

        size_t offset(const std::vector<size_t> &indices) const;

        Shape unravel_index(size_t flat_index, const Shape &shape);
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
    std::vector<size_t> Tensor<T>::_compute_strides(const Shape &shape) {
        // ROW-MAJOR STRIDES

        const size_t ndim = shape.ndim();
        std::vector<size_t> strides(ndim);

        if (ndim == 0) {
            return strides;
        }

        strides[ndim - 1] = 1;

        for (size_t i = ndim - 1; i > 0; --i) {
            strides[i - 1] = strides[i] * shape[i];
        }
        return strides;
    }

    template<typename T>
    void Tensor<T>::compute_strides() {
        strides_ = _compute_strides(shape_);
    }

    template<typename T>
    void Tensor<T>::validate_size() const {
        if (data_.size() != shape_.numel())
            throw std::invalid_argument("Data size does not match Tensor shape");
    }


    // Map standard brace-enclosed multi-index to flat 1D memory offset
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

    // Map dynamic/container multi-index to flat 1D memory offset
    template<typename T>
    size_t Tensor<T>::offset(const std::vector<size_t> &indices) const {
        if (indices.size() != shape_.ndim()) {
            throw std::invalid_argument("Indices size does not match Tensor shape");
        }
        size_t offset = 0;
        for (size_t dim = 0; dim < indices.size(); ++dim) {
            size_t idx_val = indices[dim];

            if (idx_val >= shape_[dim]) {
                throw std::invalid_argument("Index out of bounds");
            }

            offset += strides_[dim] * idx_val;
        }
        return offset;
    }

    template<typename T>
    typename Tensor<T>::Shape Tensor<T>::unravel_index(size_t flat_index, const Shape &shape) {
        if (flat_index >= shape_.numel()) {
            throw std::invalid_argument("Index out of bounds");
        }

        Shape result;
        const std::vector<size_t> strides = _compute_strides(shape);
        result.dims_.resize(shape.ndim());

        for (size_t i = 0; i < shape.ndim(); ++i) {
            result.dims_[i] = flat_index / strides[i];
            flat_index %= strides[i];
        }

        return result;
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
    T &Tensor<T>::operator()(std::initializer_list<size_t> indices) {
        return data_[offset(indices)];
    }

    template<typename T>
    bool Tensor<T>::operator==(const Tensor &other) const {
        return shape_ == other.shape_ && data_ == other.data_;
    }

    template<typename T>
    const T &Tensor<T>::operator()(const std::vector<size_t> &indices) const {
        return data_[offset(indices)];
    }

    template<typename T>
    T &Tensor<T>::operator()(const std::vector<size_t> &indices) {
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

    template<typename T>
    T Tensor<T>::sum() const {
        T sum = T{};
        for (const auto &x: data_) sum += x;

        return sum;
    }

    template<typename T>
    T Tensor<T>::mean() const {
        return this->sum() / static_cast<T>(shape_.numel());
    }
}


#endif //TINYML_TENSOR_H
