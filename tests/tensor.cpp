//
// Created by Vyron Arvanitis on 05.05.26.
//

#include <gtest/gtest.h>
#include "tensor.tpp"

using tinyml::Tensor;

TEST(TensorTest, ConstructsWithShape) {
    Tensor<double> t({2, 3});

    EXPECT_EQ(t.shape().ndim(), 2);
    EXPECT_EQ(t.shape().numel(), 6);
    EXPECT_EQ(t.data().size(), 6);
}

TEST(TensorTest, AddsTwoTensors) {
    Tensor<double> a({3}, {1.0, 2.0, 3.0});
    Tensor<double> b({3}, {4.0, 5.0, 6.0});

    Tensor<double> c = a + b;

    EXPECT_EQ(c.data(), std::vector<double>({5.0, 7.0, 9.0}));
}
