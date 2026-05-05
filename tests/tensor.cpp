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


TEST(TensorTest, AccessElements) {
    const std::vector<int> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    Tensor<int> t({2, 2, 3}, data);
    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
            for (size_t k = 0; k < 3; k++) {
                const size_t idx = i * 2 * 3 + j * 3 + k;
                EXPECT_EQ(t({i, j, k}), data[idx]);
            }
        }
    }
}

TEST(TensorTest, AssignElements) {
    Tensor<size_t> t({2, 2, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    t({0, 0, 0}) = 5;
    t({1, 1, 2}) = 40;
    EXPECT_EQ(t({0, 0, 0}), 5);
    EXPECT_EQ(t({1, 1, 2}), 40);
}

TEST(TensorTest, EqualityOfTensors) {
    Tensor<double> a({3}, {1.0, 2.0, 3.0});
    Tensor<double> b({3}, {2.0, 2.0, 3.0});
    Tensor<double> c({1, 3}, {1.0, 2.0, 3.0});
    EXPECT_NE(a, b);
    EXPECT_NE(a, c);
}

TEST(TensorTest, AddsTwoTensors) {
    Tensor<double> a({3}, {1.0, 2.0, 3.0});
    Tensor<double> b({3}, {4.0, 5.0, 6.0});

    Tensor<double> c = a + b;

    EXPECT_EQ(c.data(), std::vector<double>({5.0, 7.0, 9.0}));
    EXPECT_EQ(a.data(), std::vector<double>({1.0, 2.0, 3.0}));
    EXPECT_EQ(b.data(), std::vector<double>({4.0, 5.0, 6.0}));
}

TEST(TensorTest, AddAssignsTwoTensors) {
    Tensor<double> a({3}, {1.0, 2.0, 3.0});
    Tensor<double> b({3}, {4.0, 5.0, 6.0});

    a += b;

    EXPECT_EQ(a.data(), std::vector<double>({5.0, 7.0, 9.0}));
    EXPECT_EQ(b.data(), std::vector<double>({4.0, 5.0, 6.0}));
}

TEST(TensorTest, SubtractsTwoTensors) {
    Tensor<double> a({3}, {1.0, 2.0, 3.0});
    Tensor<double> b({3}, {4.0, 5.0, 6.0});

    Tensor<double> c = a - b;
    EXPECT_EQ(c.data(), std::vector<double>({-3.0, -3.0, -3.0}));

}

TEST(TensorTest, SubtractAssignsTwoTensors) {
    Tensor<double> a({3}, {1.0, 2.0, 3.0});
    Tensor<double> b({3}, {4.0, 5.0, 6.0});

    a -= b;

    EXPECT_EQ(a.data(), std::vector<double>({-3.0, -3.0, -3.0}));
    EXPECT_EQ(b.data(), std::vector<double>({4.0, 5.0, 6.0}));
}
