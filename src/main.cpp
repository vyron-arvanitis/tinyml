#include <iostream>
#include <cstddef>
#include <memory>
#include "matrix.h"
#include "linear_regression.h"
#include "loss.h"
#include "random_gen.h"

#include <omp.h>
#include <stdio.h>
#include <chrono>

using namespace std::chrono;

void example_linear_regression()
{
    std::cout << "The LINEAR REGRESSION EAMPLEeeeee!!!!" << "\n";
    std::size_t N = 10000;
    std::size_t F = 20;

    tinyml::Matrix X(N, F);
    tinyml::Matrix y(N, 1);

    tinyml::RNG rng(132);
    std::uniform_real_distribution<float> xdist(0.0f, 5.0f);
    std::normal_distribution<float> noise(0.0f, 0.1f); 

    // ground-truth weights (100x1) + bias
    tinyml::Matrix w_true(F, 1);
    float b_true = 1.0f;

    for (std::size_t j = 0; j < F; ++j)
    {
        w_true(j, 0) = 0.1f * static_cast<float>(j + 1); 
    }

    for (std::size_t i = 0; i < N; ++i)
    {
        float yi = b_true;

        for (std::size_t j = 0; j < F; ++j)
        {
            float x = xdist(rng);
            X(i, j) = x;
            yi += x * w_true(j, 0);
        }

        yi += noise(rng); // optional
        y(i, 0) = yi;
    }

    tinyml::LinearRegression model(0.001f, 500, std::make_unique<tinyml::MSELoss>());
    std::cout << "weights:\n"
              << model.weights().size() << "\n";
    std::cout << "bias:\n"
              << model.bias().size() << "\n";
    auto start = high_resolution_clock::now();
    model.fit(X, y);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start); // microseconds

    std::cout << "Runtime of program is " << duration.count() << "\n\n";

    tinyml::Matrix preds = model.predict(X);
    auto size_preds = preds.shape();
    std::cout << "What is the shape of predictions " << size_preds.rows << " " << size_preds.cols << "\n";

    // std::cout << "weights:\n"
    //           << model.weights() << "\n";
    // std::cout << "bias:\n"
    //           << model.bias() << "\n";

    std::cout << "first 5 rows:\n";
    for (std::size_t i = 0; i < 5; ++i)
    {
        std::cout << "x=" << X(i, 0)
                  << " y=" << y(i, 0)
                  << " y_hat=" << preds(i, 0)
                  << "\n";
    }

    // std::cout<< "The loss history is the following: " << model.loss_history() << "\n"; # shows the whole loss hsitory is messy
    float avg_loss = 0.0f;
    for(std::size_t i=0; i<model.loss_history().rows(); i++){
        avg_loss +=model.loss_history()(i,0);
    }

    avg_loss = avg_loss/ static_cast<float>(model.loss_history().rows());
    std::cout<< "The avg loss is: " << avg_loss<< "\n";
}

void parallel_loop_ex()
{

#pragma omp parallel for // num_threads()
    for (int i = 1; i <= 10; i++)
    {
        int tid = omp_get_thread_num(); // Returns current thread ID
        printf("The thread %d  executes i = %d\n", tid, i);
    }
}

int main()
{
    tinyml::Matrix m(3, 3);
    std::cout << "The rows of the Matrix are " << m.rows() << " The cols are " << m.cols() << "\n";
    auto shape_m = m.shape();
    std::cout << "The shape of the Matrix is " << shape_m.rows << " " << shape_m.cols << "\n";
    std::cout << "The first element of the matrix is : " << m.get(0, 0) << "\n";
    std::cout << "Setting the firstelemtn to : 100 \n";
    m.set(0, 0, 100);
    std::cout << "The first element of the matrix is : " << m(0, 0) << "\n";
    std::cout << "Testing THE RANDOM GEN FILE \n";

    tinyml::RNG rng(123); // or std::random_device{}()
    tinyml::Matrix n = tinyml::random_normal(3, 2, 0.0, 1.0, rng);
    std::cout << "The matrix n is \n"
              << n << "\n";

    std::cout << "The first element of the new matrix `n` is : " << n(1, 1) << "\n";

    n(1, 1) = 50;
    std::cout << "The first element (after changing it) of the new matrix `n` is : " << n(1, 1) << "\n";

    tinyml::Matrix c = m * n;
    std::cout << "Multiply matrices m x n  " << "\n";
    std::cout << "The matrix c is \n"
              << c << "\n";

    std::cout << "Multiply matrices (1X3)(3X1) " << "\n";
    tinyml::Matrix m_13 = tinyml::random_uniform(1, 3, 0, 1, rng);
    tinyml::Matrix m_31 = tinyml::random_uniform(3, 1, 0, 1, rng);
    tinyml::Matrix scalar = m_13 * m_31;
    for (std::size_t i = 0; i < scalar.rows(); ++i)
    {
        for (std::size_t j = 0; j < scalar.cols(); ++j)
        {
            std::cout << scalar(i, j) << (j + 1 == scalar.cols() ? '\n' : ' ');
        }
    }

    std::cout << "And the length of the scalar object is: " << scalar.size() << "\n";

    std::cout << " Addition of matrices (1X3)(3X1) " << "\n";
    tinyml::Matrix matrix_add = m_13 + m_13;
    std::cout << "The reuslt of the addition of matrices is" << "\n";
    std::cout << "The matrix_add n is \n"
              << matrix_add << "\n";

    std::cout << " Subtraction of matrices (1X3)(3X1) " << "\n";
    tinyml::Matrix matrix_sub = m_13 * 4 - 2 * m_13;
    std::cout << "The matrix_sub n is \n"
              << matrix_add << "\n";

    example_linear_regression();
    parallel_loop_ex();

    return 0;
}
