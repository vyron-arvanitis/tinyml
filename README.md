# tinyml

`tinyml` is my personal learning project for building small machine learning pieces in C++.

This is not meant to be a production ML library. The goal is to learn by implementing the pieces myself: matrix operations, linear regression, optimization, memory layout, and eventually a small tensor/autograd system.

## Current State

The project currently focuses on:

- basic matrix operations
- linear regression
- loss computation
- gradient-based parameter updates
- small experiments with OpenMP parallelization

The training code is still mostly manual. For example, linear regression computes gradients directly and updates weights during each epoch.

## Direction

The next step is to introduce a `Tensor` or `Parameter` abstraction.

The plan is to move gradually:

1. Keep the current manual gradients working.
2. Add tensor storage with flat contiguous memory.
3. Store shape, strides, data, gradients, and `requires_grad`.
4. Update optimizers so they work with parameters instead of raw values.
5. Later explore a small computation graph and autograd system.

The longer-term idea is to understand how frameworks like PyTorch organize tensors, gradients, and backward passes, but at a much smaller scale.

## Notes

This repository is mainly for learning C++, memory management, numerical code, and ML internals. Code may change often as I experiment with better designs.
