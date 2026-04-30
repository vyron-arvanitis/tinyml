# Where This Is Going

## Decision

We will grow the project toward a tiny PyTorch-like training system, but we will not jump directly into full autograd for tensors and matrices.

The next step is to build a `Tensor` or `Parameter` abstraction that stores values and gradients. The current manual gradient code can keep working, but gradients should eventually be written into these objects instead of being handled as loose variables inside each training loop.

This gives us a clean migration path:

1. Keep manual gradients while the math is still simple.
2. Introduce tensors/parameters that own `data`, `grad`, and `requires_grad`.
3. Make optimizers update parameters generically.
4. Later add computation graph metadata and `backward()`.

## Why Not Start With Full Autograd Yet

Full autograd is the long-term interesting idea, but starting there adds several design problems at once:

- graph ownership
- operation nodes
- backward traversal order
- gradient accumulation
- broadcasting rules
- matrix and vector gradient shapes
- memory cleanup

It is better to first make the training code use parameter objects cleanly, then add graph behavior once the basic API feels right.

## Target Shape

Eventually the training code should move toward something like this:

```cpp
Tensor w;
Tensor b;

Tensor loss = model.forward(X, y);

loss.backward();     // later, when autograd exists
optimizer.step();    // updates parameters from their gradients
optimizer.zero_grad();
```

At first, `loss.backward()` may not exist. The model can still manually compute gradients and write them into:

```cpp
w.grad
b.grad
```

The important part is that the optimizer should not need to know the exact model. It should only know how to update a list of parameters.

## Suggested Build Checklist

- [ ] Create a `Tensor` or `Parameter` class.
- [ ] Store the value/data inside the class.
- [ ] Store the gradient inside the class.
- [ ] Store tensor shape metadata.
- [x] General N-dimensional tensors.
- [ ] Use contiguous row-major storage for tensor data.
- [ ] Add indexing helpers so the rest of the code does not manually compute offsets everywhere.
- [ ] Add `requires_grad`.
- [ ] Add `zero_grad()`.
- [ ] Update linear regression so `w` and `b` are parameters.
- [ ] Keep the current manual gradient formulas for now.
- [ ] Write the computed `dL/dw` and `dL/db` into `w.grad` and `b.grad`.
- [ ] Update the optimizer so it receives parameters instead of raw weights.
- [ ] Make `optimizer.step()` update all registered parameters.
- [ ] Make `optimizer.zero_grad()` reset all parameter gradients.
- [ ] Keep `loss_history` as a simple vector of scalar losses.
- [ ] Add a small test or example proving that training still reduces loss.

## Later Autograd Checklist

- [ ] Add parent/dependency tracking to tensor-like values.
- [ ] Store which operation produced each value.
- [ ] Store a backward function for each operation.
- [ ] Implement reverse graph traversal.
- [ ] Start with scalar autograd before matrix autograd.
- [ ] Support simple operations first: add, multiply, power, mean.
- [ ] Add matrix operations only after scalar autograd is solid.

## Current Direction

Start by building the `Tensor` or `Parameter` class.

Do not try to make it a full computation graph yet. Give it the fields that autograd will need later, but keep the first implementation simple enough that linear regression can move onto it without becoming messy.

## Memory And Tensor Design

The tensor step is also where memory efficiency starts. Autograd will add graph memory later, but the tensor class owns the most important low-level choice: how numerical data is stored.

The main rule is to store tensor values in one contiguous block:

```cpp
std::vector<double> data;
```

Avoid storing matrices like this:

```cpp
std::vector<std::vector<double>>
```

A nested vector is easier to read at first, but it spreads rows across separate allocations. That is worse for cache locality, harder to parallelize cleanly, and less like real tensor libraries.

For the current `Matrix` class, 2D indexing probably looks conceptually like this:

```cpp
offset = row * cols + col;
```

That is row-major indexing. It means rows are stored one after another in memory.

For an N-dimensional tensor, this idea generalizes using `shape` and `strides`.

Example shape:

```text
shape = [2, 3, 4]
```

This means:

```text
2 blocks, each with 3 rows, each with 4 columns
```

The total number of elements is:

```text
2 * 3 * 4 = 24
```

For row-major storage, the strides would be:

```text
shape   = [2, 3, 4]
strides = [12, 4, 1]
```

Then indexing:

```text
tensor[1, 2, 3]
```

becomes:

```text
offset = 1 * 12 + 2 * 4 + 3 * 1
offset = 23
```

So the N-dimensional version of:

```cpp
row * cols + col
```

is:

```cpp
offset = index[0] * stride[0]
       + index[1] * stride[1]
       + index[2] * stride[2]
       + ...
       + index[n] * stride[n];
```

This means a tensor does not need nested memory for N dimensions. It can still use one flat `std::vector<double>` and map N-dimensional indexes into that flat storage.

## Tensor Vs Parameter

A `Tensor` should represent numerical data:

```cpp
data
shape
strides
grad
requires_grad
```

A `Parameter` can be a special kind of tensor used by models:

```cpp
Parameter w;
Parameter b;
```

The difference is conceptual:

- A `Tensor` can be any value: input data, predictions, loss, temporary results.
- A `Parameter` is a trainable tensor owned by a model.
- Optimizers should update `Parameter` objects.
- Not every `Tensor` should be updated by the optimizer.

For the first implementation, it is fine if `Parameter` is very small. It could wrap a `Tensor`, inherit from `Tensor`, or just be the same class with `requires_grad = true`. The important design idea is that model weights and biases should be discoverable as trainable parameters.

## Memory Checklist

- [ ] Store tensor data with tempalte T
- [ ] Store gradient data with tempalte T
- [ ] Store `shape`, for example `[rows, cols]` or `[dim0, dim1, dim2]`.
- [ ] Store `strides`, so N-dimensional indexes can map to flat memory.
- [ ] Add a helper that computes flat offsets from N-dimensional indexes.
- [ ] Keep row-major layout unless there is a strong reason to do otherwise.
- [ ] Pass tensors by `const Tensor&` when reading them.
- [ ] Update parameters in-place inside the optimizer.
- [ ] Avoid unnecessary tensor copies in training loops.
- [ ] Use OpenMP later on large contiguous loops, not tiny operations.
- [ ] Delay complicated view/slice memory sharing until the basic tensor works.
