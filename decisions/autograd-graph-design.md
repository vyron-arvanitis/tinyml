# Autograd Graph Design Notes

## Purpose

These notes collect the current thinking about how autograd should work in this project.

The goal is not to implement full tensor autograd immediately. The goal is to understand the shape of the design before coding it:

- why autograd needs a graph
- what a node should represent
- whether we need an edge class
- where parents should be stored
- how operations like `+` and `*` fit into the graph
- why pointers appear in examples like `std::vector<Tensor*> parents_`
- how C++ ownership tools such as raw pointers, `std::shared_ptr`, and `std::unique_ptr` relate to this problem

## Main Decision: Use A DAG, Not A Linked List

Autograd should use a DAG: a directed acyclic graph.

- Directed: gradients flow from output back to inputs.
- Acyclic: an output depends on earlier values, not on itself.
- Graph: one result can depend on multiple previous values, and one value can influence the result through multiple paths.

A linked list is too limited because it only represents one chain:

```text
a -> b -> c -> loss
```

But even simple math is not always a chain:

```cpp
z = x * y + x;
```

Conceptually this is:

```text
x -----\
       multiply ---- tmp ---- add ---- z
y -----/                      /
x ----------------------------/
```

The value `x` affects `z` through two paths:

1. through `x * y`
2. directly through `+ x`

During backward, both paths must contribute to `x.grad`.

That is why autograd needs a graph instead of a linked list.

## What Is A Node?

A node represents a value in the computation.

For this project, the simplest mental model is:

```text
Tensor = data + grad + graph metadata
```

So the `Tensor` itself can act as the graph node.

A node should store:

- its numerical value, for example `data_`
- its accumulated gradient, for example `grad_`
- whether gradients are needed, for example `requires_grad_`
- pointers/references to parent nodes
- a small backward function that knows how to pass gradients to the parents
- optionally, a string name for debugging, such as `"+"`, `"*"`, `"mean"`

Conceptually:

```cpp
class Tensor {
    std::vector<double> data_;
    std::vector<double> grad_;
    bool requires_grad_;

    std::vector<Tensor*> parents_;
    std::function<void()> backward_;
    std::string op_;
};
```

This is not final code. It is the shape of the idea.

The important point is that the node stores the value and the graph connections needed for backpropagation.

## Should Nodes Store Data Or Only Pointers?

The node should store its own data.

For example, if we compute:

```cpp
c = a + b;
```

then `c` is a real tensor with its own result data. It is not only a graph marker.

So a node stores:

```text
the actual value of this tensor
the gradient of the final loss with respect to this tensor
links to the tensors that produced it
the backward rule for its operation
```

The parent list stores pointers to other nodes. The node itself still owns its own value.

## Do We Need An Edge Class?

Not at first.

A separate edge class might sound natural:

```text
Node A ---- Edge ---- Node B
```

But for simple autograd, it is usually enough for each output node to store its parents:

```cpp
std::vector<Tensor*> parents_;
```

This means:

```text
if c.parents_ = { &a, &b }
then a and b are parents of c
```

The "edges" are implicit. We know there is a connection because the parent appears in the parent list.

An explicit edge class might become useful later for:

- graph visualization
- debugging connection metadata
- storing extra information per connection
- more advanced autograd experiments

But it is extra complexity for the first version.

## What About Edge Strength?

The "strength" of a connection is the local derivative.

Example:

```cpp
z = x * y;
```

The local derivatives are:

```text
dz/dx = y
dz/dy = x
```

So the connection from `z` back to `x` depends on the value of `y`.
The connection from `z` back to `y` depends on the value of `x`.

This is why storing a simple fixed edge strength is usually not enough.

Instead, each operation stores a backward rule:

```text
for multiply:
    x.grad += y.data * z.grad
    y.grad += x.data * z.grad
```

So the local derivative is encoded inside the backward function, not usually stored as `edge.strength`.

## Where Should Parents Be Stored?

The output tensor should store its parents.

Example:

```cpp
c = a + b;
```

Then:

```text
c stores parents: a, b
```

Why the output?

Because during backward we start from the final result, usually a scalar loss:

```cpp
loss.backward();
```

Then we walk backwards:

```text
loss -> previous tensors -> previous tensors -> parameters
```

If every output remembers what created it, we can walk from the loss back to the original inputs.

## Should Parents Be `left` And `right`?

Use a vector, not only `left` and `right`.

For binary operations like `+`, `-`, `*`, `/`, there are two parents:

```text
c = a + b
c.parents = {a, b}
```

But other operations may have only one parent:

```text
y = mean(x)
y.parents = {x}
```

Later, some operations may have more than two inputs.

So this is more flexible:

```cpp
std::vector<Tensor*> parents_;
```

than this:

```cpp
Tensor* left_;
Tensor* right_;
```

## How Operations Fit Into The Graph

Each tensor operation has two jobs:

1. Forward pass: compute the output data.
2. Graph setup: if gradients are needed, store parents and a backward rule on the output.

For addition:

```cpp
c = a + b;
```

Forward:

```text
c.data = a.data + b.data
```

Backward idea:

```text
a.grad += c.grad
b.grad += c.grad
```

Because:

```text
d(a + b)/da = 1
d(a + b)/db = 1
```

For multiplication:

```cpp
c = a * b;
```

Forward:

```text
c.data = a.data * b.data
```

Backward idea:

```text
a.grad += b.data * c.grad
b.grad += a.data * c.grad
```

Because:

```text
d(a * b)/da = b
d(a * b)/db = a
```

The Tensor operation does not only compute the result. It also records how to propagate gradients from the result back to its inputs.

## Why `std::vector<Tensor*> parents_` And Not `std::vector<Tensor>`?

This is an important C++ point.

If we wrote:

```cpp
std::vector<Tensor> parents_;
```

then the vector would store copies of the parent tensors.

That is not what we want.

Suppose:

```cpp
c = a + b;
```

If `c.parents_` stores copies of `a` and `b`, then during backward we would update the copied gradients, not the original `a.grad_` and `b.grad_`.

That would be wrong.

We want `c` to point back to the original tensors.

That is why we use pointers:

```cpp
std::vector<Tensor*> parents_;
```

This means:

```text
parents_ stores addresses of tensors
not copies of tensors
```

Example:

```cpp
Tensor a;
Tensor b;

std::vector<Tensor*> parents;
parents.push_back(&a);
parents.push_back(&b);
```

Here:

- `a` is the actual object
- `&a` means "the memory address of `a`"
- `Tensor*` means "a pointer to a Tensor"

Later:

```cpp
parents[0]->zero_grad();
```

means:

```text
go to the Tensor pointed to by parents[0], then call zero_grad()
```

The arrow operator `->` is used when accessing something through a pointer.

## Why Not `std::vector<Tensor&>`?

C++ references are not normal assignable objects.

This does not work well:

```cpp
std::vector<Tensor&> parents_;
```

Standard containers such as `std::vector` need elements that can be copied, moved, assigned, and stored internally. References are aliases, not objects, so `std::vector<Tensor&>` is not a valid normal design.

There is `std::reference_wrapper<Tensor>`, but for this project it is better to understand pointers first.

So the practical choices are:

```cpp
std::vector<Tensor*> parents_;
std::vector<std::shared_ptr<Tensor>> parents_;
std::vector<std::weak_ptr<Tensor>> parents_;
```

Each has different ownership meaning.

## Raw Pointers And The Null Pointer Problem

A raw pointer is just an address.

```cpp
Tensor* p = &a;
```

It does not own the object. It only points to it.

A pointer can also point to nothing:

```cpp
Tensor* p = nullptr;
```

If you try to use a null pointer:

```cpp
p->zero_grad();
```

the program has undefined behavior and will usually crash.

So if raw pointers are used, code must be careful:

```cpp
if (p != nullptr) {
    p->zero_grad();
}
```

Raw pointers also have a lifetime problem.

Example:

```cpp
Tensor* p;

{
    Tensor a;
    p = &a;
}

// a is destroyed here
// p now points to memory that no longer contains a valid Tensor
```

This is called a dangling pointer.

For autograd, dangling pointers are a serious risk because graph nodes may point to parent tensors that no longer exist.

## `std::shared_ptr`

`std::shared_ptr<T>` is an owning smart pointer.

It means:

```text
multiple places may share ownership of the same object
the object is destroyed automatically when the last shared_ptr goes away
```

Example:

```cpp
auto a = std::make_shared<Tensor>();
auto b = a;
```

Now both `a` and `b` point to the same Tensor object, and both share ownership.

If one goes away, the object is still alive because the other still owns it.

For autograd, this can help because an output can keep its parents alive:

```cpp
std::vector<std::shared_ptr<Tensor>> parents_;
```

Meaning:

```text
this node owns/shared-owns its parents
parents will not disappear while this node still needs them
```

Potential downside:

- `shared_ptr` is heavier than raw pointers
- it can create ownership cycles if parents and children own each other

For a first learning version, `shared_ptr` is often a good way to avoid dangling pointers while learning graph logic.

## `std::unique_ptr`

`std::unique_ptr<T>` is exclusive ownership.

It means:

```text
only one unique_ptr owns this object
the object is destroyed when that unique_ptr goes away
```

Example:

```cpp
auto a = std::make_unique<Tensor>();
```

You cannot copy a `unique_ptr`:

```cpp
auto b = a; // not allowed
```

But you can move it:

```cpp
auto b = std::move(a);
```

After this:

```text
b owns the Tensor
a owns nothing
```

`unique_ptr` is good when there is one clear owner.

For autograd parent lists, `unique_ptr` is usually not the natural fit because a node may need to be referenced by several later nodes. That is shared graph structure, not exclusive ownership.

So:

- use `unique_ptr` when one object clearly owns another object
- use `shared_ptr` when several objects need to keep the same object alive
- use raw pointers when lifetime is guaranteed elsewhere and you only need a non-owning link

## Lambda Functions

Autograd often stores a small function inside each output node.

In C++, this is commonly done with a lambda.

A lambda is an unnamed function written directly where you need it.

Basic syntax:

```cpp
[capture](parameters) {
    body
}
```

Example:

```cpp
auto add = [](int a, int b) {
    return a + b;
};

int result = add(2, 3); // result is 5
```

The `[]` part is the capture list.

It controls which outside variables the lambda can use.

## Lambda With No Capture

```cpp
auto f = []() {
    return 42;
};
```

This lambda does not use any outside variables.

## Lambda Capturing By Value

```cpp
int x = 10;

auto f = [x]() {
    return x + 1;
};
```

The lambda gets its own copy of `x`.

If `x` changes later, the lambda still has the old copied value.

```cpp
int x = 10;

auto f = [x]() {
    return x;
};

x = 20;

int y = f(); // y is 10
```

## Lambda Capturing By Reference

```cpp
int x = 10;

auto f = [&x]() {
    x += 1;
};

f();
// x is now 11
```

The lambda uses the original `x`, not a copy.

This is powerful, but it has a lifetime risk.

If the lambda outlives the variable it references, it becomes dangerous.

## Lambda In Autograd

For autograd, a node might store a backward lambda:

```cpp
out.backward_ = [&a, &b, &out]() {
    a.grad += out.grad;
    b.grad += out.grad;
};
```

This means:

```text
when backward_ is called,
use a, b, and out from the surrounding scope
```

But this specific example captures by reference. If `a`, `b`, or `out` are destroyed before the lambda runs, the lambda becomes unsafe.

That is why ownership matters so much in C++ autograd.

With smart pointers, the idea may look more like:

```cpp
auto a = std::make_shared<Tensor>();
auto b = std::make_shared<Tensor>();

auto backward = [a, b]() {
    a->zero_grad();
    b->zero_grad();
};
```

Here `a` and `b` are captured by value, but the values are `shared_ptr`s.

That means the lambda keeps shared ownership of the tensors.

## `std::function<void()>`

A lambda has its own compiler-generated type.

If we want to store different lambdas in the same class field, we often use `std::function`.

Example:

```cpp
std::function<void()> backward_;
```

This means:

```text
backward_ stores any callable thing that takes no arguments and returns nothing
```

Example:

```cpp
std::function<void()> f;

f = []() {
    // do work
};

f();
```

For autograd, this is useful because:

- addition has one backward rule
- multiplication has another backward rule
- mean has another backward rule

But all of them can be stored as:

```cpp
std::function<void()> backward_;
```

## Recommended Learning Path

Do not begin with every tensor feature at once.

Recommended order:

1. Build a very small scalar autograd object first.
2. Make it store `data`, `grad`, `parents`, `backward`, and `op`.
3. Support only `+` and `*`.
4. Make `backward()` traverse the graph in reverse.
5. Only after that, move the idea into `Tensor`.

This avoids mixing too many hard problems:

- graph traversal
- pointer ownership
- lambda captures
- tensor shapes
- broadcasting
- matrix operations
- gradient accumulation

The graph idea is already enough to learn first.

