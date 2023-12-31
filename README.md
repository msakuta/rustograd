# Rustograd

An experimental implementation of reverse-mode, define-and-run autograd in Rust

## Overview

Neural networks heavily rely on differentiation.
In a previous project of mine [DeepRender](https://github.com/msakuta/DeepRender), I implemented derivatives of each activation function by hand, but practical deep learning libraries usually comes with automatic differentiation (autograd).

Inspired by [Andrej Karpathy's video](https://youtu.be/VMj-3S1tku0), it seems not too difficult to build such a library myself, so I gave it a try.

**NOTE**: This project is experimental research project. If you are looking for a serious autograd library, check out [rust-autograd](https://github.com/raskr/rust-autograd).

## Usage

First, allocate an object of type called `Tape`.
You can create variables with `Tape::term` method.
Then you can build an expression with usual Rust arithmetics.

```rust
let tape = rustograd::Tape::new();
let a = tape.term("a", 123.);
let b = tape.term("b", 321.);
let c = tape.term("c", 42.);
let ab = a + b;
let abc = ab * c;
```

Next, you can derive the expression with any variable and get the coefficient.

```rust
let abc_a = abc.derive(&a);
println!("d((a + b) * c) / da = {}", abc_a); // 42
let abc_b = abc.derive(&b);
println!("d((a + b) * c) / db = {}", abc_b); // 42
let abc_c = abc.derive(&c);
println!("d((a + b) * c) / dc = {}", abc_c); // 444
```

Lastly, you can call `backprop` to update all terms at once, much more efficiently than calling `derive` for every one of them.

```rust
abc.backprop();
```

It's a little easier to see it with Graphviz than the console, so output the `.dot` file like this:

```rust
abcd.dot(&mut std::io::stdout()).unwrap();
```

Copy and paste the output into [Graphviz online](https://dreampuf.github.io/GraphvizOnline).

![graphviz](images/graphviz.svg)

Animated sequence of forward and backpropagation:

![animated graphviz](images/backprop.gif)


## Rc, reference and tape versions

Rustograd terms come in three flavors.

* reference-based terms, `Term<'a>`.
* Rc-based terms, `RcTerm`.
* Tape memory arena based terms, `TapeTerm`.

The reference-based term is more efficient when you run the calculation only once, since it doesn't have reference counting overhead.
However, it has very strict restrictions that is difficult to scale, therefore deprecated.

First, you need to take a reference (like `&a`) to apply arithmetics due to how operator overloading works in Rust.
Second, every intermediate term is required to live as long as the expression is evaluated.
It means you can't even compile a function below, because the temporary variable `b` will be dropped when the function returns.

```rust
fn model<'a>() -> (Term<'a>, Term<'a>) {
    let a = Term::new("a", 1.);
    let b = Term::new("b", 2.);
    let ab = &a * &b;
    (a, ab)
}
```

`RcTerm` works even in this case since it is not bounded by any lifetime:

```rust
fn model() -> (RcTerm, RcTerm) {
    let a = RcTerm::new("a", 1.);
    let b = RcTerm::new("b", 2.);
    let ab = &a * &b;
    (a, ab)
}
```

It is especially handy when you want to put the expression model into a struct, because it would require self-referential struct with `Term<'a>`.
Current Rust has no way of constructing a self-referential struct explicitly.
Also you don't have to write these lifetime annotations.

`TapeTerm` uses a shared memory arena called the `Tape`.
It removes the need of explicit management of the lifetimes and has ergonomic arithmetic operation (since it is a copy type), but you need to pass around the arena object.
See [this tutorial](https://rufflewind.com/2016-12-30/reverse-mode-automatic-differentiation) for more details.

```rust
fn model(tape: &Tape) -> (TapeTerm, TapeTerm) {
    let a = tape.term("a", 1.);
    let b = tape.term("b", 2.);
    let ab = a * b;
    (a, ab)
}
```

Generally `RcTerm` is the most convenient to use, but it adds some cost in reference counting.
`TapeTerm` is fairly convenient and if you don't mind carrying around a reference to `tape` object everywhere, probably it is the most efficient and ergonomic way.
The tape is expected to be more efficient because the nodes are allocated in a contiguous memory and they will keep being accessed throughout training.

## Adding a unary function

You can add a custom function in the middle of expression tree.

For example, you can get a term `sin(a)` from `a` by calling `apply` method.
You need to supply with the function and its derivative as function pointers.

```rust
    let a = Term::new("a", a_val);
    let sin_a = a.apply("sin", f64::sin, f64::cos);
```

You can get the plot of the derivative of the expression by evaluating with various values for the input variable.

```rust
    for i in -10..=10 {
        let x = i as f64 / 10. * std::f64::consts::PI;
        a.set(x).unwrap();
        sin_a.eval();
        println!("[{x}, {}, {}],", sin_a.eval(), sin_a.derive(&a));
    }
```

![sine](images/sine.png)

Of course, this is so stupidly simple example, but it can work with more complex expression.

```rust
    let a = Term::new("a", a_val);
    let sin_a = a.apply("sin", f64::sin, f64::cos);
    let ten = Term::new("5", 5.);
    let b = &a * &ten;
    let c = Term::new("c", 0.2);
    let sin_b = b.apply("sin", f64::sin, f64::cos);
    let c_sin_b = &c * &sin_b;
    let all = &sin_a + &c_sin_b;
```

![mixed_sine_graphviz](images/mixed_sine_graphviz.svg)

![sine](images/mixed_sine.png)

See [mixed_sine.rs](examples/mixed_sine.rs) for the full example.


## A procedural macro to build expression

It is pretty tedious to write a complex expression like the one above.
There is a feature flag `macro` which can simplify writing it.

For example, the example above can be written like this:

```rust
use rustograd_macro::rustograd;

rustograd! {{
    let a = 0.;
    let all = sin(a) + 0.2 * sin(a * 5.);
}}
```

You need to enable the feature flag like below:

```
cargo r --features macro --example mixed_sine_macro
```

For a reason in `syn` crate's design, the `rustograd!` macro needs to wrap the contents in double braces `{{}}`.

You need to define functions derivatives by postfixing `_derive` to automatically bind derivatives of the function.
For example, in the example above, we need `sin` function.
Its derivative shall have a name `sin_derive`, like below.

```rust
fn sin(x: f64) -> f64 { x.sin() }
fn sin_derive(x: f64) -> f64 { x.cos() }
```

## Generating derived sub-graph for higher order differentiation

You can use `TapeTerm::gen_graph()` method to generate a node that represents a new sub-expression which has differentiation of the original variable.
This can be applied recursively to generate higher order differentiation, or even mixed order of differentiations in an expression.
For example, you can represent a function like below.

$$
f(x) = g(x) + \frac{\partial g(x)}{\partial x}
$$

The plot below shows a series of higher order differentiation on a Gaussian function.

![higher order Gaussian differentiations](images/higher-order-gaussian.png)

To use this feature, you need to provide with an implementation of a trait called `UnaryFn` for the functions.
See [the example code](examples/tape_gen_graph_gaussian.rs) for how to do that.

Note that this feature is not implemented in `RcTerm` yet.


## `Dvec` type for higher order differentiation

`*Term` types do not support higher order differentiation. It's hard to support them in reverse-mode automatic differentiation, and they require a buffer size proportional to the maximum order of differentiation you would want to calculate, so it is inefficient to allocate memory for all of them.

This library provides with `Dvec` type, which is a direct translation from a paper [^1].

[^1]: [Higher Order Automatic Differentiation with Dual Numbers](https://pp.bme.hu/eecs/article/download/16341/8918/87511)

You can find a usage example [with scalars](examples/dvec.rs) and [with tensors](examples/dvec_tensor.rs) in the examples directory.


## Example applications

In the following examples, I use `TapeTerm` because it is expected to be the most efficient one in complex expressions.

### Curve fitting

[examples/tape_curve_fit.rs](examples/tape_curve_fit.rs)

Let's say, we have a set of measured data that supposed to contain some Gaussian distribution.
We don't know the position and spread (standard deviation) of the Gaussian.
We can use least squares fitting to determine the parameters $\mu, \sigma, s$ in the expression below.

$$
f(x; \mu, \sigma, s) = s \exp \left(-\frac{(x - \mu)^2}{\sigma^2} \right)
$$

To apply least squares fitting, we define the loss function like below (denoting $x_i, y_i$ as the $i$th sample).

$$
L = \sum_i (f(x_i) - y_i)^2
$$

We could calculate the gradient by taking partial derivates of the loss function with respect to each parameter and descend by some descend rate $\alpha$, but it is very tedious to calculate by hand (I mean, it's not too bad with this level, but if you try to expand this method, it quickly becomes unmanageable).

$$
\begin{align*}
\mu &\leftarrow \mu - \alpha \frac{\partial L}{\partial \mu} \\
\sigma &\leftarrow \sigma - \alpha \frac{\partial L}{\partial \sigma} \\
s &\leftarrow s - \alpha \frac{\partial L}{\partial s}
\end{align*}
$$

Here autograd comes to rescue! With autograd, all you need is to build the expression like below and call `loss.backprop()`.

```rust
    let x = tape.term("x", 0.);
    let mu = tape.term("mu", 0.);
    let sigma = tape.term("sigma", 1.);
    let scale = tape.term("scale", 1.);
    let x_mu = x - mu;
    let gaussian = scale * (-(x_mu * x_mu) / sigma / sigma).apply("exp", f64::exp, f64::exp);
    let sample_y = tape.term("y", 0.);
    let diff = gaussian - sample_y;
    let loss = diff * diff;
```

Below is an animation of gradient descent in action, using gradient calculated by autograd.

![curve_fit](images/curve_fit.gif)

The computational graph is like below.

![curve_fit_graph](images/backprop_curve_fit.gif)

It may not seem so impressive since you can estimate the parameters directly from sample mean and standard deviation like below if the distribution is a Gaussian, but it gets more interesting from the next section.

$$
\begin{align*}
\mu &= \frac{1}{N}\sum_i x \\
\sigma &= \sqrt{\frac{1}{N - 1}\sum_i (x - \mu)^2}
\end{align*}
$$


### Peak separation

[examples/peak_separation.rs](examples/peak_separation.rs)

Another important application often comes up with measurement is peak separation.
It is similar to curve fitting, but the main goal is to identify each parameter from a mixed signal of Gaussian distributions.

$$
f(x; \mathbf{\mu}, \mathbf{\sigma}, \mathbf{s}) = \sum_k s_k \exp \left(-\frac{(x - \mu_k)^2}{\sigma_k^2} \right)
$$

The model is quite similar to the previous example, but there are 2 Gaussian distributions, which shares the structure, so I used a lambda to factor it.

```rust
    let x = tape.term("x", 0.);

    let gaussian = |i: i32| {
        let mu = tape.term(format!("mu{i}"), 0.);
        let sigma = tape.term(format!("sigma{i}"), 1.);
        let scale = tape.term(format!("scale{i}"), 1.);
        let x_mu = x - mu;
        let g = scale * (-(x_mu * x_mu) / sigma / sigma).apply("exp", f64::exp, f64::exp);
        (mu, sigma, scale, g)
    };

    let (mu0, sigma0, scale0, g0) = gaussian(0);
    let (mu1, sigma1, scale1, g1) = gaussian(1);
    let y = g0 + g1;

    let sample_y = tape.term("y", 0.);
    let diff = y - sample_y;
    let loss = diff * diff;
```


![peak_separation](images/peak_separation.gif)

At this point, the computation graph becomes so complicated that I won't even bother calculating by hand. However, autograd keeps information of factored values and do not repeat redundant calculation.

![peak_separation_graph](images/backprop_peak_separation.gif)

There is one notable thing about this graph.
The variable $x$ is shared among 2 Gaussians, so it appears as a node with 2 children.
This kind of structure in the computational graph is automatically captured with autograd.

Also there are 2 arrows from `(x - mu0)` to `(x - mu0) * (x - mu0)` because we repeat the same expression to calculate square.

![peak_separation_graph_zoomed](images/peak_separation_graph_zoomed.png)


### Path smoothing

You can find the source code example [here](examples/tape_path_smooth.rs)

Another common (and relatively easy) problem is the path smoothing.
Suppose we have a path of points (a list of x, y coordinates pairs).
Our goal is to smooth the path with a given potential.

In this example, we put a Gaussian potential, whose mean is at (1, 1).
We need to minimize the total length of the path, but at the same time, we need to avoid the source of the potential.

The update rule can be written like this:

$$
\vec{x}_{i, t+1} = \vec{x}_{i, t} - \nabla p(\vec{x}_{i, t}) - \frac{\partial L}{\partial \vec{x}_{i, t }}
$$

where $\vec{x}_{i, t}$ is a position vector of node $i$ at iteration $t$.

The potential $p(\vec{x})$ here is a Gaussian function, but you can use any function.

$$
p(\vec{x}; \vec{\mu}, \sigma, s) = -s \exp\left(-\frac{(\vec{x} - \vec{\mu})^2}{\sigma^2}\right)
$$

The cost function for the total distance $L$ is defined as follows:

$$
L = \sum_i (\vec{x}_{i + 1} - \vec{x}_i)^2
$$

![path-smoothing](images/path_smooth.gif)

![path-smoothing](images/path_smooth.svg)


### Classification

A binary class classification problem, found in [tape_classifier.rs](examples/tape_classifier.rs).

It is a typical applicatoin in a machine learning, where autograd can be useful.

For simplicity, we will present a 2-class classification problem example.
Suppose we had $t$ samples with $x$ as class labels $(x_i, t_i) (i=0,1,..., N)$ as training samples.
The weights of a 1-layer neural networks can be written in a matrix form $W$.
The bias term is written as $\mathbf{b}$.
Then the output of the first layer can be written as:

$$
\mathbf{s} = W x + \mathbf{b}
$$

Consider a case of linearly separable problem, so that only the first layer is sufficient to classify.
Then we can put the $s$ above into the softmax function:

$$
y_k = \frac{\exp{s_k}}{\sum_j \exp{s_j}}
$$

where $f_k$ is the predicted probability of class $k$ and $s_k$ is the $k$-th element of the vector $\mathbf{s}$.

We put it through the cross entropy function:

$$
E(W) = -\sum_{n=0}^N \{ t_n \ln y_n + (1 - t_n) \ln(1 - y_n) \}
$$

The parameters $W$ (2 x 2 matrix) and $\mathbf{b}$ (2-vector) can be learned by automatic differentiation (backpropagation).
The animation below shows the process.
The intensity of white indicates the probability of class 0.

![classification](images/classification.gif)

You can also apply regularization to avoid overfitting.

$$
E(W) = -\sum_{n=0}^N \{t_n \ln y_n + (1 - t_n) \ln(1 - y_n) \} - \lambda || W || ^2
$$

![classification_regularized](images/classification_regularized.gif)

You can extend the method to classify 3 classes.

$$
E(W_1, ..., W_K) = -\sum_{k=1}^K t_k \ln y_k
$$

![classification_3classes](images/classification_3classes.gif)



## Performance investigation

We can measure the performance of each type of terms.

We tested with `RcTerm`, `TapeTerm<f64>` and `TapeTerm<MyTensor>`, where
`MyTensor` is custom implementation of 1-D vector.

The plot below shows comparison of various implementations.
The codes are in [rc_curve_fit](examples/rc_curve_fit.rs), [rc_tensor_curve_fit](examples/rc_tensor_curve_fit.rs), [tape_curve_fit](examples/tape_curve_fit.rs) and [tape_tensor_curve_fit](examples/tape_tensor_curve_fit.rs).

![term_perf](images/term_perf.png)

* rc - `RcTerm`, with `Cell<f64>` as the payload
* rc_refcell - `RcTerm<f64>`, with `RefCell<f64>` as the payload
* rc2_refcell - same as rc_refcell, except topological sorting of nodes in backprop
* rc_tensor - `RcTerm<MyTensor>`, with `RefCell<MyTensor>` with 80 elements as the payload
* rc_t - same as rc2_refcell, except using generic function implementation
* rc_bcast - same as rc_tensor, except using broadcast and sum operators
* tape - `TapeTerm<f64>` (rolling tape implementation)
* tape_tensor - `TapeTerm<MyTensor>`, with `MyTensor` having 80 elements as the payload (rolling tape implementation)
* tape_t - same as tape, except using generic function implementation
* tape_bcast - same as tape_tensor, except using broadcast and sum operators
* manual - a handwritten gradient function to optimize, measured as a reference

The difference between `rc` and `rc_refcell` is that the former uses `Cell` as the container of the value, while the latter uses `RefCell`.
If the value is a `Copy`, we could use `Cell` and there is no overhead in runtime borrow checking, but if it was not a `Copy`, we have to use `RefCell` to update it.
A tensor is (usually) not a copy, so we have to use `RefCell`, so I was interested in the overhead.

As you can see, `TapeTerm<f64>` is faster than `RcTerm<f64>`.
Most of the `RcTerm<f64>`'s overhead is that it needs to build a topologically sorted list to scan, while `TapeTerm` already do that during construction.
This overhead is reduced by collecting values in a tensor, as you can see in rc_tensor, because the list only needs to be built once per the whole vector.

The performance gain by changing from `f64` (scalar) to `MyTensor` is greater than the introduction of `RefCell`, and it is almost the same between `RcTerm<MyTensor>` and `TapeTerm<MyTensor>`.

It indicates that even though `MyTensor` uses additional heap memory allocation, it is still faster to aggregate operation in an expression, rather than scanning the scalar value and evaluating each of them.
Next step is to investigate if using a memory arena for the contents of the tensor helps further.

On top of that, utilizing the tape's property that every variable has no dependencies after that node makes it possible to "roll" the tape and calculate differentials in one evaluation per node.
It improves the performance even further by removing redundant visits.

The last bit of optimization comes from broadcasting and summing operators.
See the details in [this PR](https://github.com/msakuta/rustograd/pull/6).
After using these operators, `RcTerm<MyTensor>` is even faster than `TapeTerm<MyTensor>`, but I don't know why.

As a reference, I measured the speed of manually calculated gradient on paper.
See [examples/manual_curve_fit.rs](examples/manual_curve_fit.rs) for the full code.
It is kind of theoretical limit of speed (although I could factor out more to improve speed).
Automatic differentiation has inevitable overhead, but you can see that it is not too large.
