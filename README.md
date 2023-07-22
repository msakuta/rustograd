# Rustograd

An experimental implementation of autograd in Rust

## Overview

Neural networks heavily rely on differentiation.
In a previous project of mine [DeepRender](https://github.com/msakuta/DeepRender), I implemented derivatives of each activation function by hand, but practical deep learning libraries usually comes with automatic differentiation (autograd).

Inspired by [this video](https://youtu.be/VMj-3S1tku0), it seems not too difficult to build such a library myself, so I gave it a try.

## Usage

First, build an expression with usual Rust arithmetics, but wrap the value in `Term::Value`.
Note that you need to take a reference (like `&a`) to apply arithmetics due to how operator overloading works in Rust.

```rust
    let a = TermInt::new(123.);
    let b = TermInt::new(321.);
    let c = TermInt::new(42.);
    let ab = &a + &b;
    let abc = &ab * &c;
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
