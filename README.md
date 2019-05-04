## Introduction

[![crates.io](https://img.shields.io/crates/v/ndarray-tensorflow.svg)](https://crates.io/crates/ndarray-tensorflow)
[![docs.rs](https://docs.rs/ndarray-tensorflow/badge.svg)](https://docs.rs/ndarray-tensorflow/)
[![Travis CI](https://img.shields.io/travis/danieldk/ndarray-tensorflow.svg)](https://travis-ci.org/danieldk/ndarray-tensorflow)

This crate provides a wrapper for the
[`Tensor`](https://tensorflow.github.io/rust/tensorflow/struct.Tensor.html) type
of the [`tensorflow` crate](https://crates.io/crates/tensorflow) that can create
[`ArrayView`](https://docs.rs/ndarray/0.12.1/ndarray/type.ArrayView.html) and
[`ArrayViewMut`](https://docs.rs/ndarray/0.12.1/ndarray/type.ArrayViewMut.html)
instances. This makes it possible to use tensors through the
[`ndarray`](https://crates.io/crates/ndarray) API.
