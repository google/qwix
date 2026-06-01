# HiJAX

This folder contains code update to `QArray`s to use HiJAX.
HiJAX is a relatively new jax feature that allows the user to create
types which persist in a `jaxpr`. This gives finer control over how jax
deals with `QArray`s.

Note that this implementation is still in an experimental phase and
everything in this folder is subject to change without notice. After hitting
feature parity and implementing all desired functionality, we will slowly
migrate this implementation into the `qwix/_src/core` directory.

## Motivation

The overall goal is to integrate `QArray`s more closely with jax and provide
better support at the `jaxpr` level. This gives finer grained control over
how jax deals with `QArrays`.

Let `Low`, `Hi` denote low and hi precision types, resp. We typically think of
`Low` as a type where differentiation doesn't make sense (e.g. integers). The
cotangent type of `Low` is `Float0` (a trivial type).

The current `QArray` type isn't Array-like in jax.

- `QArray` is a pytree that looks roughly like
  `tuple[Array[Low], Array[Hi], Array[Hi] | None]`
- The cotangent type of this representation is
  `tuple[Array[Float0], Array[Hi], Array[Hi]]`.
  Refer to
  [jax.dtypes.float0](https://docs.jax.dev/en/latest/_autosummary/jax.dtypes.float0.html)
  for more information.
- Techniques like the straight-through estimator assume that the cotangents are
  stored as `Array`s or `QArray`s where the data is non-trivial.
- Using HiJAX, we can define the cotangent types of `QArray`s.
- Goal: Reduce reliance on `custom_vjp` for large functions.

More motivation to come!
We seek to simplify the following:

- Kernel integration
- Autograd semantics
- Integration with advanced jax features

## Some Implementation Notes

The current implementation attempts to minimize crossover between the core
library and the hijax implementation. This way we can add features without
impacting current qwix users.

This implementation uses the naming `HiQArray` for the hijax implementation of
a `QArray`.

- `hiqarray_common`: Common jax based functions and dataclasses for `HiQArray`.
- `hiquant_utils`: Common jax utilities for use in `HiQArray`.
- `hiqarray`: Coming soon.
