# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import threading
from absl.testing import absltest
import jax
from jax import numpy as jnp
from qwix._src import interception


class InterceptionTest(absltest.TestCase):

  def test_interception_recursion(self):
    def func1(x):
      return jnp.sin(x)

    def func2(x):
      return func1(x) + 1

    self.assertEqual(func2(0), 1.0)

    def replaced_sin(x):
      # We could still call the original sin() function.
      return jnp.sin(x) + 10

    interceptor = lambda: {"jax.numpy.sin": replaced_sin}

    with self.subTest("single_interceptor"):
      func1 = interception.wrap_func_intercepted(
          func1,
          interceptor,
          output_transform=lambda x: x + 100,
      )
      self.assertEqual(func1(0), 110.0)
      self.assertEqual(func2(0), 111.0)

    # The same interceptor can be applied to multiple functions but only one
    # interception is applied.
    with self.subTest("single_interceptor_multiple_functions"):
      func2 = interception.wrap_func_intercepted(
          func2,
          interceptor,
          output_transform=lambda x: x + 1000,
      )
      # Because func2 is intercepted, the interception of func1 is not applied.
      self.assertEqual(func2(0), 1011.0)

    # However, different interceptors can be applied to the same function.
    with self.subTest("multiple_interceptors"):
      replaced_sin2 = lambda x: jnp.sin(x) * 2 + 0.1
      interceptor2 = lambda: {"jax.numpy.sin": replaced_sin2}
      func2 = interception.wrap_func_intercepted(
          func2,
          interceptor2,
          output_transform=lambda x: x + 10000,
      )
      # The output of func2 should be
      #    jnp.sin(0) = 0 => the original sin() is called
      #   * 2 + 0.1 = 0.1 => replaced_sin2 because interceptor2 is applied first
      #       + 10 = 10.1 => replaced_sin
      #                   => interceptor on func1 is ignored.
      #        + 1 = 11.1 => func2
      #   + 1000 = 1011.1 => output_transform of interceptor on func2
      # + 10000 = 11011.1 => output_transform of interceptor2 on func2
      self.assertEqual(func2(0), 11011.1)

  def test_interception_thread_local(self):
    # Interception is thread-local.
    lock1 = threading.Lock()
    lock2 = threading.Lock()
    lock1.acquire()
    lock2.acquire()

    def sin3(x):
      return jax.lax.sin(x) + 3  # jax.lax.sin is restored.

    def func(x):
      lock1.release()  # Order: 1
      lock2.acquire()  # Order: 4
      return jnp.astype(jnp.sin(x), jnp.int8, copy=True)

    func3 = interception.wrap_func_intercepted(
        func,
        lambda: {
            "jax.numpy.sin": sin3,
            "jax.lax.sin": lambda: 42,
            "jax.numpy.astype": lambda x, dtype, *, copy: x,
        },
    )
    res = []
    th = threading.Thread(target=lambda: res.append(func3(0.0)), daemon=True)
    th.start()
    lock1.acquire()  # Order: 2
    self.assertEqual(jnp.sin(0.0), 0)
    self.assertEqual(jax.lax.sin(0.0), 0)
    self.assertEqual(jnp.astype(0.0, jnp.int8, copy=True), 0)
    lock2.release()  # Order: 3
    th.join()  # Order: 5
    self.assertEqual(res[0], 3)

  def test_interception_of_code_object(self):
    def replaced_sin(x):
      return jax.lax.sin(x) + n

    self.assertEmpty(jax.lax.sin.__code__.co_freevars)
    self.assertNotEmpty(replaced_sin.__code__.co_freevars)

    # ValueError: sin() requires a code object with 0 free vars, not 1
    with self.assertRaises(ValueError):
      jax.lax.sin.__code__ = replaced_sin.__code__

    alias_sin = jax.lax.sin

    def func(x):
      return alias_sin(x)

    # jax.lax.sin is a function, so the interception is applied to its code
    # object, making alias_sin also intercepted.
    func2 = interception.wrap_func_intercepted(
        func, lambda: {"jax.lax.sin": replaced_sin}
    )
    n = 42
    self.assertEqual(func2(0.0), 42)

  def test_interception_of_pjit_function(self):
    alias_div = jnp.true_divide  # which is a PjitFunction

    def func(x, y):
      return alias_div(x, y)

    func = interception.wrap_func_intercepted(
        func, lambda: {"jax.numpy.true_divide": lambda x, y: x / y + 1}
    )
    self.assertEqual(func(0.0, 1.0), 1.0)

  def test_double_interception(self):
    def func(x):
      return x

    func = interception.wrap_func_intercepted(
        func,
        # swish is an alias of silu.
        lambda: {"jax.nn.swish": lambda x: 42, "jax.nn.silu": lambda x: 43},
    )
    with self.assertRaises(ValueError):
      func(1.0)

  def test_intercept_class_method(self):
    def func(x):
      return jax.nn.relu(x)

    def custom_jvp_call(self, *args, **kwargs):
      return self(*args, **kwargs) + 1

    func = interception.wrap_func_intercepted(
        func, lambda: {"jax.custom_jvp.__call__": custom_jvp_call}
    )
    self.assertEqual(func(-1.0), 1.0)

  def test_scan_custom_vjp(self):
    @jax.custom_vjp
    def replaced_sin(x):
      return replaced_sin_fwd(x)[0]

    # When replaced_sin is called inside a scan, the replaced_sin_fwd function
    # will be called in another code path, e.g.
    #
    #  File "interception_test.py", in func
    #    _, y = jax.lax.scan(lambda carry, x: (carry, jnp.sin(x)), 0.0, x[None])
    #  <... a bunch of Jax code ...>
    #  File "interception_test.py", in replaced_sin_fwd
    #    return jnp.sin(x) + 1.0, ()
    #
    # This causes the jnp.sin in replaced_sin_fwd to also be intercepted.
    # There's no better way to detect this, because we do want to intercept
    # replaced_sin_fwd if it's defined by the user. So we have to manually
    # disable the interception here.
    @interception.disable_interceptions
    def replaced_sin_fwd(x):
      return jnp.sin(x) + 1.0, ()

    def replaced_sin_bwd(res, g):
      del res
      return (jnp.sin(g - 1),)

    replaced_sin.defvjp(replaced_sin_fwd, replaced_sin_bwd)

    interceptor = lambda: {"jax.numpy.sin": replaced_sin}

    def func(x):
      x = jnp.asarray(x)
      _, y = jax.lax.scan(lambda carry, x: (carry, jnp.sin(x)), 0.0, x[None])
      return y.squeeze()

    func = interception.wrap_func_intercepted(func, interceptor)
    self.assertEqual(replaced_sin(0.0), 1.0)
    self.assertEqual(func(0.0), 1.0)
    out, grads = jax.value_and_grad(func)(0.0)
    self.assertEqual(out, 1.0)
    self.assertEqual(grads, 0.0)


if __name__ == "__main__":
  absltest.main()
