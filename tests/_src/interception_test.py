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

    interceptor = lambda: interception.Interceptor(
        mapping={"jax.numpy.sin": replaced_sin}, id=1
    )

    with self.subTest("single_interceptor"):
      func1 = interception.wrap_func_intercepted(
          func1,
          interceptor,
          output_transform=lambda x: x + 100,
          disable_jit=True,
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
          disable_jit=True,
      )
      # Because func2 is intercepted, the interception of func1 is not applied.
      self.assertEqual(func2(0), 1011.0)

    # However, different interceptors can be applied to the same function.
    with self.subTest("multiple_interceptors"):
      replaced_sin2 = lambda x: jnp.sin(x) * 2 + 0.1
      interceptor2 = lambda: interception.Interceptor(
          mapping={"jax.numpy.sin": replaced_sin2}, id=2
      )
      func2 = interception.wrap_func_intercepted(
          func2,
          interceptor2,
          output_transform=lambda x: x + 10000,
          disable_jit=True,
      )
      # The output of func2 should be
      #    jnp.sin(0) = 0 => the original sin() is called
      #         + 10 = 10 => replaced_sin is called first because it's installed
      #                      later. See _on_intercepted_called for details.
      #  * 2 + 0.1 = 20.1 => replaced_sin2 is called second.
      #                   => interceptor on func1 is ignored.
      #        + 1 = 21.1 => func2
      #   + 1000 = 1021.1 => output_transform of interceptor on func2
      # + 10000 = 11021.1 => output_transform of interceptor2 on func2
      self.assertEqual(func2(0), 11021.1)

  def test_interception_recursion_disable_jit_false(self):
    # Tests that when disable_jit=False is used, identical interceptors
    # correctly avoid recursion loops via module setattr assignments.
    def func1(x):
      return jnp.sin(x)

    def func2(x):
      return func1(x) + 1

    self.assertEqual(func2(0), 1.0)

    def replaced_sin(x):
      # We could still call the original sin() function.
      return jnp.sin(x) + 10

    interceptor = lambda: interception.Interceptor(
        mapping={"jax.numpy.sin": replaced_sin}, id=3
    )

    with self.subTest("single_interceptor"):
      func1 = interception.wrap_func_intercepted(
          func1,
          interceptor,
          output_transform=lambda x: x + 100,
          disable_jit=False,
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
          disable_jit=False,
      )
      # Because func2 is intercepted, the interception of func1 is not applied.
      self.assertEqual(func2(0), 1011.0)

  def test_interception_thread_local(self):
    # Interception is thread-local.
    lock1 = threading.Lock()
    lock2 = threading.Lock()
    lock1.acquire()
    lock2.acquire()

    def func(x, in_thread2=False):
      if in_thread2:
        lock1.release()  # Order: 1
        lock2.acquire()  # Order: 4
      return jnp.sin(x)

    func = interception.wrap_func_intercepted(
        func,
        lambda: interception.Interceptor(
            mapping={
                "jax.numpy.sin": lambda x: jax.lax.sin(x) + 42.0,
                "jax.lax.sin": lambda x: x / 0,  # This should not be called.
            },
            id=4,
        ),
        disable_jit=True,
    )

    # Run func in a separate thread and save the result in res.
    res = []
    th = threading.Thread(
        target=lambda: res.append(func(0.0, in_thread2=True)), daemon=True
    )
    th.start()

    lock1.acquire()  # Order: 2

    # Now |func| in the second thread is ongoing, but the interception should
    # be disabled for the main thread.
    self.assertEqual(jnp.sin(0.0), 0.0)
    self.assertEqual(jax.lax.sin(0.0), 0.0)

    # However, calling the same |func| in the main thread should be intercepted.
    self.assertEqual(func(0.0), 42.0)
    lock2.release()  # Order: 3
    th.join()  # Order: 5
    self.assertEqual(res[0], 42.0)

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
        func,
        lambda: interception.Interceptor(
            mapping={"jax.lax.sin": replaced_sin}, id=5
        ),
        disable_jit=True,
    )
    n = 42
    self.assertEqual(func2(0.0), 42)

  def test_interception_of_pjit_function(self):
    alias_div = jnp.true_divide  # which is a PjitFunction

    def func(x, y):
      return alias_div(x, y)

    func = interception.wrap_func_intercepted(
        func,
        lambda: interception.Interceptor(
            mapping={"jax.numpy.true_divide": lambda x, y: x / y + 1}, id=6
        ),
        disable_jit=True,
    )
    self.assertEqual(func(0.0, 1.0), 1.0)

  def test_interception_of_pjit_function_disable_jit_false(self):
    # When disable_jit=False is used, setattr mapping occurs. Local aliases
    # are NOT tracked natively like they are under disable_jit=True since
    # _fun is not intercepted.
    alias_div = jnp.true_divide

    def func(x, y):
      return alias_div(x, y)

    func = interception.wrap_func_intercepted(
        func,
        lambda: interception.Interceptor(
            mapping={"jax.numpy.true_divide": lambda x, y: x / y + 1}, id=7
        ),
        disable_jit=False,
    )
    # The interception will fail to capture alias_div because getattr mapping
    # cannot see local function refs at JIT time.
    self.assertEqual(func(0.0, 1.0), 0.0)

    # However, standard lazy-evaluated calls will be intercepted natively.
    def func_standard(x, y):
      return jnp.true_divide(x, y)

    func_standard = interception.wrap_func_intercepted(
        func_standard,
        lambda: interception.Interceptor(
            mapping={"jax.numpy.true_divide": lambda x, y: x / y + 1}, id=8
        ),
        disable_jit=False,
    )
    self.assertEqual(func_standard(0.0, 1.0), 1.0)

  def test_double_interception(self):
    with self.assertRaises(ValueError):
      i = interception.Interceptor(
          mapping={
              "jax.nn.swish._fun.__code__": lambda x: 42,
              "jax.nn.silu._fun.__code__": lambda x: 43,
          },
          id=1,
      )
      interception.interception_manager.activate_interceptor(i)

  def test_intercept_class_method(self):
    def func(x):
      return jax.nn.relu(x)

    def custom_jvp_call(self, *args, **kwargs):
      return self(*args, **kwargs) + 1

    func = interception.wrap_func_intercepted(
        func,
        lambda: interception.Interceptor(
            mapping={"jax.custom_jvp.__call__": custom_jvp_call}, id=9
        ),
        disable_jit=True,
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

    interceptor = lambda: interception.Interceptor(
        mapping={"jax.numpy.sin": replaced_sin}, id=10
    )

    def func(x):
      x = jnp.asarray(x)
      _, y = jax.lax.scan(lambda carry, x: (carry, jnp.sin(x)), 0.0, x[None])
      return y.squeeze()

    func = interception.wrap_func_intercepted(
        func, interceptor, disable_jit=True
    )
    self.assertEqual(replaced_sin(0.0), 1.0)
    self.assertEqual(func(0.0), 1.0)
    out, grads = jax.value_and_grad(func)(0.0)
    self.assertEqual(out, 1.0)
    self.assertEqual(grads, 0.0)

  def test_has_attribute(self):
    self.assertTrue(interception.has_attribute("jax.numpy.sin"))
    self.assertFalse(interception.has_attribute("jax.xxx.sin"))
    self.assertFalse(interception.has_attribute("xxx.sin"))

  def test_multiple_interceptions(self):
    def replaced_sin(x):
      return jax.lax.sin(x) + 10

    def replaced_cos(x):
      return jax.lax.cos(x) + 100

    interceptor1 = lambda: interception.Interceptor(
        mapping={
            "jax.lax.sin": replaced_sin,
            "jax.lax.cos": replaced_cos,
        },
        id=11,
    )
    interceptor2 = lambda: interception.Interceptor(
        mapping={
            "jax.lax.sin": replaced_sin,
            "jax.lax.cos": replaced_cos,
        },
        id=11,
    )

    def func(x):
      return jax.lax.sin(x) + jax.lax.cos(x)

    func = interception.wrap_func_intercepted(
        func, interceptor1, disable_jit=True
    )
    func = interception.wrap_func_intercepted(
        func, interceptor2, disable_jit=True
    )
    # The interceptors are identical in value, so the tuple(sorted(...)) hash
    # deduplicates the second one, preventing runaway recursion.
    self.assertEqual(func(0.0), 111.0)

  def test_interception_manager_multiple_interceptions(self):
    i1 = interception.Interceptor(
        mapping={"jax.lax.sin": lambda x: jax.lax.sin(x) + 1}, id=1
    )
    interception.interception_manager.activate_interceptor(i1)
    self.assertEqual(jax.lax.sin(0.0), 1.0)
    # Installing the same interceptor again will raise an error.
    with self.assertRaises(ValueError):
      i1_dup = interception.Interceptor(
          mapping={"jax.lax.sin": lambda x: jax.lax.sin(x) + 2}, id=1
      )
      interception.interception_manager.activate_interceptor(i1_dup)
    self.assertEqual(jax.lax.sin(0.0), 1.0)
    # Installing a different interceptor should apply both. And the first
    # interceptor is applied first.
    i2 = interception.Interceptor(
        mapping={"jax.lax.sin": lambda x: jax.lax.sin(x) * 2 + 10}, id=2
    )
    interception.interception_manager.activate_interceptor(i2)
    self.assertEqual(jax.lax.sin(0.0), 11.0)
    interception.interception_manager.deactivate_interceptor(i2)
    self.assertEqual(jax.lax.sin(0.0), 1.0)
    interception.interception_manager.deactivate_interceptor(i1)
    self.assertEqual(jax.lax.sin(0.0), 0.0)

    # Reinstalling.
    i3 = interception.Interceptor(
        mapping={"jax.lax.sin": lambda x: jax.lax.sin(x) + 2}, id=1
    )
    interception.interception_manager.activate_interceptor(i3)
    self.assertEqual(jax.lax.sin(0.0), 2.0)
    interception.interception_manager.deactivate_interceptor(i3)

  def test_interception_manager_code_object(self):
    i_code = interception.Interceptor(
        mapping={"jax.lax.sin.__code__": lambda x: jax.lax.sin(x) + 1}, id=1
    )
    interception.interception_manager.activate_interceptor(i_code)
    self.assertEqual(jax.lax.sin(0.0), 1.0)
    interception.interception_manager.deactivate_interceptor(i_code)
    self.assertEqual(jax.lax.sin(0.0), 0.0)


if __name__ == "__main__":
  absltest.main()
