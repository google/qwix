# Copyright 2025 Google LLC
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
from absl.testing import absltest

import jax
from jax import numpy as jnp

from qwix._src import qconfig
from qwix._src.providers import ptq
from qwix._src.providers import ptq_pad
from qwix._src.core import einsum as core_einsum
from qwix._src.core import dot_general as core_dot
from qwix._src import model as qwix_model
from flax import linen as nn

D = 2880
tile_size = 256
D_padded = ((D + tile_size - 1) // tile_size) * tile_size
T, E, F = 16, 32, 5760

class PtqPadEquivalenceTest(absltest.TestCase):

  def test_einsum_simple(self):
        x = jax.random.normal(jax.random.key(0), (T, D), dtype=jnp.float32)
        w = jax.random.normal(jax.random.key(1), (E, D, F), dtype=jnp.float32)

        # Pad the weight for PTQ
        w_padded = jnp.pad(w, ((0, 0), (0, D_padded - D), (0, 0)))
        x_padded = jnp.pad(x, ((0, 0), (0, D_padded - D)))

        rule = qconfig.QuantizationRule(
            module_path='.*',
            weight_qtype='float8_e4m3fn',
            act_qtype="float8_e4m3fn",
            tile_size=tile_size,
        )

        how = core_einsum.get_how_to_quantize(
            einsum_str='td,edf->tef',
            ndims=(2, 3),
            for_lhs=False,
            qtype=rule.weight_qtype,
            tile_size=tile_size,
            calibration_method='absmax',
        )

        w_qarray_pad = ptq_pad.quantize_act(w, how, rule, None)
        w_qarray_ptq = ptq.quantize_act(w_padded, how, rule, None)

        q_diff = jnp.max(jnp.abs(w_qarray_pad.qvalue[:, :D, :] - w_qarray_ptq.qvalue[:, :D, :]))
        print(f'Max diff in calibration: {float(q_diff)}')
        scale_diff = jnp.max(jnp.abs(w_qarray_pad.scale - w_qarray_ptq.scale))
        print(f'Max diff in scales: {float(scale_diff)}')
        assert jnp.allclose(w_qarray_pad.qvalue[:, :D, :], w_qarray_ptq.qvalue[:, :D, :], atol=1e-6)
        assert jnp.allclose(w_qarray_pad.scale, w_qarray_ptq.scale, atol=1e-6)

        # Use tiny dummy modules so provider rule lookup has a module context.
        class EinsumModel(nn.Module):
            e: int
            d: int
            f: int
            w_init: jax.Array
            @nn.compact
            def __call__(self, x):
                w = self.param('w', lambda *args, **kwargs: self.w_init, (self.e, self.d, self.f))
                return jnp.einsum('td,edf->tef', x, w)

        pad_model = EinsumModel(E, D, F, w)
        pad_qmodel = qwix_model.quantize_model(pad_model, ptq_pad.PtqPadProvider([rule]))
        pad_vars = pad_qmodel.init(jax.random.key(0), x)
        result_pad = pad_qmodel.apply(pad_vars, x)

        ptq_model = EinsumModel(E, D_padded, F, w_padded)
        base_qmodel = qwix_model.quantize_model(ptq_model, ptq.PtqProvider([rule]))
        base_vars = base_qmodel.init(jax.random.key(1), x_padded)
        result_ptq = base_qmodel.apply(base_vars, x_padded)

        result_diff = jnp.max(jnp.abs(result_pad - result_ptq))
        print(f'Max diff in einsum results: {float(result_diff)}')

        assert jnp.allclose(result_pad, result_ptq, atol=1e-3)

  def test_dot_general_simple(self):
        x = jax.random.normal(jax.random.key(0), (T, D), dtype=jnp.float32)
        w = jax.random.normal(jax.random.key(1), (E, D, F), dtype=jnp.float32)

        w_padded = jnp.pad(w, ((0, 0), (0, D_padded - D), (0, 0)))
        x_padded = jnp.pad(x, ((0, 0), (0, D_padded - D)))

        rule = qconfig.QuantizationRule(
            module_path='.*',
            weight_qtype='float8_e4m3fn',
            act_qtype="float8_e4m3fn",
            tile_size=tile_size,
        )

        dimension_numbers = (([1], [1]), ([], []))
        how = core_dot.get_how_to_quantize(
            dimension_numbers=dimension_numbers,
            ndims=(2, 3),
            for_lhs=False,
            qtype=rule.weight_qtype,
            tile_size=tile_size,
            calibration_method='absmax',
        )

        w_qarray_pad = ptq_pad.quantize_act(w, how, rule, None)
        w_qarray_ptq = ptq.quantize_act(w_padded, how, rule, None)

        q_diff = jnp.max(jnp.abs(w_qarray_pad.qvalue[:, :D, :] - w_qarray_ptq.qvalue[:, :D, :]))
        print(f'Max diff in calibration: {float(q_diff)}')
        scale_diff = jnp.max(jnp.abs(w_qarray_pad.scale - w_qarray_ptq.scale))
        print(f'Max diff in scales: {float(scale_diff)}')
        assert jnp.allclose(w_qarray_pad.qvalue[:, :D, :], w_qarray_ptq.qvalue[:, :D, :], atol=1e-6)
        assert jnp.allclose(w_qarray_pad.scale, w_qarray_ptq.scale, atol=1e-6)

        class DotModel(nn.Module):
            e: int
            d: int
            f: int
            w_init: jax.Array
            @nn.compact
            def __call__(self, x):
                w = self.param('w', lambda *args, **kwargs: self.w_init, (self.e, self.d, self.f))
                return jax.lax.dot_general(x, w, dimension_numbers)

        pad_model = DotModel(E, D, F, w)
        pad_qmodel = qwix_model.quantize_model(pad_model, ptq_pad.PtqPadProvider([rule]))
        pad_vars = pad_qmodel.init(jax.random.key(0), x)
        result_pad = pad_qmodel.apply(pad_vars, x)

        ptq_model = DotModel(E, D_padded, F, w_padded)
        base_qmodel = qwix_model.quantize_model(ptq_model, ptq.PtqProvider([rule]))
        base_vars = base_qmodel.init(jax.random.key(1), x_padded)
        result_ptq = base_qmodel.apply(base_vars, x_padded)

        result_diff = jnp.max(jnp.abs(result_pad - result_ptq))
        print(f'Max diff in dot_general results: {float(result_diff)}')
        assert jnp.allclose(result_pad, result_ptq, atol=1e-3)


if __name__ == '__main__':
  absltest.main()
