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


import unittest.mock
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import opt_einsum
from qwix._src.core import einsum_info
from qwix._src.core import qarray


class SymbolicDim:
  """Mock class for symbolic dimension."""

  def __init__(self, name):
    self.name = name

  def __int__(self):
    raise TypeError(
        f'Symbolic dimension {self.name} cannot be converted to int'
    )

  def __index__(self):
    raise TypeError(
        f'Symbolic dimension {self.name} cannot be converted to index'
    )

  def __str__(self):
    return self.name

  def __repr__(self):
    return self.name

  def __mul__(self, other):
    return SymbolicDim(f'{other}*{self.name}')

  def __rmul__(self, other):
    return SymbolicDim(f'{other}*{self.name}')

  def __eq__(self, other):
    return isinstance(other, SymbolicDim) and self.name == other.name

  def __hash__(self):
    return hash(self.name)

  def __gt__(self, other):
    raise TypeError('Symbolic dimension cannot be compared')


class EinsumInfoTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='standard',
          einsum_str='abc,bcd->acd',
          expected_lhs='abc',
          expected_rhs='bcd',
          expected_out='acd',
          expected_batch=['c'],
          expected_contract=['b'],
      ),
      dict(
          testcase_name='contract',
          einsum_str='ab,bc->ac',
          expected_lhs='ab',
          expected_rhs='bc',
          expected_out='ac',
          expected_batch=[],
          expected_contract=['b'],
      ),
      dict(
          testcase_name='mixed',
          einsum_str='bth,thk->btk',
          expected_lhs='bth',
          expected_rhs='thk',
          expected_out='btk',
          expected_batch=['t'],
          expected_contract=['h'],
      ),
      dict(
          testcase_name='scalar_lhs',
          einsum_str=',i->i',
          expected_lhs='',
          expected_rhs='i',
          expected_out='i',
          expected_batch=[],
          expected_contract=[],
      ),
      dict(
          testcase_name='ellipsis_lhs',
          einsum_str='...a,ab->...b',
          ndims=(2, 2),
          expected_lhs='da',
          expected_rhs='ab',
          expected_out='db',
          expected_batch=[],
          expected_contract=['a'],
      ),
      dict(
          testcase_name='ellipsis_rhs',
          einsum_str='ab,...b->...a',
          ndims=(2, 2),
          expected_lhs='ab',
          expected_rhs='db',
          expected_out='da',
          expected_batch=[],
          expected_contract=['b'],
      ),
      dict(
          testcase_name='ellipsis_batch_match',
          einsum_str='...a,...b->...',
          ndims=(2, 2),
          expected_lhs='da',
          expected_rhs='db',
          expected_out='d',
          expected_batch=['d'],
          expected_contract=[],
      ),
  )
  def test_parse(
      self,
      einsum_str,
      expected_lhs,
      expected_rhs,
      expected_out,
      expected_batch,
      expected_contract,
      ndims=None,
  ):
    info = einsum_info.EinsumInfo.parse(einsum_str, ndims=ndims)
    self.assertEqual(info.lhs, expected_lhs)
    self.assertEqual(info.rhs, expected_rhs)
    self.assertEqual(info.out, expected_out)
    self.assertEqual(info.batch_chars, expected_batch)
    self.assertEqual(info.contract_chars, expected_contract)

  @parameterized.named_parameters(
      dict(testcase_name='unary', einsum_str='ii->i'),
      dict(testcase_name='repeated_lhs', einsum_str='ii,j->ij'),
      dict(testcase_name='repeated_rhs', einsum_str='i,jj->ij'),
      dict(testcase_name='repeated_out', einsum_str='i,j->ii'),
      dict(testcase_name='trinary', einsum_str='i,j,k->ijk'),
      dict(testcase_name='implicit', einsum_str='ij,jk'),
      dict(testcase_name='non_alpha', einsum_str='i.j,jk->ik'),
  )
  def test_parse_validation(self, einsum_str):
    with self.assertRaises(NotImplementedError):
      einsum_info.EinsumInfo.parse(einsum_str)

  def test_dimension_numbers(self):
    info = einsum_info.EinsumInfo.parse('abc,bcd->acd')
    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = (
        info.dimension_numbers
    )
    # batch: c(2) in lhs; c(1) in rhs
    self.assertEqual(lhs_batch, (2,))
    self.assertEqual(rhs_batch, (1,))
    # contract: b(1) in lhs; b(0) in rhs
    self.assertEqual(lhs_contract, (1,))
    self.assertEqual(rhs_contract, (0,))

  def test_output_perm(self):
    # 'abc,bcd->acd'.
    # batch: c. remaining: a, d.
    # current_out_chars order: c, a, d
    # desired: a, c, d
    # indices in current: a->1, c->0, d->2 -> (1, 0, 2)
    info = einsum_info.EinsumInfo.parse('abc,bcd->acd')
    self.assertEqual(info.output_perm, (1, 0, 2))


class BroadcastOperandsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='basic',
          lhs_is_qarray=False,
          rhs_is_qarray=False,
          op1_shape=(2, 1, 4),
          op2_shape=(1, 3, 4),
          expected_shape=(2, 3, 4),
      ),
      dict(
          testcase_name='tiling',
          lhs_is_qarray=False,
          rhs_is_qarray=False,
          op1_shape=(4,),
          op2_shape=(8,),
          expected_shape=(8,),
      ),
      dict(
          testcase_name='single_qarray',
          lhs_is_qarray=True,
          rhs_is_qarray=False,
          op1_shape=(2, 1, 4),
          op2_shape=(1, 3, 4),
          expected_shape=(2, 3, 4),
      ),
      dict(
          testcase_name='two_qarrays',
          lhs_is_qarray=True,
          rhs_is_qarray=True,
          op1_shape=(2, 1, 4),
          op2_shape=(1, 3, 4),
          expected_shape=(2, 3, 4),
      ),
  )
  def test_broadcast_operands_concrete(
      self,
      op1_shape,
      op2_shape,
      expected_shape,
      lhs_is_qarray=False,
      rhs_is_qarray=False,
  ):
    def create_operand(shape, is_qarray):
      val = jnp.zeros(shape)
      if is_qarray:
        # Create a channelwise-like scale for the first dimension.
        scale_shape = (shape[0],) + (1,) * (len(shape) - 1)
        return qarray.QArray(val, jnp.zeros(scale_shape))
      return val

    op1 = create_operand(op1_shape, lhs_is_qarray)
    op2 = create_operand(op2_shape, rhs_is_qarray)

    subs = ['abc'[: len(op1_shape)], 'abc'[: len(op2_shape)]]
    ops = einsum_info.broadcast_operands([op1, op2], subs)

    self.assertEqual(ops[0].shape, expected_shape)
    self.assertEqual(ops[1].shape, expected_shape)

    if lhs_is_qarray:
      self.assertEqual(ops[0].qvalue.shape, expected_shape)
      self.assertEqual(ops[0].scale.shape, expected_shape)

    if rhs_is_qarray:
      self.assertEqual(ops[1].qvalue.shape, expected_shape)
      self.assertEqual(ops[1].scale.shape, expected_shape)

  def test_broadcast_operands_error(self):
    op1 = jnp.zeros((4,))
    op2 = jnp.zeros((5,))
    with self.assertRaisesRegex(ValueError, 'Cannot broadcast'):
      einsum_info.broadcast_operands([op1, op2], ['a', 'a'])

  @parameterized.named_parameters(
      dict(
          testcase_name='qarray_with_symbolic',
          op1_info=(('N', 1, 4), True, 'abc'),
          op2_info=((1, 3, 4), False, 'abc'),
          expected_shape=('N', 3, 4),
          updates=[(0, ('N', 3, 4)), (1, ('N', 3, 4))],
      ),
      dict(
          testcase_name='one_then_symbolic',
          op1_info=((1,), False, 'a'),
          op2_info=(('N',), False, 'a'),
          expected_shape=('N',),
          updates=[(0, ('N',))],
      ),
      dict(
          testcase_name='symbolic_then_one',
          op1_info=(('N',), False, 'a'),
          op2_info=((1,), False, 'a'),
          expected_shape=('N',),
          updates=[(1, ('N',))],
      ),
      dict(
          testcase_name='symbolic_mismatch',
          op1_info=(('N',), False, 'a'),
          op2_info=(('M',), False, 'a'),
          expected_shape=('N',),
          updates=[(1, ('N',))],
      ),
  )
  def test_broadcast_operands_symbolic_parameterized(
      self, op1_info, op2_info, expected_shape, updates
  ):
    n_dim = SymbolicDim('N')
    m_dim = SymbolicDim('M')
    dim_map = {'N': n_dim, 'M': m_dim}

    def create_op(info):
      shape_raw, is_qarray, _ = info
      shape = tuple(dim_map.get(s, s) for s in shape_raw)
      if is_qarray:
        return qarray.QArray(
            jax.ShapeDtypeStruct(shape=shape, dtype=jnp.float32),
            jax.ShapeDtypeStruct(shape=shape, dtype=jnp.float32),
        )
      return jax.ShapeDtypeStruct(shape=shape, dtype=jnp.float32)

    op1 = create_op(op1_info)
    op2 = create_op(op2_info)
    operands = [op1, op2]
    subs = [op1_info[2], op2_info[2]]

    expected_shape_concrete = tuple(dim_map.get(s, s) for s in expected_shape)

    with unittest.mock.patch.object(qarray, 'broadcast_to') as mock_broadcast:
      mock_broadcast.side_effect = lambda op, shape: jax.ShapeDtypeStruct(
          shape=shape, dtype=jnp.float32
      )

      ops = einsum_info.broadcast_operands(operands, subs)

      self.assertEqual(ops[0].shape, expected_shape_concrete)
      self.assertEqual(ops[1].shape, expected_shape_concrete)

      for op_idx, target_shape_raw in updates:
        target_shape = tuple(dim_map.get(s, s) for s in target_shape_raw)
        mock_broadcast.assert_any_call(operands[op_idx], target_shape)


class SymbolicEinsumTest(absltest.TestCase):

  def test_einsum_contract_path_symbolic(self):
    n_dim = SymbolicDim('N')
    operands = [
        jax.ShapeDtypeStruct(shape=(n_dim, 10), dtype=jnp.float32),
        jax.ShapeDtypeStruct(shape=(10, n_dim), dtype=jnp.float32),
    ]
    input_subs = 'ij,jk'
    output_subs = 'ik'

    mock_shapes = [einsum_info.sanitize_shape(op.shape) for op in operands]

    path, contractions = opt_einsum.contract_path(
        f'{input_subs}->{output_subs}',
        *mock_shapes,
        shapes=True,
        einsum_call=True,
    )
    self.assertIsNotNone(path)
    self.assertNotEmpty(contractions)

  def test_sanitize_shape(self):
    n_dim = SymbolicDim('N')
    shape = (64, n_dim, 64 * n_dim)
    sanitized = einsum_info.sanitize_shape(shape)
    self.assertEqual(sanitized, (64, 1, 1))


if __name__ == '__main__':
  absltest.main()
