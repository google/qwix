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

from absl.testing import absltest
from absl.testing import parameterized
from jax import numpy as jnp
from qwix._src.core import einsum_info


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

  def test_broadcast_operands_basic(self):
    op1 = jnp.zeros((2, 1, 4))
    op2 = jnp.zeros((1, 3, 4))
    ops = einsum_info.broadcast_operands([op1, op2], ['abc', 'abc'])
    self.assertEqual(ops[0].shape, (2, 3, 4))
    self.assertEqual(ops[1].shape, (2, 3, 4))

  def test_broadcast_operands_tiling(self):
    # 'N' dim: 4 vs 8. Allowed in Qwix if 8 % 4 == 0.
    op1 = jnp.zeros((4,))
    op2 = jnp.zeros((8,))
    ops = einsum_info.broadcast_operands([op1, op2], ['a', 'a'])
    self.assertEqual(ops[0].shape, (8,))  # 4 * (2,)
    self.assertEqual(ops[1].shape, (8,))

  def test_broadcast_operands_error(self):
    op1 = jnp.zeros((4,))
    op2 = jnp.zeros((5,))
    with self.assertRaisesRegex(ValueError, 'Cannot broadcast'):
      einsum_info.broadcast_operands([op1, op2], ['a', 'a'])


if __name__ == '__main__':
  absltest.main()
