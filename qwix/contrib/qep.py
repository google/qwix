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

"""Integration of QEP (Quantization Error Propagation) into Qwix.

QEP extends GPTQ by accounting for quantization noise in input activations seen
by later layers.

The algorithm operates stagewise through the layers of a model: for each stage,
it collects paired float vs progressively quantized inputs across the full
calibration set, quantizes that stage, updates the quantized model state, and
then continues to the next stage.

This implementation currently supports Flax linen models only.
"""

from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
import dataclasses
import itertools
from typing import Any, cast

import flax
from flax import linen as nn
import jax
from qwix._src import averaging
from qwix._src import model as qwix_model
from qwix._src import qconfig
from qwix._src.core import qarray
from qwix._src.providers import ptq
from qwix.contrib import calibration
from qwix.contrib import gptq
from qwix.contrib import gptq_core
from qwix.contrib import qep_core


@dataclasses.dataclass(frozen=True, kw_only=True)
class QepRule(gptq.GptqRule):
  """Use this rule to enable QEP (input-compensated GPTQ).

  QEP extends GPTQ by accounting for quantization noise in input activations
  from previous layers. While standard GPTQ quantizes each layer independently
  assuming perfect (float) inputs, QEP measures the actual accumulated
  quantization error from preceding layers and shifts the current layer's
  weights to compensate for that noise.

  This rule tells the quantization pipeline to treat matched weights as part
  of a QEP stage. It configures the hyperparameters for the QEP correction
  step, which computes a weight delta using the difference between float
  and quantized input activations.

  Attributes:
    correction_factor: Weight correction factor. 0.0 = no correction (equivalent
      to standard GPTQ), 1.0 = full correction. Default is 0.5 per the QEP paper
      recommendations, balancing noise compensation against excessive weight
      drift.
    damping_factor: Damping factor for QEP weight correction Hessian inversion.
      Default 0.01. This stabilizes the inverse Hessian computation.
    apply_correction: Whether to apply QEP weight correction before GPTQ. Set
      this to ``False`` for stages that should run GPTQ without the QEP
      correction term (e.g., the very first layer which has no preceding noise).
  """

  correction_factor: float = 0.5
  damping_factor: float = 0.01
  apply_correction: bool = True


@dataclasses.dataclass(frozen=True)
class QepStage:
  """Metadata about one QEP stage.

  This class represents a single algorithmic step (a "stage") constructed during
  the topological discovery pass. Unlike standard GPTQ which looks at each
  parameter independently, QEP groups interconnected operations (e.g., parallel
  attention heads) into a single stage. This is because these operations share
  the same input activations, and thus experience the same cascading
  quantization noise from previous layers.

  Example:
    If a model has two dense layers operating on the same input:

    >>> stage = QepStage(
    ...     index=0,
    ...     param_paths=(('Block_0', 'Dense_0', 'kernel'),
    ...                  ('Block_0', 'Dense_1', 'kernel')),
    ...     module_paths=('Block_0/Dense_0', 'Block_0/Dense_1')
    ... )
    >>> print(f"Stage {stage.index} processes {stage.module_paths}")
    Stage 0 processes ('Block_0/Dense_0', 'Block_0/Dense_1')

  Attributes:
    index: The chronological sequence index of this stage in the quantization
      process (0-indexed).
    param_paths: A tuple of tuple paths, each identifying the exact variable
      being quantized in this stage.
    module_paths: A tuple of strings, each showing the hierarchical submodule
      path of the quantized operations.
  """

  index: int
  param_paths: tuple[tuple[str, ...], ...]
  module_paths: tuple[str, ...]


@dataclasses.dataclass(frozen=True, kw_only=True)
class QepResult:
  """Stagewise results from a QEP run.

  This class encapsulates the final artifacts returned after the QEP
  quantization process successfully finishes quantizing all identified stages.

  Example:
    After running the QEP algorithm block, you will receive a result object
    containing the updated model components ready for inference.

    >>> result = run_qep(model, params, calibration_dataset)
    >>> print(f"Quantized {len(result.stages)} stages.")
    >>>
    >>> # The result contains the updated model components ready for inference
    >>> inference_output = result.model.apply(
    ...     {'params': result.params, 'quant_stats': result.quant_stats},
    ...     sample_input
    ... )

  Attributes:
    model: The modified `flax.linen.Module` architecture bound to PTQ logic.
    params: The deeply nested tree containing quantized parameters (e.g.
      `qarray` objects).
    quant_stats: The nested dictionary encompassing floating statistics, scales,
      and quantization error metadata (including variables tracked via `_qep`
      suffixes).
    stages: A chronological tuple of `QepStage` metadata objects documenting
      which parameters were grouped and quantized at each step.
  """

  model: nn.Module
  params: Any
  quant_stats: Any
  stages: tuple[QepStage, ...]


@dataclasses.dataclass(frozen=True)
class _MatchedOp:
  """One supported op matched during discovery.

  This is an internal tracking structure generated during the discovery pass. It
  isolates individual occurrences of rule-conforming operations (like
  `dot_general`) and binds their logical path, operator definitions, and
  underlying python memory ID (`lhs_id`) of the activations.

  Examining the `lhs_id` allows the orchestrator to identify operations that
  consume the identically shared activation array memory, grouping them
  together.

  Example:
    >>> op = _MatchedOp(
    ...     op_key=('dot_general', ('Block_0', 'Dense_0'), 'dot_id_42'),
    ...     path=('Block_0', 'Dense_0', 'kernel'),
    ...     lhs_id=1405234234123,
    ...     rule=QepRule(correction_factor=0.5)
    ... )

  Attributes:
    op_key: A unique tuple consisting of the operation name, the module path,
      and an optional internal ID (e.g. `('dot_general', ('Dense_0',), 'id1')`)
      to map telemetry captures.
    path: The exact path targeting the parameter variable (e.g. `('Dense_0',
      'kernel')`).
    lhs_id: The Python runtime `id()` belonging to the LHS input structure
      array. If multiple ops share this ID, they are bundled together.
    rule: The specific `QepRule` object directing the mathematical execution
      branch for this operator.
  """

  op_key: tuple[Any, ...]
  path: tuple[str, ...]
  lhs_id: int
  rule: QepRule


@dataclasses.dataclass(frozen=True)
class _StageSpec:
  """Internal stage specification.

  This class acts as the functional counterpart to the public `QepStage`
  metadata.
  It bundles localized matching operator artifacts (`_MatchedOp` records) that
  must be quantized concurrently. The QEP pipeline consumes this specification
  to run batch extraction and localized quantization logic.

  Example:
    >>> # Assuming 2 ops share the same lhs_id
    >>> spec = _StageSpec(
    ...     index=0,
    ...     members=(op1, op2)
    ... )
    >>> # The orchestrator runs quantization operations targeting just these
    members
    >>> qep_runner.run_stage(spec, gptq_block_size=128,
    gptq_damping_factor=0.01)

  Attributes:
    index: The chronological sequence index (0-indexed) defining the evaluation
      order.
    members: A tuple of `_MatchedOp` objects. Every operator listed here shares
      identical input features, meaning they are bound physically and updated
      simultaneously during this stage.
  """

  index: int
  members: tuple[_MatchedOp, ...]


def _unfreeze_params_tree(tree: Any) -> Any:
  """Unfreezes a parameter tree if it is a flax FrozenDict.

  Linen frequently returns model parameters wrapped in an immutable
  `flax.core.FrozenDict`. Weight quantization acts as an in-place
  transformation, modifying layers incrementally. This helper provides a unified
  way to operate on the potentially frozen variable dictionary by unfreezing it
  into a standard Python mutable dictionary when necessary.

  Args:
    tree: The parameter tree to convert, potentially a `FrozenDict`.

  Returns:
    A mutable nested dictionary representation of the parameter tree.
  """
  if isinstance(tree, flax.core.FrozenDict):
    return flax.core.unfreeze(tree)
  return tree


def _flatten_params_tree_to_tuple_paths(
    tree: Any,
) -> dict[tuple[str, ...], Any]:
  """Flattens a nested parameters tree into a single-level dictionary.

  Allows treating structured parameters nested within multiple module levels
  (e.g., `{'Dense_0': {'kernel': array}}`) as independent units indexed by
  a tuple path (e.g., `('Dense_0', 'kernel')`). This format is essential for
  incremental updates during individual QEP stages.

  Args:
    tree: A deeply nested parameter tree (e.g., variables['params']).

  Returns:
    A flat dictionary mapping tuple paths to their corresponding leaf values.
  """
  return flax.traverse_util.flatten_dict(_unfreeze_params_tree(tree))


def _unflatten_tuple_paths_to_params_tree(
    flat_tree: dict[tuple[str, ...], Any],
) -> Any:
  """Reconstructs a deeply nested parameters tree from a flat dictionary.

  This reverses the transformation enacted by
  `_flatten_params_tree_to_tuple_paths`, assembling dictionary branches from the
  keys of the tuple path leaves.

  Args:
    flat_tree: A flat dictionary of parameters keyed by tuple paths.

  Returns:
    The unflattened nested parameter tree, compatible with flax linen apply.
  """
  return flax.traverse_util.unflatten_dict(flat_tree)


def _append_qep_suffix_to_path(path: tuple[str, ...]) -> tuple[str, ...]:
  """Appends the `_qep` suffix to the leaf weight name in a tuple path.

  Stats arrays for QEP are distinguished from base float weights or generic
  GPTQ stats by this specialized suffix. For example, a tuple path targeting
  `('Dense_0', 'kernel')` becomes `('Dense_0', 'kernel_qep')`.

  Args:
    path: The original parameter tuple path targeting the weight variable.

  Returns:
    A new tuple path modified to target the QEP stats leaf counterpart.
  """
  return (*path[:-1], path[-1] + '_qep')


def _update_flat_stats_with_moving_average(
    flat_stats: dict[tuple[str, ...], Any],
    path: tuple[str, ...],
    stats: dict[str, jax.Array],
) -> None:
  """Aggregates batch-level structural statistics via a moving average.

  The core `compute_qep_stats` algorithm provides localized computations from
  only the *current* batch replay. QEP requires estimating noise distribution
  spanning the *entire* calibration dataset before making permanent weight
  updates.

  This helper continually updates the cumulative average stored in `flat_stats`
  using flax's `averaging.SimpleMovingAverage`.

  Args:
    flat_stats: Flat dictionary persisting the accumulated QEP batch stats over
      time, indexed by the target stats path. Mutated in-place.
    path: The weight tuple path acting as the target key (the suffix will be
      added internally).
    stats: The localized batch-level statistical arrays returned direct from
      `compute_qep_stats`.
  """
  aggregator = averaging.SimpleMovingAverage()
  stat_path = _append_qep_suffix_to_path(path)
  quant_stat = flat_stats.get(stat_path)
  if quant_stat is None:
    quant_stat = aggregator.init(stats)
  quant_stat = aggregator.update(quant_stat, stats)
  flat_stats[stat_path] = quant_stat


class _CaptureProvider(calibration.CalibrationProvider):
  """Provider that records matched operations and captures activations.

  Unlike standard GPTQ/AWQ calibration providers, this specialized provider does
  not proactively inject arrays into the model's `quant_stats` graph state.
  Instead, it acts as a telemetry listener across the linen forward pass,
  exposing a Python API used by the QEP orchestration loop.

  It has two distinct modes of operation depending on the exact stage:
  1. Discovery Pass: Enumerates supported matched ops in topological
    forward-pass order, recording their module paths, object references, and
    assigned ops targets.
  2. Capture Pass: Records and persists the real, normalized LHS activations
    directly into python memory for pre-selected subsets of ops.

  The parent `CalibrationProvider` continues to perform rule matching, shape
  validation, parameter resolution, and initial activation axis slicing behind
  the scenes.
  """

  def __init__(self, rules: Sequence[qconfig.QuantizationRule]):
    """Initializes the provider with the matched rules.

    Args:
      rules: A sequence of quantization rules. Usually instances of `QepRule`.
    """
    super().__init__(rules)
    self._discovered_ops: list[_MatchedOp] = []
    self._capture_keys: set[tuple[Any, ...]] | None = None
    self._captures: dict[tuple[Any, ...], jax.Array] = {}

  def get_rule_type(self) -> type[qconfig.QuantizationRule]:
    """Restricts activation capture strictly to operations matching `QepRule`.

    Returns:
      The expected `QepRule` class type.
    """
    return QepRule

  def get_stats_suffix(self) -> str:
    """Returns the dedicated suffix tracking the QEP calibration artifacts.

    Returns:
      The string suffix `'_qep'`.
    """
    return '_qep'

  def prepare_for_discovery(self) -> None:
    """Readies the telemetry listener for a new topological discovery pass.

    This resets internal collections, forgetting all currently recorded matched
    ops, capture settings, and previous telemetry payload sets in preparation
    for sweeping the model from input blocks to output blocks to establish stage
    dependencies.
    """
    self._discovered_ops.clear()
    self._capture_keys = None
    self._captures = {}

  def prepare_for_capture(
      self, op_keys: Collection[tuple[Any, ...]]
  ) -> dict[tuple[Any, ...], jax.Array]:
    """Prepares the provider to intercept activations for the selected ops.

    By passing the operator identifiers tracked during the discovery pass, you
    tell the provider to skip the broader indexing process and actively save
    tensor slices out to the returned `captures` dictionary on the next
    `model.apply()`.

    Args:
      op_keys: A collection of operation keys (derived from discovery data) that
        are slated for LHS activation extraction during the subsequent run.

    Returns:
      A reference to the active dictionary where captured activations will be
      populated asynchronously during the forward execution pass.
    """
    self._capture_keys = set(op_keys)
    self._captures = {}
    return self._captures

  @property
  def discovered_ops(self) -> tuple[_MatchedOp, ...]:
    """Returns chronologically traced QEP ops mapping exact stage assignments.

    Returns:
      An immutable sequence of mapped operations detailing rule matches and LHS
      objects, preserved exactly in the topological order they were evaluated.
    """
    return tuple(self._discovered_ops)

  def _collect_stats(
      self,
      lhs: jax.Array,
      weight_name: str,
      *,
      module_path: tuple[str, ...],
      op_name: str,
      op_id: str | None,
      lhs_id: int,
  ) -> None:
    """Intervenes in `CalibrationProvider` callbacks to persist state traces.

    The parent `CalibrationProvider` triggers this callback every time it
    encounters a rule-abiding dot/einsum operator structure holding a validated
    parameter tensor.

    During a Discovery run (if `prepare_for_capture` hasn't armed keys yet),
    this method establishes the foundational mapping linking the physical Python
    variable (the LHS tensor) to module names, topological ordering indices, and
    applied rules.

    During a Capture run, it ignores new topological data and strictly dumps the
    normalized sequence representation of the LHS tensor into memory mapped by
    the operation's key.

    Args:
      lhs: The rearranged LHS activation.
      weight_name: The leaf key belonging to the weight array (e.g.,
        `'kernel'`).
      module_path: The traversed hierarchical path (e.g., `('Block_0',
        'Dense_1')`).
      op_name: The specific function op alias intercepted (e.g.,
        `'dot_general'`).
      op_id: Internal serialization ID assigned internally by the trace
        mechanics.
      lhs_id: Raw python memory tracker `id(lhs)` to link shared references
        traversing independent graph layers.
    """
    path = (*module_path, weight_name)
    op_key = (op_name, module_path, op_id)

    if self._capture_keys is None:
      rule, _ = self._get_current_rule_and_op_id(op_name)
      assert isinstance(rule, QepRule)
      self._discovered_ops.append(
          _MatchedOp(
              op_key=op_key,
              path=path,
              lhs_id=lhs_id,
              rule=rule,
          )
      )
    elif op_key in self._capture_keys:
      self._captures[op_key] = lhs


def _group_discovered_ops_into_stages(
    discovered_ops: tuple[_MatchedOp, ...],
) -> tuple[_StageSpec, ...]:
  """Constructs grouped sequential quantization stages from discovered layers.

  Quantization stages are determined by runtime input locality. By evaluating
  object identities (LHS ID tracking), consecutive interconnected operator
  layers that tap the identical physical activation tensor are pooled into a
  single processing unit. This effectively binds parallel attention heads or
  sibling feed-forward blocks so their errors are co-adjusted comprehensively
  rather than disrupting one another.

  Args:
    discovered_ops: Chronologically ordered tuple of identified QEP operator
      nodes retrieved from the telemetry discovery pass.

  Returns:
    A sequential tuple mapping grouped batches of connected operators into
    discrete
    mathematical `_StageSpec` computation units.

  Raises:
    ValueError: If no valid ops were discovered matching the active rules.
    ValueError: If divergent operator bindings attempt to mutate the exact same
      variable subspace.
  """
  if not discovered_ops:
    raise ValueError(
        'No supported QEP ops were discovered. Ensure the rules match '
        'supported dot_general or einsum weight ops.'
    )

  stages = []
  for index, (_, members) in enumerate(
      itertools.groupby(discovered_ops, key=lambda op: op.lhs_id)
  ):
    stages.append(_StageSpec(index=index, members=tuple(members)))

  seen_paths: set[tuple[str, ...]] = set()
  for stage in stages:
    stage_paths = {op.path for op in stage.members}
    if seen_paths & stage_paths:
      raise ValueError(
          'QEP does not support quantizing the same param path across multiple '
          'stages.'
      )
    seen_paths.update(stage_paths)
  return tuple(stages)


def _convert_internal_stage_to_public_metadata(stage: _StageSpec) -> QepStage:
  """Translates internal stage structures into user-facing metadata.

  Condenses heavily linked object trackers into clean primitive path strings
  appropriate for logging debugging details out to terminal streams.

  Args:
    stage: An inter-linked active stage unit used by the runner pipeline.

  Returns:
    A serializable, human-readable metadata record detailing the paths bundled
    during this stage pass.
  """
  unique_paths = tuple(dict.fromkeys(op.path for op in stage.members))
  unique_module_paths = tuple(
      dict.fromkeys('/'.join(op.path[:-1]) for op in stage.members)
  )
  return QepStage(
      index=stage.index,
      param_paths=unique_paths,
      module_paths=unique_module_paths,
  )


def _run_model_forward_with_injected_params(
    model: nn.Module,
    variables: Any,
    params: Any,
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
) -> Any:
  """Executes the flax linen graph, replacing the `params` branch dynamically.

  In order to sample the true quantization discrepancy cascading from layers
  near the origin of the network down to the later end, we need to spin up
  identical networks operating against newly "dequantized" temporary weights.
  This helper injects replacement weights precisely before executing the graph
  application.

  Args:
    model: The immutable linen architecture definition struct.
    variables: The base frozen variables spanning the network graph (e.g. batch
      stat nodes).
    params: The mutated, targeted parameter branch replacing the original float
      branch.
    args: Extrapolated positional arguments fed into the network input tensor
      structure.
    kwargs: Keyword-tied supplementary pipeline inputs for the model step.

  Returns:
    The direct execution product rendered from traversing the active model.
  """
  apply_variables = {**variables, 'params': params}
  return model.apply(apply_variables, *args, **kwargs)


def _create_ptq_model_and_abstract_quantized_params(
    model: nn.Module,
    rules: Sequence[qconfig.QuantizationRule],
    methods: Collection[str],
    sample_args: Sequence[Any],
    sample_kwargs: Mapping[str, Any],
    abstract_quantized: Any,
) -> tuple[nn.Module, Any]:
  """Creates the PTQ model and abstract quantized parameters.

  It duplicates the fundamental base graph structure into a discrete mapping
  capable of parsing integer bindings. Next, it establishes the PTQ baseline
  bounds by evaluating JAX shape primitives across an idle graph run, carving
  out the exact array definitions expected of the final outputs without
  materializing the physical arrays into memory.

  Args:
    model: The clean float reference mapping of the targeted graph.
    rules: Active sequence arrays describing binding patterns mapping to
      quantization rules.
    methods: Registered target sub-functions exposed on the model.
    sample_args: Representative arrays simulating proper dimensionality
      bindings.
    sample_kwargs: Additional parameters directing conditional sub-graph
      branches.
    abstract_quantized: Overriding manually initialized PTQ definitions, if
      supplied.

  Returns:
    A tuple structure `(ptq_model, abstract_quantized)` where:
    - `ptq_model` is the wrapped module accepting quantized bounds arrays.
    - `abstract_quantized` is the deeply nested shape-aware definition tree
    matching
      layer-by-layer scale attributes.
  """
  ptq_model = qwix_model.quantize_model(
      model, ptq.PtqProvider(rules), methods=methods
  )
  if abstract_quantized is None:
    abstract_quantized = jax.eval_shape(
        ptq_model.init,
        jax.random.key(0),
        *sample_args,
        **sample_kwargs,
    )['params']
  return ptq_model, abstract_quantized


def _quantize_weight(
    ctx: calibration.CalibratedQuantContext,
    rule: QepRule,
    gptq_block_size: int,
    gptq_damping_factor: float,
) -> ptq.WithAux:
  """Generates a compressed discrete weight, adapting the QEP formula.

  Extracts the raw floating weight, adjusts it using the QEP compensation metric
  arrays (Hessian + specific structural error deltas tracking previous layer
  quantization), and immediately hands off the adjusted float tensor into the
  core GPTQ compression algorithm to produce an isolated, stable tensor format
  payload bounded neatly into blocks.

  Args:
    ctx: Parameter structure array populated actively with corresponding
      calibration traits.
    rule: Targeted execution instructions modifying block corrections
      dynamically.
    gptq_block_size: Block sizing granularity guiding discrete chunks across
      matrix vectors.
    gptq_damping_factor: Numeric scale controlling error bound stability during
      division paths.

  Returns:
    A newly integrated structural payload referencing the replaced discrete
    numerical definitions wrapping natively around the initial tensor placement
    array bounds.

  Raises:
    ValueError: If a correction logic flow was mandated without generating
    corresponding delta arrays across preliminary QEP stats.
  """
  hessian = ctx.calibration_stats['hessian']
  assert (
      hessian.shape[0] == ctx.weight.shape[1]
      and hessian.shape[1] == ctx.weight.shape[1]
  )

  weight = ctx.weight
  if rule.apply_correction:
    hessian_delta = ctx.calibration_stats.get('hessian_delta')
    if hessian_delta is None:
      raise ValueError(f'hessian_delta not found in QEP stats for {ctx.path}.')
    weight = qep_core.weight_correct(
        weight,
        hessian,
        hessian_delta,
        correction_factor=rule.correction_factor,
        damping_factor=rule.damping_factor,
    )

  weight = gptq_core.quantize_weight(
      weight,
      hessian,
      ctx.how,
      blocksize=gptq_block_size,
      percdamp=gptq_damping_factor,
  )[0]
  weight = ctx.restore_shape(weight)
  return ctx.abs_w.replace(array=weight)  # pyrefly: ignore[missing-attribute]


def quantize(
    model: nn.Module,
    calibration_data: Iterable[Any] | Callable[[], Iterable[Any]],
    rules: Sequence[QepRule],
    *,
    variables: Any = None,
    batch_adapter: (
        Callable[[Any], tuple[Sequence[Any], Mapping[str, Any]]] | None
    ) = None,
    methods: Collection[str] = ('__call__',),
    abstract_quantized: Any = None,
    allow_extra_params: bool = False,
    gptq_block_size: int = 128,
    gptq_damping_factor: float = 0.01,
) -> QepResult:
  """Executes Quantization Error Propagation (QEP) on a flax linen model.

  QEP extends iterative PTQ/GPTQ by observing the actual numerical divergence
  between the float and quantized model states. It sweeps through the
  calibration dataset continuously, adjusting the parameters of each localized
  "stage" by accounting both for the layer's internal structural Hessian and the
  shifted activations stemming from all preceding, previously quantized layers.

  The quantization proceeds as follows:
  1. Discover supported matched ops in forward order on one float pass.
  2. Infer stages by grouping consecutive matched ops that share the same input
     activation object.
  3. For each stage, replay the calibration set twice per batch:
     one float forward and one forward using dequantized weights from already
     quantized earlier stages.
  4. Accumulate `_qep` stats locally targeting the stage.
  5. Apply the QEP formula compensating against errors, and discrete the format
  with GPTQ.
  6. PTQ-quantize any remaining rule-matched weights.

  Args:
    model: The targeted `flax.linen.Module` architecture describing the float
      network.
    calibration_data: A reiterable collection representing the calibration
      dataset, or a zero-argument callable serving fresh iterators over the full
      dataset.
    rules: Sequence of definitions mapping layers to their quantization rules.
    variables: Frozen variables collection natively encompassing the mutating
      `params` array branch.
    batch_adapter: Utility restricting array parameters across arguments
      formatting sequentially.
    methods: Exposes subset routines of the internal models bounded
      sub-functions.
    abstract_quantized: Cached metadata representation bounding discrete
      structures.
    allow_extra_params: Permits external structural fields evading discrete
      arrays constraints.
    gptq_block_size: Resolution block constraints managing boundaries correctly
      mapping natively.
    gptq_damping_factor: Numeric scale lowering evaluation Hessian variances
      across numeric types.

  Returns:
    A structured tuple holding the target model, the resulting discretized array
    formats, and the
    statistics metadata variables bounding exact error structures statically.
  """
  if not isinstance(model, nn.Module):
    raise ValueError('qep.quantize currently supports linen models only.')
  if variables is None:
    raise ValueError('variables is required for linen QEP quantization.')

  # Adapt a raw calibration batch into model-compatible positional arguments.
  # The default adapter assumes that each yielded element from the calibration
  # data iterator is a stand-alone positional argument.
  if batch_adapter is None:
    batch_adapter = lambda batch: ((batch,), {})

  # QEP evaluates the neural network stagewise, so it must process the entire
  # calibration dataset multiple times. Thus, we ensure calibration_data can
  # repeatedly produce fresh iterators over the dataset without exhausting it.
  if callable(calibration_data):
    batch_iter_factory = calibration_data
  else:
    iterator = iter(calibration_data)
    if iterator is calibration_data:
      raise ValueError(
          'calibration_data must be reiterable, or a zero-arg callable that '
          'returns a fresh iterable.'
      )
    batch_iter_factory = lambda: iter(cast(Iterable[Any], calibration_data))
  first_iterator = iter(batch_iter_factory())

  try:
    first_batch = next(first_iterator)
  except StopIteration as exc:
    raise ValueError(
        'calibration_data must contain at least one batch.'
    ) from exc

  sample_args, sample_kwargs = batch_adapter(first_batch)
  float_params = _unfreeze_params_tree(variables['params'])

  float_provider = _CaptureProvider(rules)
  quant_provider = _CaptureProvider(rules)
  float_model = qwix_model.quantize_model(
      model, float_provider, methods=methods
  )
  quant_model = qwix_model.quantize_model(
      model, quant_provider, methods=methods
  )

  float_provider.prepare_for_discovery()
  _run_model_forward_with_injected_params(
      float_model, variables, float_params, sample_args, sample_kwargs
  )
  stages = _group_discovered_ops_into_stages(float_provider.discovered_ops)

  ptq_model, abstract_quantized = (
      _create_ptq_model_and_abstract_quantized_params(
          model,
          rules,
          methods,
          sample_args,
          sample_kwargs,
          abstract_quantized,
      )
  )

  flat_float_params = _flatten_params_tree_to_tuple_paths(float_params)
  flat_abstract_quantized = _flatten_params_tree_to_tuple_paths(
      abstract_quantized
  )

  current_dequantized_params_flat = dict(flat_float_params)
  final_quantized_params_flat: dict[tuple[str, ...], Any] = {}
  flat_quant_stats: dict[tuple[str, ...], Any] = {}
  staged_paths: set[tuple[str, ...]] = set()

  def replay_and_collect_stats(stage: _StageSpec) -> dict[tuple[str, ...], Any]:
    """Sweeps calibration data across dual paths to capture divergent artifacts.

    To pinpoint actual cascaded quantization discrepancies, this method loops
    completely across the entire raw calibration payload natively to a pure
    unquantized graph layout versus an increasingly modified/discretized graph
    structure containing previously processed parameter sets locally on the
    current stage run.

    This captures diverging states in array layouts, calculates the localized
    variance, and returns localized statistics capturing exactly how to
    mathematically shift future parameters.

    Args:
      stage: Core structural node definition identifying paths grouped into
        active stage bounds.

    Returns:
      Consolidated flat dictionary tracking cumulative statistics mapping
      directly targeting QEP logic branches.

    Raises:
      ValueError: Upon absent LHS arrays recorded during active capture cycles.
    """
    stage_op_keys = tuple(op.op_key for op in stage.members)
    stage_flat_stats: dict[tuple[str, ...], Any] = {}
    current_dequantized_params = _unflatten_tuple_paths_to_params_tree(
        current_dequantized_params_flat
    )

    for batch in batch_iter_factory():
      args, kwargs = batch_adapter(batch)

      float_captures = float_provider.prepare_for_capture(stage_op_keys)
      _run_model_forward_with_injected_params(
          float_model, variables, float_params, args, kwargs
      )

      quant_captures = quant_provider.prepare_for_capture(stage_op_keys)
      _run_model_forward_with_injected_params(
          quant_model,
          variables,
          current_dequantized_params,
          args,
          kwargs,
      )

      for op in stage.members:
        float_lhs = float_captures.get(op.op_key)
        quant_lhs = quant_captures.get(op.op_key)
        if float_lhs is None or quant_lhs is None:
          raise ValueError(
              f'Missing captured QEP activations for {"/".join(op.path[:-1])}.'
          )
        _update_flat_stats_with_moving_average(
            stage_flat_stats,
            op.path,
            qep_core.compute_qep_stats(quant_lhs, float_lhs),
        )
    return stage_flat_stats

  def apply_quantization(
      stage_rule_by_path: dict[tuple[str, ...], QepRule],
      stage_flat_stats: dict[tuple[str, ...], Any],
  ) -> None:
    """Rewrites discrete block parameters based on stage statistics.

    Deploys mathematically guided structural changes derived from error
    artifacts, finalizing the discrete structural weight layout using GPTQ
    compression directly substituting target blocks cleanly onto the final
    arrays while storing localized floating formats internally out for usage via
    remaining forward runs.

    Args:
      stage_rule_by_path: Nested targeted graph identifying active structural
        rules dictating operations.
      stage_flat_stats: Sliced numeric representations tracking cumulative stage
        artifacts mathematically.

    Raises:
      ValueError: If parameter constraints prevent successful translation within
        quantization rules.
    """
    for path, rule in stage_rule_by_path.items():
      w = flat_float_params[path]
      abs_w = flat_abstract_quantized[path]
      stats = stage_flat_stats[_append_qep_suffix_to_path(path)]
      ctx = calibration.extract_calibrated_quant_context(path, w, abs_w, stats)
      if ctx is None:
        raise ValueError(f'Failed to infer quantization parameters for {path}')

      final_quantized_params_flat[path] = _quantize_weight(
          ctx,
          rule,
          gptq_block_size=gptq_block_size,
          gptq_damping_factor=gptq_damping_factor,
      )
      current_dequantized_params_flat[path] = qarray.dequantize(
          final_quantized_params_flat[path].array
      )

  for stage in stages:
    stage_rule_by_path = {op.path: op.rule for op in stage.members}
    stage_flat_stats = replay_and_collect_stats(stage)
    apply_quantization(stage_rule_by_path, stage_flat_stats)
    flat_quant_stats.update(stage_flat_stats)
    staged_paths.update(stage_rule_by_path)

  remaining_flat_params = {
      path: value
      for path, value in flat_float_params.items()
      if path not in staged_paths
  }
  if remaining_flat_params:
    remaining_quantized = ptq.quantize_params(
        _unflatten_tuple_paths_to_params_tree(remaining_flat_params),
        abstract_quantized,
        allow_extra_params=allow_extra_params,
    )
    final_quantized_params_flat.update(
        _flatten_params_tree_to_tuple_paths(remaining_quantized)
    )

  return QepResult(
      model=ptq_model,
      params=_unflatten_tuple_paths_to_params_tree(final_quantized_params_flat),
      quant_stats=_unflatten_tuple_paths_to_params_tree(flat_quant_stats),
      stages=tuple(
          _convert_internal_stage_to_public_metadata(stage) for stage in stages
      ),
  )


def quantize_params(
    params: Any,
    abstract_quantized_params: Any,
    qep_quant_stats: Any,
    *,
    allow_extra_params: bool = False,
    gptq_block_size: int = 128,
    gptq_damping_factor: float = 0.01,
    correction_factor: float = 0.5,
    damping_factor: float = 0.01,
    apply_correction: bool = True,
) -> Any:
  """Quantizes parameters from precomputed QEP statistical metrics offline.

  This helper specifically ingests fully materialized `_qep` calibration
  distributions applying discrete adjustments isolated entirely against
  independent variable arrays without tracing complete inference forward passes
  locally.

  Running QEP using `quantize()` demands tracing complete model invocations
  entirely across all memory dimensions representing huge evaluation
  contexts. Large production pipelines often distribute metric sampling
  processes creating massive array components dumped securely onto distributed
  formats.

  This discrete pipeline parses pre-allocated data lakes, directly loading
  statistical mapping representations bounding structural variances seamlessly
  against floating nodes without memory bloat dependencies natively.

  Args:
    params: Isolated tree structure binding raw floating point model parameter
      arrays.
    abstract_quantized_params: PTQ definitions mapping exact structural
      requirements array limits restrictively.
    qep_quant_stats: Fully parsed dictionary array variables defining numerical
      error gradients cleanly.
    allow_extra_params: Toggles enforcement constraints mapping unlisted
      structural parameters cleanly.
    gptq_block_size: Structural parameter restriction configuring discrete array
      boundaries explicitly.
    gptq_damping_factor: Numeric variables scaling exact threshold limit
      inversions safely natively.
    correction_factor: Compensation scaling bounds resolving shift constraints
      across array outputs consistently.
    damping_factor: Hessian bound restrictions scaling float dependencies
      sequentially across inverses.
    apply_correction: Enables integrated mappings adjusting original layers
      entirely bounding structures.

  Returns:
    A fully quantized structures tree wrapping `qarray` limits bounding
    completely isolated discrete variables.
  """

  def _quantize(prepared: calibration.CalibratedQuantContext) -> Any:
    return _quantize_weight(
        prepared,
        QepRule(
            correction_factor=correction_factor,
            damping_factor=damping_factor,
            apply_correction=apply_correction,
        ),
        gptq_block_size,
        gptq_damping_factor,
    )

  return calibration.quantize_params_with_calibration(
      params,
      abstract_quantized_params,
      qep_quant_stats,
      '_qep',
      _quantize,
      allow_extra_params=allow_extra_params,
  )
