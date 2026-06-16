# [RFC]: Quantization Code Refactoring for Improved Maintainability and Usability

> **Labels**: RFC
> **Status**: Open for Feedback
> **Feedback Period**: At least 1 week

---

## Motivation

The vllm-ascend quantization subsystem has grown significantly as new quantization formats and hardware variants have been added. Today it encompasses **~30 concrete scheme classes**, **27 registered scheme entries** in the main registry (plus 5 in the 310P registry), and spans **~47 quantization-related Python files**. While the three-layer architecture (Config → Registry → Scheme) has served us well, the rapid expansion has introduced substantial code redundancy, deeply branched logic, and maintenance burden that now impedes development velocity.

Several concrete pain points have emerged:

1. **MoE `apply()` boilerplate**: Nearly every MoE scheme duplicates a ~30–40 line pattern covering shared-expert counting, expert selection, force-load-balancing, and fused-experts dispatch. This exact pattern is copy-pasted across 7+ files (`w8a8_dynamic.py`, `w8a8_mxfp8.py`, `w4a8.py`, `w4a4_mxfp4.py`, `w4a8_mxfp4.py`, `w4a16.py`, and the 310P variant). Any bug fix or optimization must be applied to every copy.

2. **Weight post-processing duplication**: Almost every linear scheme repeats the same `transpose → maybe_trans_nz → flatten` sequence in `process_weights_after_loading()`. Similarly, all MX-type MoE schemes duplicate identical 3D transpose + reshape logic for scale tensors.

3. **Parallel 310P registry**: The `_310p/quantization/` directory duplicates the entire registry infrastructure, base class parameter specs, and weight processing logic from the main registry, creating a maintenance fork within the same repository.

4. **Overly complex individual files**: `w4a8.py` alone is 777 lines, handling four different weight format variants (ModelSlim old, ModelSlim 1.0.0 per-group, ModelSlim 1.0.0 per-channel, LLM-Compressor) with separate code paths for each.

5. **Config class boilerplate**: All three config classes (`AscendModelSlimConfig`, `AscendCompressedTensorsConfig`, `AscendFp8Config`) repeat identical implementations of `get_supported_act_dtypes()`, `get_min_capability()`, and `get_config_filenames()`.

6. **Hardcoded model mapping bloat**: `packed_modules_model_mapping` in `modelslim_config.py` spans 260 lines with many near-duplicate entries for architecturally similar models.

These issues make it error-prone to add new quantization methods, difficult to fix bugs consistently, and challenging for new contributors to understand the codebase. As the Ascend ecosystem continues to grow with new quantization formats (MXFP4, FlatQuant, LAOS, etc.), addressing this technical debt is urgent.

## Proposed Change

We propose a systematic refactoring of the vllm-ascend quantization code organized around four pillars: **extract common logic**, **unify the registry**, **simplify configuration**, and **improve extensibility**.

### Pillar 1: Extract Common MoE Logic into Base Class

Move the duplicated MoE `apply()` boilerplate into `AscendMoEScheme` (the abstract base class) as a template method. Concrete subclasses would only override the quantization-specific kernel call, not the entire expert selection and dispatch flow.

```python
class AscendMoEScheme(ABC):
    def apply(self, layer, x, router_logits, top_k, ...):
        # Common logic: shared experts, expert selection, load balancing
        topk_weights, topk_ids = self._select_experts(...)
        # Dispatch to subclass-specific quantized kernel
        return self._quantized_moe_forward(layer, x, topk_weights, topk_ids, ...)

    @abstractmethod
    def _quantized_moe_forward(self, layer, x, topk_weights, topk_ids, ...):
        """Subclasses implement only the quantized computation."""
        ...
```

Similarly, extract the common `process_weights_after_loading()` patterns into reusable utility functions or base class methods:

```python
def _postprocess_linear_weights(layer):
    """Common linear weight post-processing: transpose + NZ cast + flatten."""
    layer.weight.data = maybe_trans_nz(layer.weight.data.transpose(0, 1).contiguous())
    layer.weight_scale.data = layer.weight_scale.data.flatten()
    if hasattr(layer, 'weight_offset'):
        layer.weight_offset.data = layer.weight_offset.data.flatten()
```

### Pillar 2: Unify the 310P Registry

Eliminate the separate `_310p/quantization/methods/registry.py` by extending the main registry to support hardware-variant registration. Instead of a parallel registry, use a variant tag or platform check within the unified registry:

```python
@register_scheme("W8A8", "linear", variant="310p")
class AscendW8A8Linear310pScheme(AscendLinearScheme):
    ...
```

The scheme lookup would consider the current platform (910B/C vs. 310P) and select the appropriate variant automatically. This removes the duplicated registry infrastructure and makes 310P-specific overrides explicit and discoverable.

### Pillar 3: Simplify Configuration Classes

Introduce an `AscendQuantizationConfig` mixin or intermediate base class that provides the shared boilerplate methods. The three concrete configs would then only implement their unique logic (parsing, dispatch):

```python
class AscendQuantizationConfig(QuantizationConfig):
    """Shared base for all Ascend quantization configs."""

    @classmethod
    def get_supported_act_dtypes(cls):
        return [torch.int8, torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls):
        raise NotImplementedError("Ascend hardware does not support get_min_capability.")

    @classmethod
    def get_config_filenames(cls):
        return []
```

Additionally, refactor `packed_modules_model_mapping` by grouping architecturally equivalent models and using a pattern-based matching approach rather than exhaustive per-model enumeration.

### Pillar 4: Decompose Complex Schemes

Break down overly complex scheme files (especially `w4a8.py` at 777 lines) by extracting weight-format-specific logic into dedicated helper classes or strategy objects. Each weight format variant (ModelSlim legacy, ModelSlim 1.0.0, LLM-Compressor) should be a separate, testable unit:

```python
class W4A8WeightLoader(ABC):
    @abstractmethod
    def load_weights(self, layer, loaded_weight): ...
    @abstractmethod
    def process_weights(self, layer): ...

class ModelSlimLegacyW4A8Loader(W4A8WeightLoader): ...
class ModelSlimV1W4A8Loader(W4A8WeightLoader): ...
class CompressedTensorsW4A8Loader(W4A8WeightLoader): ...
```

### Key Design Decisions

- **Template Method pattern for MoE**: The base class owns the expert selection and dispatch flow; subclasses only implement the quantized kernel call. This is preferred over composition because the flow is stable while only the kernel varies.
- **Unified registry with variant tags**: Rather than maintaining separate registries per hardware variant, a single registry with optional variant tags keeps the dispatch logic centralized and makes it obvious which schemes have hardware-specific overrides.
- **Incremental migration**: The refactoring should be done in phases (one pillar at a time) with full backward compatibility at each step. Each phase should include regression tests to ensure no functional changes.
- **No API surface changes**: The refactoring is internal only. The `@register_scheme` decorator, config class registration, and adapter layer remain unchanged from the user's perspective.

## Related PRs

No specific PRs identified yet. Implementation PRs will be created as the refactoring progresses through each pillar.

## Feedback Period

At least 1 week. We welcome feedback on the proposed architecture, concerns about edge cases, and suggestions for additional areas to address.

## CC List

@vllm-project/vllm-ascend-maintainers

## Any Other Things

### Alternatives Considered

1. **Full rewrite using upstream vLLM's `CompressedTensorsScheme` pattern**: While upstream vLLM's compressed-tensors module uses a similar scheme-based architecture, adopting it wholesale would require significant changes to the adapter layer and break the existing `@register_scheme` decorator pattern. We prefer to evolve the existing architecture incrementally.

2. **Code generation for scheme boilerplate**: Auto-generating scheme classes from a YAML specification was considered but rejected due to reduced debuggability and the learning curve for contributors.

### Success Metrics

- Reduce total lines of quantization code by ~20–30% through deduplication
- Reduce the number of files that need to change when adding a new MoE quantization method from 3+ to 1
- Reduce `w4a8.py` from 777 lines to under 300 lines
- Eliminate the parallel 310P registry entirely
- Zero functional regressions (verified by existing quantization tests)

---

> Please take a look at previous [RFCs](https://github.com/vllm-project/vllm-ascend/issues?q=label%3ARFC+sort%3Aupdated-desc) for reference.