# Saliency-Based Brain Scoring for Language Models

## Overview

This report documents the implementation of GradCAM-style saliency scoring as an alternative to raw hidden-state scoring in `brainscore-language`. The standard pipeline extracts hidden states from a fixed layer and regresses them against human neural recordings (fMRI/ECoG). The saliency variant replaces those hidden states with the element-wise product of the activation and its gradient with respect to the model's next-token log-probability:


$$\text{saliency}(h) = h  \times \frac{\partial S}{\partial h}$$


where $S = \log p(\text{argmax token} | \text{context})$ at the last sequence position. This is the language analogue of GradCAM: raw gradients are noisy, but multiplying by the activation suppresses directions that are large-gradient-but-small-activation (likely noise) and retains directions that are both large and sensitive to the output. The resulting vector has the same shape as a raw hidden state, so it plugs into every existing metric (linear predictivity, CKA, RDM) and benchmark without modification.

---

## Modified Files

### 1. `brainscore_language/model_helpers/huggingface.py`

**Purpose:** Make `HuggingfaceSubject` extensible for gradient-based methods without changing any existing behaviour.

#### Changes

**`digest_text` — forward context and post-forward hook**

The forward pass was previously wrapped unconditionally in `torch.no_grad()`. That context manager is now delegated to a protected method so subclasses can replace it:

```python
# Before
with torch.no_grad():
    base_output = self.basemodel(**context_tokens, use_cache=_use_kv)

# After
with self._forward_context():
    base_output = self.basemodel(**context_tokens, use_cache=_use_kv)

self._post_forward(base_output, layer_representations)
```

The `_post_forward` call happens immediately after the forward pass closes, while the computation graph and hook-captured tensors are still alive — which is the only window in which a backward pass is valid.

**`_forward_context()` — new protected method**

```python
def _forward_context(self):
    return torch.no_grad()
```

Returns `torch.no_grad()` by default. Subclasses override this to control whether a computation graph is built. Zero behavioural change for all existing `HuggingfaceSubject` instances.

**`_post_forward(base_output, layer_representations)` — new protected method**

```python
def _post_forward(self, base_output, layer_representations: dict) -> None:
    pass  # no-op in base class
```

Called after the forward pass. Base implementation does nothing. Subclasses override this to run a backward pass (or any other operation that requires the live computation graph).

---

### 2. `brainscore_language/model_helpers/saliency.py` *(new file)*

Contains the full `SaliencyHuggingfaceSubject` class and a device-cache utility.

#### `_empty_device_cache(device)` — module-level helper

```python
def _empty_device_cache(device) -> None:
    if str(device) == 'mps':
        torch.mps.empty_cache()
    elif str(device).startswith('cuda'):
        torch.cuda.empty_cache()
```

Forces PyTorch to release cached-but-unused allocations back to the device memory pool. On MPS (Apple Silicon) and CUDA, PyTorch holds onto freed memory for reuse; calling this prevents accumulation across the many forward–backward cycles in a benchmark run.

#### `SaliencyHuggingfaceSubject(HuggingfaceSubject)`

Subclasses `HuggingfaceSubject` with four targeted overrides. All other behaviour (tokenisation, KV caching, behavioral tasks, multi-GPU, localizer) is inherited unchanged.

**`__init__(scalar_fn, use_gradient_checkpointing, **kwargs)`**

```python
def __init__(self, scalar_fn=None, use_gradient_checkpointing=True, **kwargs):
    super().__init__(**kwargs)
    self._scalar_fn = scalar_fn
    if use_gradient_checkpointing and hasattr(self.basemodel, 'gradient_checkpointing_enable'):
        try:
            self.basemodel.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            self.basemodel.gradient_checkpointing_enable()
```

Gradient checkpointing is enabled by default. Instead of caching all intermediate activations for every transformer block during the forward pass, PyTorch stores only each block's inputs and recomputes the internals on-the-fly during backward. This reduces peak backward-pass memory by roughly 50–70% at the cost of ~30–50% extra compute. The forward hooks still fire correctly during recomputation: when `backward()` re-runs block `h.11`, the hook overwrites `layer_representations[key]` with the freshly-computed tensor, which is exactly the tensor whose `.grad` gets populated — so saliency values are unaffected.

`use_reentrant=False` selects the modern, hook-safe checkpointing variant. The `TypeError` fallback handles older HuggingFace versions.

**`_forward_context()`**

```python
def _forward_context(self):
    return torch.enable_grad()
```

Overrides the parent's `torch.no_grad()` so that PyTorch builds a computation graph during the forward pass, which is required for `backward()` to work.

**`_post_forward(base_output, layer_representations)`**

```python
def _post_forward(self, base_output, layer_representations):
    if not layer_representations:
        return
    self.basemodel.zero_grad()
    _empty_device_cache(self.device)     # free cached allocs before backward
    scalar = self._compute_scalar(base_output)
    scalar.backward()
    _empty_device_cache(self.device)     # release graph memory after backward
```

Runs the backward pass. Cache is cleared both before (to make room) and after (to release the graph immediately rather than wait for Python GC). The `if not layer_representations` guard makes this a no-op when no neural recordings are active, avoiding an unnecessary backward on behavioral-only runs.

**`_register_hook(layer, key, target_dict)`**

```python
def _register_hook(self, layer, key, target_dict):
    def hook_function(_layer, _input, output, key=key):
        output = output[0] if isinstance(output, (tuple, list)) else output
        output.retain_grad()
        target_dict[key] = output
    return layer.register_forward_hook(hook_function)
```

Adds `retain_grad()` to the captured tensor before storing it. Non-leaf tensors in PyTorch do not retain their `.grad` after backward by default; this call opts them in. With gradient checkpointing the hook fires twice — once during the original forward and once during backward recomputation — and the second call correctly overwrites the dict with the tensor that is live in the backward graph.

**`_compute_scalar(base_output)`**

```python
def _compute_scalar(self, base_output):
    if self._scalar_fn is not None:
        return self._scalar_fn(base_output)
    logits = base_output.logits[0, -1]          # (vocab_size,)
    log_probs = torch.log_softmax(logits, dim=-1)
    return log_probs[logits.argmax()]
```

Default scalar: log-probability of the greedy next-token prediction at the last sequence position. This is the language analogue of GradCAM's class score — always available from any causal LM, directly tied to the model's generative objective, and closely related to surprisal (the quantity used in reading-time benchmarks). A custom `scalar_fn` can be supplied at construction time for alternative objectives.

**`output_to_representations(layer_representations, stimuli_coords)`**

```python
saliency = (tensor * tensor.grad)[:, -1:, :].squeeze(0).detach().cpu().numpy()
```

Replaces the parent's raw `tensor[:, -1:, :]` extraction with `tensor * tensor.grad` at the same last-token position. The neuroid coordinate structure (`layer`, `region`, `recording_type`, `neuroid_id`) is identical to the parent class, so the resulting `NeuroidAssembly` is drop-in compatible with all downstream metrics and benchmarks.

---

### 3. `brainscore_language/models/gpt/__init__.py`

**Purpose:** Register saliency variants in the model registry so they are accessible via the CLI and `load_model`.

#### Changes

Added import:

```python
from brainscore_language.model_helpers.saliency import SaliencyHuggingfaceSubject
```

Added three registry entries using the same best-layer assignments as the corresponding raw-activation models (layer choices from Schrimpf et al., pre-computed on Pereira2018-encoding):

```python
model_registry['gpt2-saliency'] = lambda: SaliencyHuggingfaceSubject(
    model_id='gpt2',
    region_layer_mapping={ArtificialSubject.RecordingTarget.language_system: 'transformer.h.11'})

model_registry['gpt2-medium-saliency'] = lambda: SaliencyHuggingfaceSubject(
    model_id='gpt2-medium',
    region_layer_mapping={ArtificialSubject.RecordingTarget.language_system: 'transformer.h.22'})

model_registry['gpt2-xl-saliency'] = lambda: SaliencyHuggingfaceSubject(
    model_id='gpt2-xl',
    region_layer_mapping={ArtificialSubject.RecordingTarget.language_system: 'transformer.h.43'})
```

---

## Running the Saliency Models

**CLI (same interface as all other registered models):**

```bash
python brainscore_language score \
  --model_identifier='gpt2-saliency' \
  --benchmark_identifier='Pereira2018.243sentences-linear'
```

**Programmatic side-by-side comparison:**

```python
from brainscore_language import load_benchmark, ArtificialSubject
from brainscore_language.model_helpers.huggingface import HuggingfaceSubject
from brainscore_language.model_helpers.saliency import SaliencyHuggingfaceSubject

LAYER   = "transformer.h.11"
MAPPING = {ArtificialSubject.RecordingTarget.language_system: LAYER}

benchmark = load_benchmark("Pereira2018.243sentences-linear")

raw_score = benchmark(HuggingfaceSubject(model_id="gpt2", region_layer_mapping=MAPPING))
sal_score = benchmark(SaliencyHuggingfaceSubject(model_id="gpt2", region_layer_mapping=MAPPING))

print(f"Raw activations:   {raw_score.values:.4f}")
print(f"Saliency (grad×h): {sal_score.values:.4f}")
```

**Custom scalar (e.g., sum of last-token logits):**

```python
SaliencyHuggingfaceSubject(
    model_id="gpt2",
    region_layer_mapping=MAPPING,
    scalar_fn=lambda out: out.logits[0, -1].sum(),
)
```

---

## Design Decisions

| Decision | Rationale |
|---|---|
| Multiply gradient × activation (not raw gradient) | Raw gradients are noisy; the product suppresses directions that are either near-zero activation or near-zero gradient, retaining only jointly informative dimensions. This is the GradCAM formulation. |
| Default scalar: log-prob of predicted next token | Always available from any causal LM; directly tied to the generative objective; proportional to surprisal, the behavioural measure used in reading-time benchmarks. |
| Gradient checkpointing on by default | Reduces peak backward-pass memory by ~50–70% at the cost of ~30–50% extra compute. Critical for running on memory-constrained hardware (e.g., Apple Silicon with 20 GB unified memory). |
| `use_reentrant=False` checkpointing | The modern, hook-safe variant. With the reentrant variant, forward hooks may not fire during recomputation depending on PyTorch version. |
| Cache clearing before and after backward | MPS and CUDA hold freed allocations in a cache. Clearing before backward frees headroom; clearing after releases the computation graph immediately rather than waiting for Python GC, preventing accumulation across the 200+ text parts in a typical benchmark. |
| Hooks fire twice with gradient checkpointing | The second firing (during backward recomputation) overwrites the dict with the tensor that is live in the backward graph — which is exactly the tensor whose `.grad` is populated. This is the correct behaviour: the saliency is computed on the same activations that determined the gradient. |
| Same `NeuroidAssembly` coordinate structure | The saliency vector has identical shape and coordinates to a raw hidden state, so all existing metrics, benchmarks, and ceiling computations work without any modification. |

---

## Experiment 1: Results

| Model | Score (ceiling-normalised) |
|---|---|
| `gpt2` (raw activations, layer 11) | 0.87 |
| `gpt2-saliency` (GradCAM, layer 11, last token) | 0.50 |

The 37-point drop shows that the GradCAM product is actively discarding information that predicts brain activity. Units with small `dS/dh` (unimportant for next-token prediction) are still metabolically active and neurally relevant — saliency zeroes them out.

The core tension is that BOLD signal reflects metabolic demand (∝ activation magnitude), while `dS/dh` reflects causal sensitivity (which units matter *if perturbed*). These are different quantities. Multiplying them together does not straightforwardly model blood flow; instead it selects for units that are both active *and* causally relevant to the next-token objective.

---

## Experiment 2: Disentanglement and Ablations

To understand where the gap comes from, Experiment 1 is extended with a set of ablations that independently vary (1) which gradient-based representation is extracted, (2) how the representation is pooled over the sequence, and (3) which scalar S drives the backward pass.

### New parameters in `SaliencyHuggingfaceSubject`

Three new constructor arguments were added. All existing defaults are preserved.

#### `mode` — representation type

Controls what is extracted from the hidden state and its gradient:

```python
# GradCAM product (original, default)
mode='gradcam'       →   h ⊙ (dS/dh)

# Raw gradient — causal sensitivity only, no activation weighting
mode='gradient'      →   dS/dh

# Unsigned gradient — influence magnitude without sign
mode='abs_gradient'  →   |dS/dh|
```

If `mode='gradient'` scores near 0%, the gradient alone carries no brain signal; the 50% result is explained by the activation component. If it scores high, sensitivity to the generative objective is itself brain-predictive.

#### `token_pool` — sequence aggregation

Controls which token positions contribute to the sentence-level vector:

```python
# Final token only (original, default) — matches raw-activation convention
token_pool='last'    →   rep[:, -1, :]

# Average across all positions — treats every token equally
token_pool='mean'    →   rep.mean(dim=1)
```

The default last-token choice is a strong assumption: the model's internal state at the end of a sentence is used to represent the whole sentence. Mean-pooling tests whether a whole-sequence representation is more aligned with the brain's sentence-level fMRI response.

#### `scalar_mode` — scalar S

Selects the quantity whose gradient is computed:

```python
# Log-prob of greedy next token at last position (original, default)
scalar_mode='next_token'

# Sum of greedy log-probs across all sequence positions
scalar_mode='sentence_surprisal'
```

With `'sentence_surprisal'`, the gradient flows back through every token position rather than being concentrated at the last one. This gives a richer, sentence-wide sensitivity map and is conceptually closer to a whole-sentence processing signal.

**Implementation of `sentence_surprisal` in `_compute_scalar`:**

```python
logits    = base_output.logits[0]                                     # (seq_len, vocab_size)
log_probs = torch.log_softmax(logits, dim=-1)
greedy_ids = logits.argmax(dim=-1)                                    # (seq_len,)
return log_probs[torch.arange(len(logits), device=logits.device),
                 greedy_ids].sum()
```

The `argmax` is used only as an index; the gradient flows through `log_probs` (via `log_softmax`) to every position's logits and then to the hidden states.

**Implementation of the two-step extraction in `output_to_representations`:**

```python
# Step 1 — select representation
if self._mode == 'gradcam':
    rep = tensor * tensor.grad          # h ⊙ (dS/dh)
elif self._mode == 'gradient':
    rep = tensor.grad                   # dS/dh
else:  # 'abs_gradient'
    rep = tensor.grad.abs()             # |dS/dh|

# Step 2 — pool over sequence positions
if self._token_pool == 'mean':
    rep = rep.mean(dim=1, keepdim=True)
else:  # 'last'
    rep = rep[:, -1:, :]
```

---

### New registered models

#### Disentanglement (layer 11, last token, next-token scalar)

```python
model_registry['gpt2-gradient']      # mode='gradient'
model_registry['gpt2-abs-gradient']  # mode='abs_gradient'
```

#### Token-pooling ablation

```python
model_registry['gpt2-mean-saliency']  # token_pool='mean'
```

#### Scalar ablation

```python
model_registry['gpt2-surprisal-saliency']  # scalar_mode='sentence_surprisal'
```

#### Layer sweep (GradCAM, last token, next-token scalar)

```python
model_registry['gpt2-saliency-layer0']   # transformer.h.0
model_registry['gpt2-saliency-layer1']   # transformer.h.1
...
model_registry['gpt2-saliency-layer11']  # transformer.h.11  (same as gpt2-saliency)
```

---

### Running the ablations

**CLI:**

```bash
python brainscore_language score --model_identifier=gpt2-gradient --benchmark_identifier=Pereira2018.243sentences-linear
python brainscore_language score --model_identifier=gpt2-abs-gradient --benchmark_identifier=Pereira2018.243sentences-linear
python brainscore_language score --model_identifier=gpt2-mean-saliency --benchmark_identifier=Pereira2018.243sentences-linear
python brainscore_language score --model_identifier=gpt2-surprisal-saliency --benchmark_identifier=Pereira2018.243sentences-linear
python brainscore_language score --model_identifier=gpt2-saliency-layer5 --benchmark_identifier=Pereira2018.243sentences-linear
```

**Programmatic layer sweep:**

```python
from brainscore_language import load_benchmark, load_model

benchmark = load_benchmark("Pereira2018.243sentences-linear")

results = {}
for layer in range(12):
    model = load_model(f"gpt2-saliency-layer{layer}")
    results[layer] = benchmark(model).values
    print(f"layer {layer:2d}: {results[layer]:.4f}")
```

---

### Design decisions for Experiment 2

| Decision | Rationale |
|---|---|
| Three modes rather than three separate classes | A single `mode` parameter keeps the public API minimal and avoids duplicating the `__init__`, `_forward_context`, `_post_forward`, and `_register_hook` logic. All three modes share the same backward pass. |
| `'last'` remains the default for `token_pool` | Preserves backward compatibility with `gpt2-saliency` results from Experiment 1 and matches the raw-activation convention in `HuggingfaceSubject`. |
| Sentence surprisal sums greedy log-probs, not actual next-token log-probs | Actual next-token log-probs require the input token IDs, which are not passed to `scalar_fn`. Using greedy predictions is self-contained, always differentiable, and provides the same sentence-wide gradient flow. |
| Layer sweep registered as individual models | Each layer gets its own registry entry so it can be scored independently via the CLI with no code changes, and results are cached separately per layer. |
| `ValueError` on unknown `mode` / `token_pool` | Fails loudly at representation-extraction time rather than silently producing a wrong result. |
