import functools
import numpy as np
import torch
from numpy.core import defchararray
from transformers.modeling_outputs import CausalLMOutput
from typing import Callable, Dict, Literal, Optional, Tuple

from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_language.model_helpers.huggingface import HuggingfaceSubject


def _empty_device_cache(device) -> None:
    """Release cached (but unused) allocations back to the device memory pool."""
    device_str = str(device)
    if device_str == 'mps':
        torch.mps.empty_cache()
    elif device_str.startswith('cuda'):
        torch.cuda.empty_cache()


class SaliencyHuggingfaceSubject(HuggingfaceSubject):
    """A HuggingfaceSubject that replaces raw hidden states with gradient-based representations.

    Instead of recording raw activations, this class computes one of three representations
    derived from the gradient of a scalar S w.r.t. the hidden state h at a given layer:

    * ``'gradcam'`` (default) — GradCAM-style product ``h ⊙ (dS/dh)``:
      raw gradients are noisy; multiplying by the activation suppresses uninformative
      directions (large gradient, small activation) and retains directions that are both
      large *and* sensitive to S.

    * ``'gradient'`` — the raw gradient ``dS/dh``:
      captures which directions are causally important for S, irrespective of activation
      magnitude.  Useful for disentangling the contribution of the gradient from the
      activation in the GradCAM product.

    * ``'abs_gradient'`` — ``|dS/dh|``:
      unsigned sensitivity; removes sign information so only magnitude of influence is kept.

    The sequence position used to form the sentence-level vector is controlled by
    ``token_pool``:

    * ``'last'`` (default) — representation at the final token position, matching the
      convention used by :class:`HuggingfaceSubject` for raw activations.

    * ``'mean'`` — average across all token positions.  The brain processes the whole
      sentence; mean-pooling gives every token equal weight.

    The scalar S itself is selected via ``scalar_mode`` (or overridden with ``scalar_fn``):

    * ``'next_token'`` (default) — ``log P(argmax token | context)`` at the last position.
      This is the standard GradCAM scalar: tied to the generative objective, always
      available, related to surprisal.

    * ``'sentence_surprisal'`` — sum of ``log P(argmax token_i | context_{<i})`` across
      all sequence positions.  Provides a richer, sentence-level signal whose gradient
      flows back through every position rather than only the last one.

    All resulting vectors have the same shape as ordinary hidden states ``(hidden_dim,)``,
    so they slot directly into every existing metric (linear predictivity, CKA, RDM, …)
    and benchmark without any further modification.

    Example:
        >>> subject = SaliencyHuggingfaceSubject(
        ...     model_id="gpt2",
        ...     region_layer_mapping={"language_system": ["transformer.h.11"]},
        ...     mode="gradient",
        ...     token_pool="mean",
        ...     scalar_mode="sentence_surprisal",
        ... )
    """

    def __init__(
        self,
        scalar_fn: Optional[Callable[[CausalLMOutput], torch.Tensor]] = None,
        use_gradient_checkpointing: bool = True,
        mps_fallback_to_cpu: bool = True,
        mode: Literal['gradcam', 'gradient', 'abs_gradient'] = 'gradcam',
        token_pool: Literal['last', 'mean'] = 'last',
        scalar_mode: Literal['next_token', 'sentence_surprisal'] = 'next_token',
        **kwargs,
    ):
        """Initialise a gradient-based subject.

        Args:
            scalar_fn: A callable that receives the model's ``CausalLMOutput``
                and returns a scalar (0-dimensional) ``torch.Tensor``.  This is
                the quantity S whose gradient is computed.  When provided, it takes
                priority over ``scalar_mode``.  Must be differentiable w.r.t. the
                hidden states.
            use_gradient_checkpointing: When ``True`` (default), enables PyTorch
                gradient checkpointing on the base model.  Reduces peak backward-pass
                memory by ~50–70% at the cost of ~30–50% extra compute.  The forward
                hooks still fire correctly during recomputation.
            mps_fallback_to_cpu: When ``True`` (default) and the model was loaded on
                MPS (Apple Silicon), move it to CPU before running.  MPS shared memory
                is insufficient for gradient-tracked forward passes on memory-constrained
                machines.  Set to ``False`` on machines with ample free unified memory.
            mode: Which gradient-based representation to extract.

                * ``'gradcam'`` — ``h ⊙ (dS/dh)`` (default).
                * ``'gradient'`` — raw gradient ``dS/dh``.
                * ``'abs_gradient'`` — unsigned gradient ``|dS/dh|``.

            token_pool: How to aggregate the representation across sequence positions.

                * ``'last'`` — use only the final token position (default, matches the
                  raw-activation convention in :class:`HuggingfaceSubject`).
                * ``'mean'`` — average across all token positions.

            scalar_mode: Which built-in scalar S to use when ``scalar_fn`` is ``None``.

                * ``'next_token'`` — ``log P(argmax token | context)`` at the last
                  sequence position (default).
                * ``'sentence_surprisal'`` — sum of ``log P(argmax token_i)`` across
                  all positions; gradient flows back through every token, not just the
                  last one.

            **kwargs: All remaining keyword arguments are forwarded verbatim to
                :class:`~brainscore_language.model_helpers.huggingface.HuggingfaceSubject`.
        """
        super().__init__(**kwargs)
        self._scalar_fn = scalar_fn
        self._mode = mode
        self._token_pool = token_pool
        self._scalar_mode = scalar_mode

        if mps_fallback_to_cpu and str(self.device) == 'mps':
            print("Saliency subject: moving model from MPS to CPU "
                  "(MPS shared memory is insufficient for gradient-tracked forward passes "
                  "on memory-constrained machines; set mps_fallback_to_cpu=False to override).")
            self.basemodel.to('cpu')
            self.device = 'cpu'

        if use_gradient_checkpointing and hasattr(self.basemodel, 'gradient_checkpointing_enable'):
            try:
                # use_reentrant=False is the modern, hook-safe variant
                self.basemodel.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                # older HuggingFace versions do not accept the kwargs argument
                self.basemodel.gradient_checkpointing_enable()

    # ------------------------------------------------------------------
    # Extension points (override HuggingfaceSubject hooks)
    # ------------------------------------------------------------------

    def _forward_context(self):
        """Enable gradient tracking during the forward pass.

        Overrides the parent's ``torch.no_grad()`` default so that PyTorch
        builds a computation graph, which is required for the subsequent
        backward pass.

        Returns:
            torch.enable_grad context manager.
        """
        return torch.enable_grad()

    def _post_forward(
        self,
        base_output: CausalLMOutput,
        layer_representations: dict,
    ) -> None:
        """Run a backward pass to populate ``.grad`` on the captured activations.

        Called by :meth:`digest_text` immediately after the forward pass.  The
        gradient of the scalar S w.r.t. every captured hidden-state tensor is
        computed here and stored in ``tensor.grad``, ready for
        :meth:`output_to_representations`.

        Args:
            base_output: Raw ``CausalLMOutput`` from the base model.
            layer_representations: The activation tensors captured by the forward
                hooks, keyed by ``(recording_target, recording_type, layer_name)``.
                An empty dict is a no-op (e.g., behavioral-only runs where no
                neural hooks were registered).
        """
        if not layer_representations:
            return
        self.basemodel.zero_grad()
        _empty_device_cache(self.device)   # free any unreleased cached allocs first
        scalar = self._compute_scalar(base_output)
        scalar.backward()
        _empty_device_cache(self.device)   # release graph memory immediately after

    def _register_hook(
        self,
        layer: torch.nn.Module,
        key: Tuple[str, str, str],
        target_dict: dict,
    ):
        """Register a forward hook that preserves the gradient of the output.

        Because the captured tensors are non-leaf nodes in PyTorch's autograd
        graph, their ``.grad`` attribute is ordinarily discarded after the
        backward pass.  Calling ``retain_grad()`` before storing prevents this.

        Args:
            layer: The module whose output should be captured.
            key: Identifier tuple ``(recording_target, recording_type, layer_name)``.
            target_dict: Dict in which the captured tensor will be stored.

        Returns:
            A removable hook handle.
        """
        def hook_function(_layer, _input, output, key=key):
            output = output[0] if isinstance(output, (tuple, list)) else output
            output.retain_grad()
            target_dict[key] = output

        return layer.register_forward_hook(hook_function)

    # ------------------------------------------------------------------
    # Scalar computation
    # ------------------------------------------------------------------

    def _compute_scalar(self, base_output: CausalLMOutput) -> torch.Tensor:
        """Compute the scalar S used for the backward pass.

        Dispatch order:
        1. ``scalar_fn`` supplied at construction — used as-is.
        2. ``scalar_mode='sentence_surprisal'`` — sum of greedy log-probs
           across all sequence positions.
        3. ``scalar_mode='next_token'`` (default) — log-prob of the greedy
           next-token prediction at the last sequence position.

        Args:
            base_output: The ``CausalLMOutput`` from the model's forward pass.

        Returns:
            A 0-dimensional differentiable ``torch.Tensor``.
        """
        if self._scalar_fn is not None:
            return self._scalar_fn(base_output)

        if self._scalar_mode == 'sentence_surprisal':
            # logits: (batch=1, seq_len, vocab_size)
            logits = base_output.logits[0]                          # (seq_len, vocab_size)
            log_probs = torch.log_softmax(logits, dim=-1)
            # Greedy prediction at each position; argmax is used only as an index,
            # so the gradient flows through log_probs (via log_softmax) to logits.
            greedy_ids = logits.argmax(dim=-1)                      # (seq_len,)
            return log_probs[
                torch.arange(len(logits), device=logits.device),
                greedy_ids
            ].sum()

        # Default: 'next_token'
        # log-prob of the greedy prediction at the final sequence position.
        logits = base_output.logits[0, -1]                          # (vocab_size,)
        log_probs = torch.log_softmax(logits, dim=-1)
        return log_probs[logits.argmax()]

    # ------------------------------------------------------------------
    # Representation extraction
    # ------------------------------------------------------------------

    def output_to_representations(
        self,
        layer_representations: Dict[Tuple[str, str, str], torch.Tensor],
        stimuli_coords: dict,
    ) -> NeuroidAssembly:
        """Build a NeuroidAssembly from gradient-based representation vectors.

        For each recorded layer the representation is computed in two steps:

        **Step 1 — select representation (controlled by** ``mode`` **):**

        .. code-block:: text

            gradcam      →  h ⊙ (dS/dh)          element-wise product
            gradient     →  dS/dh                  raw gradient
            abs_gradient →  |dS/dh|                unsigned gradient magnitude

        **Step 2 — pool over sequence positions (controlled by** ``token_pool`` **):**

        .. code-block:: text

            last  →  rep[:, -1, :]        final token position only
            mean  →  rep.mean(dim=1)      average across all token positions

        The resulting array has shape ``(1, hidden_dim)`` per layer, identical to
        what :meth:`HuggingfaceSubject.output_to_representations` produces for raw
        activations, so all downstream metrics work unchanged.

        Args:
            layer_representations: Activation tensors (with ``.grad`` populated)
                captured by the forward hooks, keyed by
                ``(recording_target, recording_type, layer_name)``.
            stimuli_coords: Presentation-dimension coordinates to attach to the
                returned assembly (stimulus text, context, part_number, …).

        Returns:
            A ``NeuroidAssembly`` of shape ``(1, total_hidden_dim)`` where
            ``total_hidden_dim`` is the sum of hidden dimensions across all
            recorded layers.

        Raises:
            RuntimeError: If any captured tensor has ``None`` gradient, which
                indicates the layer's output is detached from the computation graph.
            ValueError: If ``mode`` is not one of ``'gradcam'``, ``'gradient'``,
                ``'abs_gradient'``.
        """
        saliency_parts = []
        for key, tensor in layer_representations.items():
            if tensor.grad is None:
                layer_name = key[2]
                raise RuntimeError(
                    f"Gradient is None for layer '{layer_name}'. "
                    "The layer output may be detached from the computation graph "
                    "(e.g., via an internal stop_gradient or detach() call). "
                    "Try a layer earlier in the model, or inspect whether the "
                    "model uses gradient checkpointing."
                )

            # --- Step 1: select representation based on mode ---
            if self._mode == 'gradcam':
                # GradCAM product: large only where activation AND gradient are large
                rep = tensor * tensor.grad              # h ⊙ (dS/dh)
            elif self._mode == 'gradient':
                # Raw gradient: causal sensitivity irrespective of activation magnitude
                rep = tensor.grad                       # dS/dh
            elif self._mode == 'abs_gradient':
                # Unsigned gradient: magnitude of influence, sign-agnostic
                rep = tensor.grad.abs()                 # |dS/dh|
            else:
                raise ValueError(
                    f"Unknown mode '{self._mode}'. "
                    "Choose from 'gradcam', 'gradient', 'abs_gradient'."
                )

            # --- Step 2: pool over sequence positions ---
            if self._token_pool == 'mean':
                # Average across all token positions: (batch, seq_len, hidden) → (batch, 1, hidden)
                rep = rep.mean(dim=1, keepdim=True)
            elif self._token_pool == 'last':
                # Final token only: (batch, seq_len, hidden) → (batch, 1, hidden)
                rep = rep[:, -1:, :]
            else:
                raise ValueError(
                    f"Unknown token_pool '{self._token_pool}'. "
                    "Choose from 'last', 'mean'."
                )

            saliency_parts.append(rep.squeeze(0).detach().cpu().numpy())

        representation_values = np.concatenate(saliency_parts, axis=-1)

        neuroid_coords = {
            'layer': (
                'neuroid',
                np.concatenate([
                    [layer_name] * tensor.shape[-1]
                    for (_, _, layer_name), tensor in layer_representations.items()
                ]),
            ),
            'region': (
                'neuroid',
                np.concatenate([
                    [recording_target] * tensor.shape[-1]
                    for (recording_target, _, _), tensor in layer_representations.items()
                ]),
            ),
            'recording_type': (
                'neuroid',
                np.concatenate([
                    [recording_type] * tensor.shape[-1]
                    for (_, recording_type, _), tensor in layer_representations.items()
                ]),
            ),
            'neuron_number_in_layer': (
                'neuroid',
                np.concatenate([
                    np.arange(tensor.shape[-1])
                    for tensor in layer_representations.values()
                ]),
            ),
        }
        neuroid_coords['neuroid_id'] = 'neuroid', functools.reduce(
            defchararray.add,
            [
                neuroid_coords['layer'][1],
                '--',
                neuroid_coords['neuron_number_in_layer'][1].astype(str),
            ],
        )

        return NeuroidAssembly(
            representation_values,
            coords={**stimuli_coords, **neuroid_coords},
            dims=['presentation', 'neuroid'],
        )
