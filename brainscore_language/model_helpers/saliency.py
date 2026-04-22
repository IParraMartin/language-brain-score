import functools
import numpy as np
import torch
from numpy.core import defchararray
from transformers.modeling_outputs import CausalLMOutput
from typing import Callable, Dict, Optional, Tuple

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
    """A HuggingfaceSubject that replaces raw hidden states with GradCAM-style saliency.

    Instead of recording the activation h at a given layer, this class records
    the element-wise product  h * (dS/dh), where S is a scalar derived from the
    model's output (by default the log-probability of the predicted next token).
    This is the language-domain analogue of GradCAM: raw gradients are noisy, but
    multiplying by the activation itself suppresses uninformative directions and
    highlights units that are both large *and* sensitive to the output scalar.

    The resulting saliency vectors have the same shape as ordinary hidden states, so
    they slot directly into every existing metric (linear predictivity, CKA, RDM, …)
    and benchmark without any further modification.

    Example:
        >>> subject = SaliencyHuggingfaceSubject(
        ...     model_id="gpt2",
        ...     region_layer_mapping={"language_system": ["transformer.h.11"]},
        ... )
        >>> subject.start_neural_recording(
        ...     recording_target=ArtificialSubject.RecordingTarget.language_system,
        ...     recording_type=ArtificialSubject.RecordingType.fMRI,
        ... )
        >>> output = subject.digest_text(["the quick brown fox"])
        >>> output["neural"].shape  # (presentations, neuroids) same as raw-activation case

    Note:
        Gradient checkpointing is enabled by default to reduce peak memory during
        the backward pass.  It trades ~30-50% extra compute for a large reduction
        in graph memory by recomputing activations on-the-fly instead of caching
        them.  Set ``use_gradient_checkpointing=False`` to disable if the model
        does not support it.
    """

    def __init__(
        self,
        scalar_fn: Optional[Callable[[CausalLMOutput], torch.Tensor]] = None,
        use_gradient_checkpointing: bool = True,
        mps_fallback_to_cpu: bool = True,
        **kwargs,
    ):
        """Initialise a saliency-based subject.

        Args:
            scalar_fn: A callable that receives the model's ``CausalLMOutput``
                and returns a scalar (0-dimensional) ``torch.Tensor``.  This is
                the quantity S whose gradient is computed.  When ``None`` the
                default is the log-probability of the greedy next-token prediction::

                    log_softmax(logits[0, -1])[argmax(logits[0, -1])]

                Custom scalars must be differentiable w.r.t. the hidden states.
            use_gradient_checkpointing: When ``True`` (default), enables PyTorch
                gradient checkpointing on the base model.  This reduces peak
                backward-pass memory by storing only block inputs and recomputing
                activations on demand, at the cost of ~30-50% extra FLOPs.
                The forward hooks still fire correctly during recomputation, so
                the saliency values are unaffected.
            mps_fallback_to_cpu: When ``True`` (default) and the model was loaded
                on MPS (Apple Silicon), move it to CPU before running.  MPS
                unified memory is shared with the entire OS; on machines with
                16-20 GB RAM the system routinely occupies 18+ GB, leaving too
                little headroom for gradient-tracked forward passes on long
                contexts.  CPU uses virtual memory and avoids hard OOM errors.
                Set to ``False`` on machines with ample free unified memory
                (e.g., 64 GB Apple Silicon).
            **kwargs: All remaining keyword arguments are forwarded verbatim to
                :class:`~brainscore_language.model_helpers.huggingface.HuggingfaceSubject`.
        """
        super().__init__(**kwargs)
        self._scalar_fn = scalar_fn
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
            layer_representations: The activation tensors captured by the
                forward hooks, keyed by ``(recording_target, recording_type,
                layer_name)``.  An empty dict is a no-op (e.g., behavioral-only
                runs where no neural hooks were registered).
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
    # Saliency computation
    # ------------------------------------------------------------------

    def _compute_scalar(self, base_output: CausalLMOutput) -> torch.Tensor:
        """Compute the scalar S used for the backward pass.

        The default implementation returns the log-probability of the greedy
        next-token prediction at the last sequence position.  This is the
        language analogue of GradCAM's class score: it is always available
        from a causal LM, connects directly to the model's language-generation
        objective, and relates to surprisal (used in reading-time benchmarks).

        A custom ``scalar_fn`` passed to ``__init__`` takes precedence over
        this default.

        Args:
            base_output: The ``CausalLMOutput`` from the model's forward pass.

        Returns:
            A 0-dimensional differentiable ``torch.Tensor``.
        """
        if self._scalar_fn is not None:
            return self._scalar_fn(base_output)
        # logits: (batch=1, seq_len, vocab_size)
        logits = base_output.logits[0, -1]          # (vocab_size,)
        log_probs = torch.log_softmax(logits, dim=-1)
        return log_probs[logits.argmax()]

    def output_to_representations(
        self,
        layer_representations: Dict[Tuple[str, str, str], torch.Tensor],
        stimuli_coords: dict,
    ) -> NeuroidAssembly:
        """Build a NeuroidAssembly from GradCAM saliency vectors.

        For each recorded layer the saliency is computed as the element-wise
        product of the hidden-state tensor and its gradient w.r.t. the scalar
        S, evaluated at the last token position::

            saliency[:, -1, :] = h[:, -1, :] * (dS/dh)[:, -1, :]

        The resulting array has shape ``(1, hidden_dim)`` per layer, identical
        to what :meth:`HuggingfaceSubject.output_to_representations` produces
        for raw activations, so all downstream metrics work unchanged.

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
                indicates the layer's output is detached from the computation
                graph (e.g., a stop-gradient operation inside the model).
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
            # h: (batch=1, seq_len, hidden_dim) → saliency at last token → (hidden_dim,)
            saliency = (tensor * tensor.grad)[:, -1:, :].squeeze(0).detach().cpu().numpy()
            saliency_parts.append(saliency)

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
