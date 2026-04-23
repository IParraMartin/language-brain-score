from brainscore_language import model_registry
from brainscore_language import ArtificialSubject
from brainscore_language.model_helpers.huggingface import HuggingfaceSubject
from brainscore_language.model_helpers.saliency import SaliencyHuggingfaceSubject

# layer assignment based on choosing the maximally scoring layer on Pereira2018-encoding from
# https://github.com/mschrimpf/neural-nlp/blob/master/precomputed-scores.csv

model_registry['openai-gpt'] = lambda: HuggingfaceSubject(model_id='openai-gpt', region_layer_mapping={
    ArtificialSubject.RecordingTarget.language_system: 'transformer.h.11'})

model_registry['distilgpt2'] = lambda: HuggingfaceSubject(model_id='distilgpt2', region_layer_mapping={
    ArtificialSubject.RecordingTarget.language_system: 'transformer.h.5'})

model_registry['gpt2'] = lambda: HuggingfaceSubject(model_id='gpt2', region_layer_mapping={
    ArtificialSubject.RecordingTarget.language_system: 'transformer.h.11'})

model_registry['gpt2-medium'] = lambda: HuggingfaceSubject(model_id='gpt2-medium', region_layer_mapping={
    ArtificialSubject.RecordingTarget.language_system: 'transformer.h.22'})

model_registry['gpt2-large'] = lambda: HuggingfaceSubject(model_id='gpt2-large', region_layer_mapping={
    ArtificialSubject.RecordingTarget.language_system: 'transformer.h.33'})

model_registry['gpt2-xl'] = lambda: HuggingfaceSubject(model_id='gpt2-xl', region_layer_mapping={
    ArtificialSubject.RecordingTarget.language_system: 'transformer.h.43'})

model_registry['gpt-neo-125m'] = lambda: HuggingfaceSubject(model_id='EleutherAI/gpt-neo-125m', region_layer_mapping={
    ArtificialSubject.RecordingTarget.language_system: 'transformer.h.11'})

model_registry['gpt-neo-2.7B'] = lambda: HuggingfaceSubject(model_id='EleutherAI/gpt-neo-2.7B', region_layer_mapping={
    ArtificialSubject.RecordingTarget.language_system: 'transformer.h.31'})

model_registry['gpt-neo-1.3B'] = lambda: HuggingfaceSubject(model_id='EleutherAI/gpt-neo-1.3B', region_layer_mapping={
    ArtificialSubject.RecordingTarget.language_system: 'transformer.h.18'})

model_registry['gpt2-saliency'] = lambda: SaliencyHuggingfaceSubject(model_id='gpt2', region_layer_mapping={
    ArtificialSubject.RecordingTarget.language_system: 'transformer.h.11'})

model_registry['gpt2-medium-saliency'] = lambda: SaliencyHuggingfaceSubject(model_id='gpt2-medium', region_layer_mapping={
    ArtificialSubject.RecordingTarget.language_system: 'transformer.h.22'})

model_registry['gpt2-xl-saliency'] = lambda: SaliencyHuggingfaceSubject(model_id='gpt2-xl', region_layer_mapping={
    ArtificialSubject.RecordingTarget.language_system: 'transformer.h.43'})

# --- Disentanglement variants (layer 11, last-token, next-token scalar) ---
# These isolate the contribution of each component of the GradCAM product.

# Raw gradient dS/dh: causal sensitivity irrespective of activation magnitude.
model_registry['gpt2-gradient'] = lambda: SaliencyHuggingfaceSubject(
    model_id='gpt2', mode='gradient',
    region_layer_mapping={ArtificialSubject.RecordingTarget.language_system: 'transformer.h.11'})

# Unsigned gradient |dS/dh|: influence magnitude without sign information.
model_registry['gpt2-abs-gradient'] = lambda: SaliencyHuggingfaceSubject(
    model_id='gpt2', mode='abs_gradient',
    region_layer_mapping={ArtificialSubject.RecordingTarget.language_system: 'transformer.h.11'})

# --- Token-pooling variant ---
# GradCAM averaged across all token positions instead of only the last token.
# Tests whether the brain's sentence-level response is better captured by a
# whole-sequence mean than by the final-position representation.
model_registry['gpt2-mean-saliency'] = lambda: SaliencyHuggingfaceSubject(
    model_id='gpt2', token_pool='mean',
    region_layer_mapping={ArtificialSubject.RecordingTarget.language_system: 'transformer.h.11'})

# --- Scalar variant ---
# GradCAM with sentence-level surprisal as the scalar S.
# The gradient flows back through every token position rather than only the last,
# providing a richer, sentence-wide signal.
model_registry['gpt2-surprisal-saliency'] = lambda: SaliencyHuggingfaceSubject(
    model_id='gpt2', scalar_mode='sentence_surprisal',
    region_layer_mapping={ArtificialSubject.RecordingTarget.language_system: 'transformer.h.11'})

# --- Layer sweep ---
# One GradCAM model per GPT-2 layer (0–11).  Allows identifying which layer's
# saliency vectors best predict brain activity, independently of the layer chosen
# for raw activations.
for _layer_idx in range(12):
    model_registry[f'gpt2-saliency-layer{_layer_idx}'] = (
        lambda i=_layer_idx: SaliencyHuggingfaceSubject(
            model_id='gpt2',
            region_layer_mapping={
                ArtificialSubject.RecordingTarget.language_system: f'transformer.h.{i}'
            }
        )
    )
