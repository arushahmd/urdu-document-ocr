# CNN-BiLSTM-CTC recognizer

Phase 9 provides a real trainable PyTorch recognizer core. It is initialized from scratch and is
currently untrained: this document describes tensor mechanics, not OCR accuracy. CNN-BiLSTM-CTC
is a standard model family; the source in this repository is a current-original implementation of
the frozen public architecture.

## Shape and architecture contract

```text
[B,1,64,W]
   ↓
CNN
   ↓
[B,384,4,T]
   ↓ mean(H)
[B,T,384]
   ↓
2× BiLSTM
   ↓
[B,T,512]
   ↓ Linear
[B,T,C]
   ↓
CTC
```

Each of four CNN blocks contains two bias-free 3x3 convolutions with padding 1. Every convolution
is followed by GroupNorm and SiLU. Channels progress `1 → 64 → 128 → 256 → 384`; GroupNorm uses
8, 16, 32, and 32 groups by block. Pool sizes are `(2,2)`, `(2,2)`, `(2,1)`, and `(2,1)` in floor
mode. Height therefore becomes 4 and width is reduced only by the first two pools.

The sole width-to-time calculation is:

```text
T(W) = floor(floor(W / 2) / 2)
```

It gives effective height stride 16, width stride 4, and minimum technical input width 4. The
feature height is mean-reduced, not flattened, and the width axis becomes a `[B,T,384]` sequence.
A two-layer bidirectional LSTM uses hidden size 256 per direction and dropout 0.2 between layers,
producing 512 features per valid timestep. `Linear(512, len(vocabulary.characters) + 1)` emits raw
logits. Softmax is deliberately absent from the model.

## Model input preparation

`prepare_line_image` accepts one nonempty grayscale NumPy `uint8[height,width]` image. It resizes
to height 64 with OpenCV while preserving aspect ratio; shrink operations use area interpolation
and enlargements use cubic interpolation. Resized width uses exact integer half-up rounding:
`floor(original_width * 64 / original_height + 0.5)`.

The function rejects normalized widths below 4 or above the configured maximum (2048 by default).
It never squeezes, crops, or truncates a long line. Pixels map to float32 with `x / 127.5 - 1`, so
black 0 maps to -1 and white 255 maps to +1. The result is `[1,64,W]` plus integer valid width.
Batch construction belongs to Phase 10; callers currently right-pad with value +1 and retain each
valid width explicitly.

`CRNNRecognizer.forward` accepts finite float32 `[B,1,64,W_pad]` values in `[-1,+1]` and a CPU
int64 `[B]` valid-width vector. It does not infer width from white pixels. Each item is cropped to
its declared valid width before the CNN, preventing white batch padding from changing boundary
features. Resulting unsorted sequences use `pack_padded_sequence(enforce_sorted=False)` through
the BiLSTM and are restored to padded time width. The output is `RecognizerOutput` containing raw
`[B,T,C]` logits and CPU int64 valid output lengths. The implementation does not choose a device;
callers may move the model and image tensor to a supported device while keeping packed lengths on
CPU.

## Vocabulary and CTC

The existing `Vocabulary` is the only character map. Class 0 is CTC blank and characters occupy
indexes `1..N`. There is no UNK, PAD, BOS, or EOS output. Model fingerprints bind the frozen
architecture identifier, recognizer configuration fingerprint, blank semantics, and vocabulary
fingerprint for future checkpoint metadata.

`compute_ctc_loss` validates raw `[B,T,C]` logits, concatenated CPU int64 targets, positive CPU
int64 target lengths, and positive CPU int64 input lengths. It creates log probabilities with
`log_softmax`, transposes them to `[T,B,C]`, then uses `CTCLoss(blank=0, reduction="mean",
zero_infinity=True)`. Blank and out-of-range target indexes fail instead of being substituted.

CTC feasibility requires more than `target_length <= input_length`: each adjacent repeated target
label needs a separating timestep. `minimum_ctc_timesteps` returns target length plus the number
of adjacent equal pairs. `validate_ctc_alignment` reports every impossible sample index, and
`compute_ctc_loss` raises `CTCAlignmentError` before loss evaluation if any are impossible;
`zero_infinity=True` is not used to conceal invalid data.

## Greedy decoding

`greedy_ctc_decode` first truncates each argmax path to its explicit valid output length. It then
collapses adjacent repeated class predictions, removes blank zeros, and finally maps remaining
indexes through `Vocabulary`. That order preserves `character, blank, character` as two emitted
characters. Results use `OCRPrediction` with text, character indexes, and sequence length. No
confidence is computed because per-timestep maxima are not a calibrated sequence confidence.

## Reproducibility and limitations

Seeded initialization and `model.eval()` are stable when repeated in the same tested environment;
PyTorch does not promise bit-identical results across all releases or platforms. The implementation
does not use AMP, `torch.compile`, TorchScript, ONNX, pretrained weights, downloads, or automatic
device selection.

Phase 9 does not include a Dataset/DataLoader, collator, trainer, optimizer/epoch orchestration,
checkpoint persistence, beam search, language model, document inference, CER/WER, or trained public
weights. Greedy output from a random model has no recognition meaning. These boundaries keep the
first neural layer small enough to audit and prepare Phase 10 training without overstating current
capability.
