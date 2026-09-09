# OCR training and safe checkpoints

## Benchmark selection boundary

The frozen synthetic recognition benchmark invokes this same trainer without a second training
stack. It selects the best epoch solely by minimum validation CTC loss, then evaluates the held-
out test partition after training and checkpoint selection are complete. Test labels and
predictions never affect model selection or configuration.

Benchmark-generated line images and the best safetensors checkpoint live only in an explicit
temporary work directory. Compact safe result JSON is retained, while the checkpoint and bulk
images are removed and no canonical model weights are committed.

Phase 12B adds a distinct `recognition-synthetic-v2` harness without changing the shared trainer
or making the original benchmark command unexpectedly expensive. A development-only study first
showed that the unchanged CRNN can escape early blank collapse: a validation-loss-selected
16-line checkpoint reached train CER 0.0625 with 5/16 exact training lines after 288 updates.
At matched 120-update learning-rate controls, 0.001 outperformed 0.0005 and 0.0002.

The V2 protocol therefore seeds before constructing the model—as this document's reproducibility
contract already requires—then uses the existing AdamW trainer for at most 30 epochs / 240 steps,
batch 16, learning rate 0.001, weight decay 0.0001, gradient clipping 5.0, CPU, zero workers,
patience 8, and `min_delta=0`. Minimum validation CTC loss remains the only checkpoint-selection
criterion. The selected checkpoint is evaluated on the frozen 16-line test split exactly once,
then removed with the generated images. No scheduler, curriculum, classifier-bias trick,
architecture change, trainer change, or metric change was introduced.

Full reproduction is an explicit/manual operation from a checkout in which V2 result artifacts
do not yet exist:

```console
python scripts/run_recognition_benchmark_v2.py run --work-directory ../empty-v2-work
```

The work directory must already exist and be empty. The checked-in result was produced from the
built wheel in a fresh CPython 3.11.9 environment with CPU PyTorch 2.14.0 and took about 29.5
minutes for training plus 2.2 seconds for test inference and metrics. Ordinary validation uses
the script's `verify` mode and never retrains or re-infers.

Phase 10 provides a Python API for training the current CNN-BiLSTM-CTC recognizer from caller
supplied, canonical line-image manifests. It proves the engineering path; it does not provide a
trained public model or evidence of OCR quality.

## Required data

Training and validation inputs are `DatasetSample` records read through the existing strict JSONL
manifest API. Both partitions use one `Vocabulary`, normally built only from the training split,
and an explicit dataset root. Images must be root-contained PNG or JPEG files. Labels must already
satisfy `nfc-v1`; training never normalizes, substitutes, skips, or truncates them. The vocabulary
has no UNK class.

Before creating an optimizer or output directory, `train_model` loads every image once and checks
that both partitions are nonempty, sample IDs and document groups do not overlap, prepared image
content has no conflicting labels, every character is represented, normalized widths are valid,
and the exact Phase 9 width formula leaves enough timesteps for CTC including adjacent repeats.
This up-front pass costs one decode/normalization per image; each epoch then loads its samples once.

`OCRLineDataset` reuses the Phase 7 safe image loader and Phase 9 `prepare_line_image`. Each item is
float32 `[1,64,W]` in `[-1,+1]`, carries its valid width and sample ID, and contains a CPU int64
target encoded by `Vocabulary`. `collate_ocr_batch` rounds the largest width in a batch up to a
multiple of four, right-pads only to that width with white `+1`, preserves every valid width, and
concatenates targets with an explicit target-length vector. It never pads every batch to 2048.

## Training behavior

`TrainingConfig` deliberately has a small fixed-rate surface. Defaults are seed 1337, 20 epochs,
batch size 16, AdamW learning rate `5e-4`, betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay
`1e-4`, no AMSGrad/foreach/fused path, gradient norm ceiling 5.0, zero
workers, CPU, patience 5, `min_delta=0`, maximum image width 2048, and checkpoint name `best`.
The lower-than-`1e-3` starting rate is a conservative choice for the recurrent model and is not
copied from historical training. There is no scheduler, AMP, gradient accumulation, or framework.

Training shuffles with an explicitly seeded CPU generator, does not drop the final batch, and
seeds Python, NumPy, PyTorch, CUDA (when explicitly requested), and DataLoader workers. Validation
does not shuffle. These controls support reproducible reruns within a recorded environment; they
do not promise bit identity across PyTorch releases, platforms, or GPU implementations. To make
initial weights reproducible, call `set_training_seed` before constructing `CRNNRecognizer`.

The trainer rejects unavailable configured CUDA devices rather than falling back. It computes a
mean CTC loss for each batch, rejects nonfinite losses and gradients, clips gradients after
backpropagation, and takes one AdamW step per batch. Epoch loss is sample weighted:
`sum(batch_mean_loss * batch_size) / total_samples`, avoiding last-batch bias. Validation uses
inference mode and changes no parameters.

An epoch improves only when `new_validation_loss < best_validation_loss - min_delta`. Ties do not
improve. Patience counts completed validation epochs without sufficient improvement, and training
stops immediately after the counter reaches patience. The first finite validation loss becomes
best. History records epoch, train loss, validation loss, fixed learning rate, improvement flag,
best epoch/loss, early-stop state, optimization-step count, safe checkpoint name, and logical
dataset fingerprints. It contains no timestamps, document text, or absolute paths.

## Python API

```python
from urdu_document_ocr import (
    CRNNRecognizer,
    TrainingConfig,
    load_vocabulary,
    read_manifest,
    train_model,
)

training = read_manifest("splits/train.jsonl")
validation = read_manifest("splits/validation.jsonl")
vocabulary = load_vocabulary("vocabulary.json")
model = CRNNRecognizer(vocabulary)
result = train_model(
    model,
    training,
    validation,
    vocabulary,
    dataset_root="dataset",
    output_directory="run-output",
    config=TrainingConfig(device="cpu"),
)
```

The caller supplies a nonexistent or empty output directory. Existing unrelated contents fail
closed. The default output is:

```text
best/
  metadata.json
  model.safetensors
  vocabulary.json
```

Weights are contiguous CPU tensors saved through safetensors. Metadata is canonical strict JSON
binding schema/package/architecture identity, recognizer and training configurations, model and
vocabulary fingerprints, class/blank semantics, epoch and validation loss, dataset identities,
dependency versions, the vocabulary-file hash, the weight-file hash, and a metadata fingerprint.
Saving assembles and syncs a sibling temporary directory before an atomic same-parent rename.
Overwrite is rejected unless explicitly requested; replacement first moves a previously valid
checkpoint aside and restores it if installation fails.

Loading accepts no pickle and never invokes `torch.load`. It requires the exact three files,
rejects duplicate JSON keys and nonstandard numbers, verifies every hash/fingerprint/configuration,
checks tensor names, shapes, dtypes, and finite values before mutation, then performs strict state
loading. No optimizer state is serialized. Loading model weights can initialize another run, but
this is not exact training resume.

Phase 11 consumes this exact checkpoint without defining another format. `load_recognizer`
constructs the configuration- and vocabulary-bound CRNN through the safe loader, validates the
complete checkpoint, moves it to an explicit CPU or available CUDA device, selects evaluation
mode, and retains it for repeated inference. See `inference.md` for lifecycle, batching, document
assembly, and serialization behavior.

The checked-in synthetic fixtures support only a temporary CPU engineering smoke test. Their tiny
size and synthetic origin make them unsuitable for a quality claim. CER/WER, public benchmarking,
and canonical trained weights remain later-phase work. Document-level inference is implemented,
but it does not establish recognition quality.
