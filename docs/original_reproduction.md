# Original four-system reproduction

The original CropAndWeed comparison uses five seeds (1701–1705), four systems, and ten training epochs. The retained training and prediction code is the corrected final implementation. Run commands from the repository root. Paths in the runtime JSON are relative to the current working directory or may be absolute paths on your own machine.

## Data and pretrained models

Obtain the official CropAndWeed WACV2023 files separately. The default layout is `images/<sample_id>.jpg` and `labelIds/CropAndWeed/<sample_id>.png`. The label PNGs must contain original source IDs, not the three-class remapping. The public CSV files define all 8,034 samples and the complete 101-row mapping to background/soil=0, crop=1, weed=2, ignore=255.

```bash
python scripts/original/prepare_data.py \
  --data-root data/CropAndWeed/WACV2023 \
  --output-dir data/manifests/cropandweed
```

Use `--image-dir` and `--label-dir` to name different directories under `--data-root`. Files are matched by exact sample ID. Missing files, duplicate sample matches, session overlap, changed allocation, and changed sample order are rejected. Preparation hashes files but does not decode label pixels or calculate metrics.

Outputs are `train.json` (4,821 rows), `development.json` (1,206), `replication-confirmation.inputs.json` (844), and `final-method-test.inputs.json` (1,163). The last two contain no label fields in their sample rows. Every partition has a separate `<partition>.labels.json`; its `label_encoding` is `source_ids`. All paths inside manifests are relative to the supplied data root. `taxonomy.json` supplies the source-to-three-class lookup, and `cost40.inputs.json` reproduces the registered 40-session timing selection. The same manifests are used by the supplemental training and evaluation entrypoints.

Obtain the full `black-forest-labs/FLUX.2-klein-base-4B` snapshot at revision `8c44a2fbef88fae175da65df054db0f901aa9747`, including its VAE, transformer, text encoder, tokenizer, and scheduler. Obtain the OpenMMLab SegFormer MiT-B2 ADE20K 512×512 checkpoint with SHA256 `cbd414ac055c8436207914ac3a8e08c69879e8ae9e70f1a57cf0f7e17f19025a`. The loader verifies this checkpoint and loads every compatible non-classifier tensor. Only the three-class classifier is newly initialized. Dataset and model licenses remain those of their respective owners.

Copy or edit `configs/original/runtime.example.json` to point to your data, manifests, and models. Install the repository's pinned environment, including its specified Diffusers source distribution. The implementation uses FLUX.2 pipeline internals, so substituting an arbitrary Diffusers release is not a supported exact reproduction.

```bash
python scripts/original/run.py check --config configs/original/runtime.example.json
python scripts/original/run.py train --help
```

`check` and `--check-config` inspect paths and package metadata without loading models or reading semantic masks. The package-version report is separate from file availability; inspect both before execution. Training and inference use one selected CUDA device and bfloat16.

## Training and checkpoints

The system names retain their implementation identifiers:

| System | Training |
| --- | --- |
| `Fresh_S7` | Rank-8 RGB-LoRA, masked latent flow-matching loss, learning rate 1e-4 |
| `Fresh_Arm_A` | Freeze the same-seed final S7 adapter and base; train a new 99,072-parameter native head |
| `Fresh_Arm_B` | Jointly train a fresh rank-8 LoRA and native head |
| `Fresh_SegFormer_B2` | Fine-tune the loaded encoder/decoder and new three-class classifier |

Every training cell is split into two separate invocations: `process_a` completes epochs 1–5 and `process_b` resumes for epochs 6–10. The latter restores the model, optimizer, scheduler, Python/NumPy/Torch CPU/CUDA random states, and data-order generator. Launch it as a new process.

```bash
python scripts/original/run.py train --system Fresh_S7 --seed 1701 --process process_a --device 0
python scripts/original/run.py train --system Fresh_S7 --seed 1701 --process process_b --device 0
python scripts/original/run.py train --system Fresh_Arm_A --seed 1701 --process process_a --device 0
python scripts/original/run.py train --system Fresh_Arm_A --seed 1701 --process process_b --device 0
python scripts/original/run.py train --system Fresh_Arm_B --seed 1701 --process process_a --device 0
python scripts/original/run.py train --system Fresh_Arm_B --seed 1701 --process process_b --device 0
python scripts/original/run.py train --system Fresh_SegFormer_B2 --seed 1701 --process process_a --device 0
python scripts/original/run.py train --system Fresh_SegFormer_B2 --seed 1701 --process process_b --device 0
```

Repeat for seeds 1702–1705. Add `--config <file>` and `--workspace <directory>` consistently if changing defaults. The default output is `runs/original/<system>/seed_<seed>`. Arm A locates `Fresh_S7/seed_<seed>/checkpoint_epoch_10` under the same workspace; `--paired-s7` may explicitly identify that checkpoint directory. Its system, seed, completed epoch, and adapter presence are checked.

The complete resumable checkpoint is `checkpoint_epoch_05/training_state.pt` or `checkpoint_epoch_10/training_state.pt`. `model_state` contains `lora` for S7, `head` for Arm A, `head` and `lora` for Arm B, and `model` for SegFormer. S7 and Arm B also write `pytorch_lora_weights.safetensors` at epoch 10. A native head alone is not a complete model; it also needs the fixed base and the appropriate adapter. Load only checkpoints you trust, because the original full training-state format uses PyTorch serialization with optimizer and RNG objects.

The retained defaults include full-image resizing to 512×512, bilinear RGB interpolation, nearest-neighbor label interpolation, fixed class weights, and no crop. FLUX/native training uses batch size one. SegFormer uses batch size four and gradient accumulation two. The original fixed model input text is part of the method and remains in the implementation. Hyperparameters, loss reduction, sample-order hashing, and final-epoch selection are retained in the source; the path wrapper does not select checkpoints or tune parameters.

## Prediction and statistics

After completing a cell, `develop` evaluates its final checkpoint on the development set and `predict` writes all 844 primary input predictions. Development results do not select a different checkpoint.

```bash
python scripts/original/run.py develop --system Fresh_Arm_B --seed 1701 --device 0
python scripts/original/run.py predict --system Fresh_Arm_B --seed 1701 --device 0
```

Run `predict` for all four systems and all five seeds, then explicitly request the paired quality evaluation:

```bash
python scripts/original/run.py evaluate
```

The evaluator checks the full 20×844 prediction inventory and file digests, then reads the separate primary-label manifest. It pools confusion counts before computing each seed's mIoU and performs the original 10,000-draw source-stratified recording-session bootstrap with paired resampling across systems and fixed seeds. It retains H1–H3 margins and decision rules. This entrypoint does not evaluate the final-method-test partition.

For a separately requested final-partition prediction, add `--split final-method-test` to `predict`. The same final checkpoint and predictor are used on all 1,163 input-only images. Primary predictions also write the interoperable `predictions.json` index; final predictions write `final_predictions.json`. Both bind every sample to its source/session and checkpoint digest for the supplemental analysis CLI. These commands cover the four system identifiers listed above; an additional S7-initialized historical diagnostic route requires its own verified implementation and checkpoint binding.

For the original latency comparison, execute `timing` on an otherwise idle GPU for every required system/seed cell, keeping the same physical GPU throughout. The implementation preserves the 40 samples, ten warmups, three repeats, synchronization, and timing boundary. It times the original callable, including its preprocessing and output transfers; model loading, disk image reads, and one-time text embedding are outside the measured interval.

```bash
python scripts/original/run.py timing --system Fresh_S7 --seed 1701 --device 0
python scripts/original/run.py timing --system Fresh_Arm_B --seed 1701 --device 0
# Repeat both systems for all five seeds, then:
python scripts/original/run.py h4
```

`h4` reproduces the paired source-stratified session bootstrap and the geometric aggregation of per-seed median-latency ratios. Timing commands for Arm A and SegFormer are also available for the complete resource table. The original H1–H4 family remains distinct from later supplementary analyses.

## Source identity and portability

| Retained source | Original SHA256 | Public adaptation |
| --- | --- | --- |
| `s8r_ex01_training.py` | `a9d9edaa050ba816d89f869d1c9c9d5002a4dd2762f74174fc72b3d9168ef2d7` | Scientific definitions preserved; module description updated for the public entrypoint |
| `s8r_ex01_execution.py` | `d3c07eafc450b6455bbecbbcacce4b54edf30c3f273dcf33dc3afb5602991bbe` | Exact selected RNG, split-guard, inventory and statistical definitions |
| `s6_supervised.py` | See `original_source_provenance.json` | Exact SegFormer model and checkpoint-mapping definitions |

`portable_original.py` resolves caller-owned asset paths, prepares compatibility metadata, and invokes the retained functions. It replaces historical machine and authorization-file requirements with explicit CLI execution. The package initializer and LoRA helper no longer import unrelated archival or transfer modules. `original_source_provenance.json` lists source identities and retained definitions. Validation of this public assembly is limited to static source equivalence and CPU checks unless separately documented; it does not claim a new end-to-end GPU run.

## PhenoBench discovery/native arms

The PhenoBench native implementation is separate from the CropAndWeed replication. It uses all 1,407 official training images, seeds 20260723/20260724/20260725, 7,035-step and 14,070-step checkpoint boundaries, and the released frozen 500-image public-validation selection. Its label mapping is `0→0, 1→1, 2→2, 3→1, 4→2`. Its prompt text and VAE normalization order are retained from the PhenoBench implementation, including float32 normalization of VAE features before casting back to BF16. The wrapper does not substitute the CropAndWeed preprocessing or trainer.

Edit `configs/original/phenobench.example.json`. The official dataset layout is `train/images`, `train/semantics`, and `val/images`, with matching PNG filenames. The input manifests retain full filenames as sample IDs. `prepare` hashes the training image/label files and selected validation images without decoding labels. `cache` subsequently reads only training semantic pixels to build the shared VAE-token cache and fixed prompt embeddings. It uses the retained RGB target mapping and nearest-neighbor label transformation; no validation labels are read by these commands.

```bash
python scripts/original/phenobench.py check --arm B
python scripts/original/phenobench.py prepare
python scripts/original/phenobench.py cache --device 0
python scripts/original/phenobench.py train --arm B --seed 20260723 --process process_a --device 0
python scripts/original/phenobench.py train --arm B --seed 20260723 --process process_b --device 0
python scripts/original/phenobench.py predict --arm B --seed 20260723 --device 0
```

Repeat the two training invocations and prediction for the other two seeds. Native checkpoints contain `trainable.safetensors`, optimizer/scheduler/RNG states, and integrity metadata under `arm_B/seed_<seed>/checkpoint-7035` and `checkpoint-14070`. The original optimizer, class weights, per-seed sample order, trainable whitelist, full resume checks, and resource ceilings remain in `s8_native_training.py`. The shared cache binds 1,407 BF16 tensors of shape 1,024×128 and their 512×512 class masks. These cache files and weights are created by the user's own run and are not included in the code package.

Arm A additionally requires the **same-seed PhenoBench S7 RGB-LoRA adapter**. Two explicit identities are supported. `phenobench.example.json` uses `kind: published_frozen` and accepts only the historical SHA256 fixed for each seed. `phenobench.reproduction.example.json` uses `kind: public_s7_reproduction` and selects the complete `adapter_binding.json` emitted by a new run of the public S7 CLI. A free-form replacement path/SHA is not accepted as a new reproduction. Do not substitute a CropAndWeed S7 adapter.

The new-run manifest binds the adapter's actual bytes, seed, original recipe, verified 19-file FLUX snapshot, official trainer, public bridge sources, training-input identity, both completed process halves, and all 14,070 optimizer steps. The final adapter is compared tensor by tensor with checkpoint 14,070 after the official BF16 export conversion, so harmless serialization or export-dtype differences do not require identical checkpoint/export file hashes. New runs are explicitly recorded as reproductions, never as the original published checkpoint.

```bash
python scripts/original/phenobench.py check --arm A
python scripts/original/phenobench.py cache-arm-a --seed 20260723 --device 0
python scripts/original/phenobench.py train --arm A --seed 20260723 --process process_a --device 0
python scripts/original/phenobench.py train --arm A --seed 20260723 --process process_b --device 0
python scripts/original/phenobench.py predict --arm A --seed 20260723 --device 0
```

Repeat for the other two seeds. Arm A caches frozen native transformer outputs for each matched adapter and trains only the native head. Arm B retains fresh LoRA initialization and does not consume S7 weights. Both native arms bind the same seed, training manifest, shared VAE cache, base-model identity, and prompt cache; the second completed arm and any two-arm evaluation must match these bindings. Use the same reproduction workspace and configuration throughout both process halves. Predictions use the exact final native checkpoint and fixed validation input order. Run `python scripts/original/phenobench.py evaluate --arm A` (or `--arm B`) to read the separate official validation semantic masks and calculate pooled per-seed confusion matrices, class IoUs, mIoU, and the mean over all three seeds. `--seed` limits this descriptive evaluation to one seed. When both arms' prediction inventories are present, their pairing identities are checked. This semantic evaluator does not reconstruct historical discovery gate decisions or confidence intervals. The PhenoBench commands have been inspected and checked on CPU where possible; their GPU execution has not been rerun for this release.

### Rebuilding the PhenoBench S7 prerequisite

`scripts/original/phenobench_s7.py` supplies the original paired-data transformation, fixed sample order, bounded batch sampler, argument construction, and fresh-process checkpoint checks around the official Diffusers trainer. Its exact 77,918-byte source is included at `third_party/diffusers_examples/train_dreambooth_lora_flux2_klein_img2img.py` with SHA256 `6486a9d7b56264a047339c3e8291d0d55cae51266bb2c3c5d8d533872330e31e` and its Apache 2.0 license. The bridge checks that identity before importing it. `--official-trainer` may point to another local copy with the same digest.

```bash
python scripts/original/phenobench_s7.py check
python scripts/original/phenobench_s7.py prepare
python scripts/original/phenobench_s7.py train --seed 20260723 --process process_a --device 0
python scripts/original/phenobench_s7.py train --seed 20260723 --process process_b --device 0
```

The full-data route uses 1,407 training pairs, 10 epochs, rank 8, learning rate 1e-4, guidance 1.0, latent caching, and model offload, as recorded in its recipe. Repeat for the remaining two seeds. Each final `runs/phenobench/s7/seed_<seed>/adapter_binding.json` is already referenced by `phenobench.reproduction.example.json`; use `--config configs/original/phenobench.reproduction.example.json` for the new native reproduction, including Arm A cache construction. Keep the S7 run records, optimizer-step logs, and midpoint/final checkpoints with that manifest so its completion identity can be verified. S7 preparation records its source native training-manifest digest, and training rejects stale paired inputs before loading a model. The official parser is configured with `report_to=None`, AdamW without 8-bit mode, no Hub push, and no final inference. This path does not require TensorBoard, WandB, Datasets, BitsAndBytes, or ProdigyOpt for those disabled branches. No model-card upload or remote tracking is performed by the bridge.
