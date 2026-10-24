# S7-initialized JNDA

This is a separately trained route. It starts from the same fresh base-model, task-LoRA structure and native linear-head construction as Fresh JNDA, then overwrites only the LoRA tensor values with the **same-seed S7 RGB-mask model's fixed epoch10 adapter**. It is not Fresh JNDA under another name and does not use the spatial head from the supplementary T3 control.

The native head is `Linear(128,768,bias=True)` with99,072 trainable parameters. The frozen base model is FLUX.2-klein-base-4B at revision `8c44a2fbef88fae175da65df054db0f901aa9747`. Five seeds1701–1705 each receive10 categorical-training epochs,48,210 updates, on the same4,821 training images. The endpoint is fixed epoch10; development performance does not select a different checkpoint.

The implementation preserves the distinct route from upstream commit `6d5a36a304b38b314d7b7076b2e12135723afcb3` in an independent module. The original four-route implementation remains separate. Portable wrappers replace old machine, repository and protocol-receipt locations with the configured local asset paths and checksummed run identities.

## Prerequisites

Use the original environment and `configs/original/runtime.example.json`, edited with your local paths. Prepare the manifests using `scripts/original/prepare_data.py` and finish all same-seed S7 runs with `scripts/original/run.py`. Their output layout is:

```text
runs/original/Fresh_S7/seed_1701/checkpoint_epoch_10/
    training_state.pt
    pytorch_lora_weights.safetensors
    formal_checkpoint_receipt.json
```

The other seeds follow the same layout. Source S7 checkpoints must report the correct route, seed, epoch10 and48,210 updates, and match their state/adapter checksums. Newly reproduced checkpoints are supported; they need not have identical serialization bytes to the paper's original files. The base-model snapshot is checked against the published file manifest.

```bash
python scripts/original/s7_initialized.py check \
  --config configs/original/runtime.example.json
```

This inspects configuration and input availability without loading models or running CUDA.

## Two-stage training

Run each seed in two separate processes, as in the frozen implementation. The first invocation completes epochs1–5 and24,105 updates:

```bash
python scripts/original/s7_initialized.py train \
  --config configs/original/runtime.example.json \
  --recipe configs/original/s7_initialized.json \
  --source-workspace runs/original \
  --workspace runs/s7_initialized \
  --seed 1701 --process process_a --device 0
```

The second invocation restores that route's own model, optimizer, constant scheduler, RNG and step count, then finishes epochs6–10:

```bash
python scripts/original/s7_initialized.py train \
  --config configs/original/runtime.example.json \
  --recipe configs/original/s7_initialized.json \
  --source-workspace runs/original \
  --workspace runs/s7_initialized \
  --seed 1701 --process process_b --device 0
```

Repeat for seeds1702–1705. Existing outputs are not overwritten. The midpoint must come from the same recipe, prepared manifests, source adapter and model snapshot. There is no automatic retry or additional training budget.

Initialization follows this order:

1. Seed RNG and create the fresh base model and task-LoRA structure.
2. Create the same fresh native linear head.
3. Capture RNG, strictly load the paired S7 LoRA values, and restore RNG.
4. Check that base-model parameters, VAE, head and RNG are unchanged, and that LoRA differs from fresh initialization.
5. Construct a new categorical optimizer and scheduler.

No S7 optimizer, scheduler, RNG or step state is transferred. Adapter loading removes exactly one `transformer.` prefix and rejects missing, unexpected, duplicated or incompatible keys. Shapes and dtypes must match exactly. LoRA rank8, alpha8, effective scaling1 and target-module sets are verified, followed by exact tensor equality after loading.

The optimizer has three groups: head weight (LR0.001, weight decay0.01), head bias (LR0.001, no decay), and all task-LoRA parameters (LR0.0001, weight decay0.01). AdamW uses betas0.9/0.999 and epsilon1e-8, constant scheduling, gradient norm clipping1, batch size1 and no accumulation or random augmentation. Weighted cross-entropy uses class weights `[0.25,1.3431242219633113,1.4751680850839344]` and ignore255. Model/LoRA computation uses the original BF16 path; the head is FP32.

The output includes the step0 tensor-identity audit, training journals, the midpoint state, and the final checkpoint plus adapter and formal checkpoint receipt. These are scientific run records generated locally; no private execution-chat material is required.

## Final image-only prediction

After epoch10:

```bash
python scripts/original/s7_initialized.py predict \
  --config configs/original/runtime.example.json \
  --workspace runs/s7_initialized --seed 1701 --device 0
```

The input is the separately prepared `final-method-test.inputs.json`:1,163 RGB images from140 sessions, with no label fields. The predictor reconstructs a fresh LoRA structure and head, then loads the **trained S7-initialized head and LoRA**. It never replaces the trained adapter with the source S7 adapter at inference.

RGB is bilinearly resized to512×512. Output is an unresized512×512 uint8 categorical map with values0/1/2. The original final-inference context is preserved: `inference_mode` and BF16 autocast cover the packed VAE input, native transformer and head. Training deliberately uses a different scope: VAE packing is outside the transformer/head autocast. These contexts must not be unified with the supplementary control implementation.

Each seed writes `final_predictions.json` with public system name `S7_init_JNDA`, checkpoint checksum and image-level checksums. This is compatible with `scripts/supplementary/analyze.py` and the final diagnostic comparisons, which use S7-initialized JNDA as the left operand. Those diagnostics use their own supplementary bootstrap seed2026101901. The original fifth-route statistical recipe records bootstrap seed2026082301 and practical margin0.005; the two analyses must retain their separate identities.

The publication preparation verified source identities, strict-loader behavior and portable metadata contracts using static checks and synthetic CPU tensors. It did not rerun this training, load the real pretrained model or score real final labels.
