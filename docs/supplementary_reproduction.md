# Supplementary controls and analyses

Run these commands from the repository root after installing the recorded environment. Set `PYTHONPATH=src` if the repository is not installed as an editable package. The GPU paths use one visible CUDA device (`cuda:0`). Set `CUDA_VISIBLE_DEVICES` before starting a process to select a physical device.

The controls retain the published numerical definitions:

| Route | Backbone and readout | Seeds | Training |
|---|---|---|---|
| T1 | Frozen pretrained backbone; terminal linear readout | 1701–1705 | 30 epochs, head LR 0.001 |
| T2 | Frozen pretrained backbone; terminal spatial readout | 1701–1703 | 30 epochs, head LR 0.0003 |
| T3 | Fresh LoRA plus terminal spatial readout | 1701–1703 | 10 epochs, then the fixed three-seed development rule may extend all seeds to 30 |
| T4 | Frozen pretrained backbone; projection-input spatial readout | 1701–1703 | 30 epochs, head LR 0.001 |
| B1 | UPerNet with ConvNeXt-Tiny initialized from ADE20K | 1701–1705 | 40 epochs, LR 0.0001 |

`configs/supplementary/recipes.json` records the settings. Every epoch uses all 4,821 training images and all 1,206 development images. Exact ties in development mIoU select the earlier epoch. The epoch 10, development-selected and last checkpoints remain distinct roles. T3 also retains the selected endpoint from epochs 1–10.

## Data and source models

Prepare the public partitions and taxonomy with the shared builder:

```bash
python scripts/original/prepare_data.py \
  --data-root /path/to/CropAndWeed/WACV2023 \
  --output-dir /path/to/manifests
```

The prepared `train.json` and `development.json` contain checked RGB and source-label paths. Primary and final input manifests omit labels; their separate `*.labels.json` files are used only for evaluation. The code maps source labels to background 0, crop 1, weed 2 and ignore 255. RGB uses bilinear resize to 512 × 512; source labels use nearest-neighbour resize before mapping. Training never reads the primary or final partitions.

Download the complete source model snapshots identified by `configs/supplementary/flux_snapshot.json` and `upernet_snapshot.json`. The loaders check every listed file's size and SHA256. These manifests contain file identities, not model weights. The original scientific text conditioning remains a required model input. Use the recorded dependency environment, including the exact Diffusers source distribution; a matching package version alone does not establish matching model code.

## Fixed feature caches

T1 and T2 consume terminal features with shape `[1,1024,128]` stored in BF16. T4 consumes actual projection-input features with shape `[1,1024,3072]` stored in FP32. No precision conversion is applied. The cache covers train and development only. T3 computes features online because its LoRA parameters change.

```bash
python scripts/supplementary/build_cache.py \
  --feature terminal --data-root /path/to/CropAndWeed/WACV2023 \
  --train-manifest /path/to/manifests/train.json \
  --dev-manifest /path/to/manifests/development.json \
  --model-root /path/to/flux-snapshot --output /path/to/cache-terminal
```

Repeat with `--feature projection-input` and another output directory for T4. Its raw feature payload is about 75.8 GB. The terminal payload is about 1.58 GB. The portable cache uses one checked tensor file per image; its storage layout differs from the original server archive. Cache readers bind image SHA256, model snapshot, prompt text/tensor identity, feature dtype and backend.

## Training and continuation

Example T2 seed 1701:

```bash
python scripts/supplementary/train.py --route T2 --seed 1701 \
  --data-root /path/to/CropAndWeed/WACV2023 \
  --train-manifest /path/to/manifests/train.json \
  --dev-manifest /path/to/manifests/development.json \
  --taxonomy /path/to/manifests/taxonomy.json \
  --cache /path/to/cache-terminal --output /path/to/runs/T2-1701
```

Run each prescribed seed independently. T1 uses the terminal cache; T4 uses the projection-input cache. For T3 replace `--cache` with `--model-root /path/to/flux-snapshot`. For B1 use `--model-root /path/to/upernet-snapshot`. T2 and T3 construct the same fresh per-seed spatial-head tensors; their initialization hashes are recorded.

Native heads use FP32 parameters and BF16 autocast. The frozen transformer, VAE and T3 LoRA use BF16. The VAE stays outside the transformer/head autocast context. UPerNet retains the four-image microbatch, two-microbatch accumulation and 3+2 tail. Its weighted loss uses one weighted-pixel denominator across the entire accumulation group.

The initial T3 invocation ends at 10 epochs. After all three seeds finish:

```bash
python scripts/supplementary/training_budget.py \
  --curves /path/to/runs/T3-1701/development.json \
           /path/to/runs/T3-1702/development.json \
           /path/to/runs/T3-1703/development.json \
  --output /path/to/t3-budget.json
```

The fixed rule compares the equal-seed mean of `best dev epochs 6–10 minus best dev epochs 1–5` with0.002 using strict `>`. If the result is 30, invoke each T3 run with the same inputs plus `--epochs 30 --resume /path/to/runs/T3-SEED/epoch_010.pt --extension-decision /path/to/t3-budget.json`. There is no further extension rule. The original paper's supplementary run satisfied this condition and completed 30 epochs for all three seeds.

Checkpoints include model, optimizer, scheduler, RNG, completed epoch and initialization identity. Resume requires the matching JSON sidecar and the same manifests/cache/model identity. A later checkpoint on the same T3 extension chain can also resume. An interrupted partial epoch is replayed from the preceding complete checkpoint; there is no automatic mid-update recovery or automatic retry. This interface preserves scientific state and replaces the original server queues and process accounting. It does not claim identical checkpoint serialization or GPU replay across different software/hardware.

## Fixed-endpoint predictions and unified primary comparisons

`endpoints.json` points to the actual epoch 10 and development-selected checkpoints. Evaluate every role in `configs/supplementary/endpoint_plan.json`:38 supplementary groups on the common 844-image primary split, plus the 20 original groups. The recorded epoch numbers in that plan describe the published run; a new training run applies the same development selection rule to its own curves.

```bash
python scripts/supplementary/predict.py --route T2 --seed 1701 \
  --system T2__epoch10 --checkpoint /path/to/runs/T2-1701/epoch_010.pt \
  --data-root /path/to/CropAndWeed/WACV2023 \
  --manifest /path/to/manifests/replication-confirmation.inputs.json \
  --model-root /path/to/flux-snapshot --output /path/to/predictions/T2-1701-epoch10
```

Use the exact `system` identifiers in the comparison configuration, and preserve each endpoint separately. Prediction indexes include a checkpoint checksum and image-level output checksums. Predictions are uint8 categorical maps; evaluation never rescales them.

```bash
python scripts/supplementary/analyze.py --split primary \
  --data-root /path/to/CropAndWeed/WACV2023 \
  --truth-manifest /path/to/manifests/replication-confirmation.labels.json \
  --taxonomy /path/to/manifests/taxonomy.json \
  --predictions /path/to/predictions/*/predictions.json \
  --comparisons configs/supplementary/unified_primary_comparisons.json \
  --output /path/to/primary-analysis
```

The 20 supplementary contrasts use paired, source-stratified whole-session resampling: 10,000 common PCG64 draws with seed 2026101901 and linear percentile 95% intervals. All prescribed seeds remain in each comparison. Undefined operands propagate to explicit NA intervals. These intervals describe session composition conditional on the fixed trained seeds and do not replace the original hypothesis-family results.

## Sensitivity and inference resources

The nine sensitivity configurations use the fixed 92-image panel and original epoch 10 checkpoints for seeds 1701–1703. S7 varies steps 1, 2, 4, 10. Original JNDA varies timestep 0.5, 0.75, 0.999, or uses the short/empty text condition at 0.999. This is not a full cross-product.

For sensitivity, use `--route Fresh_S7` or `--route Fresh_Arm_B`, `--legacy-output` pointing to that original method/seed output directory, `--panel-ids configs/supplementary/sensitivity_panel.txt` and the common primary input manifest. Set `--steps`, `--timestep` or `--prompt` as specified by the configuration, and set `--system` to its configuration name (`S7_steps1`, `JNDA_t050`, etc.). The original checkpoint directory contains `checkpoint_epoch_10/training_state.pt` and the required adapter. Analyze with `--split sensitivity --panel-ids configs/supplementary/sensitivity_panel.txt` and `sensitivity_comparisons.json`.

Resource measurement uses `predict.py --timing`, with the same checkpoint and source model arguments. Use `--panel-ids configs/supplementary/resource_main40.txt` for each main seed. Use `resource_extra20.txt` for two separate seed 1701 calls with `--panel-kind extra-1` and `extra-2`. Run these in an isolated GPU window without other GPU work or large concurrent I/O. The timer preloads RGB, performs 10 inspection warmups, resets peak-memory statistics, then records three full calls per image. Each call includes the model and CPU categorical output; model/prompt loading, disk access and warmups remain outside the timer. Timing does not read GT or save predictions.

`summarize_resources.py --cells ... --expected-seeds ... --output ...` reports median-of-three per image, median across 40 images per seed, then the median across prescribed seeds. Throughput is the number of main valid calls divided by their total seconds. The three common 20-image blocks form a descriptive range, not a confidence interval. The sensitivity resource matrix varies S7 steps only.

## Boundary and small-weed diagnostics

Add `--diagnostics --system-aliases configs/supplementary/original_system_aliases.json` to `analyze.py` and use `primary_diagnostic_comparisons.json` or `final_diagnostic_comparisons.json`. Primary has 20 original epoch 10 groups on 844 images; final has 25 groups on 1,163 images. Primary contrasts use Fresh_JNDA as the left operand. Final contrasts use S7_init_JNDA as the left operand. They remain separate analyses.

Boundary precision/recall/F1 use pixel radii 1, 2, 4, reliable eroded valid domains and same-component matching. Small/tiny weed patches are 8-connected GT components of at most 256/16 pixels; these are not biological instances or AP. Ratios are computed after pooling counts. `--case-comparison primary_diagnostic_03` or `final_diagnostic_04` applies the published fixed-seed diagnostic case selector. It records selection ranks and exclusions, not representative prevalence.

## Software verification

```bash
python scripts/supplementary/check_synthetic.py
```

This checks synthetic metrics, NA handling, shared session resampling and the training-budget predicate. The release was also checked with small CPU tensors against the frozen spatial-head definitions and cache identity failures. Real pretrained inference, CUDA training and fresh primary/final scoring were not rerun while preparing this release.
