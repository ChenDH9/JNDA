# JNDA

JNDA adapts a generative vision backbone for direct crop and weed semantic segmentation. A task LoRA and a native categorical head are trained while the pretrained base remains frozen. This repository provides the implementation and reproduction configuration for the original comparisons and the supplementary controls.

Associated study: **Joint-native adaptation of a generative vision backbone for direct crop-weed segmentation in intelligent phytoprotection**, by Dinghao Chen, Haoji Guo, Wen Liu and Li Li.

## Installation

The recorded experiment environment is Linux x86_64, Python 3.11.15, PyTorch 2.9.1 with CUDA 12.8 and TorchVision 0.24.1. Use a separate environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install pip==26.1.2 setuptools==83.0.0 wheel==0.47.0
python -m pip install -r environments/requirements-linux-cu128.txt
python -m pip install --no-deps 'https://github.com/ChenDH9/JNDA/releases/download/v1.0.0/diffusers-0.40.0.dev0-py3-none-any.whl#sha256=e4ba4b112659d61d8254cc03e08caf29e7d59962310b65000d145d6c59ae1d81'
python -m pip install -e .
```

The separately distributed Diffusers wheel preserves the actual runtime source used by the experiments, including its upstream license. See [the dependency description](third_party/README.md) for its checksum. A newer package with the same development-version label is not an exact replacement.

## Data and models

Download the original [CropAndWeed](https://github.com/cropandweed/cropandweed-dataset) and [PhenoBench](https://www.phenobench.org/dataset.html) datasets under their respective terms. Dataset images and semantic labels are not included here. The `data/` directory contains the fixed sample assignments, source-label mapping and PhenoBench validation selection used in the study.

The following commands explicitly download and verify the required pretrained model files:

```bash
python scripts/model_assets.py --model flux
python scripts/model_assets.py --model segformer
python scripts/model_assets.py --model upernet
```

`configs/model_assets.json` fixes the model repositories, revisions and file SHA256 values. Use `--verify-only` to check an existing download without network access. The full FLUX snapshot is about 16 GB; the code does not fetch it as a side effect of importing a module or displaying CLI help.

Study-trained adapters, heads and complete training-state checkpoints are not part of this code release. Training commands produce these artifacts; the existing study checkpoints are available from the corresponding authors on reasonable request, subject to applicable third-party terms.

## Reproducing the experiments

- [Original four-system comparisons](docs/original_reproduction.md): data preparation, five-seed training, development and fixed-checkpoint prediction, primary statistics and timing.
- [Supplementary controls and analyses](docs/supplementary_reproduction.md): linear and spatial readouts, joint adaptation, alternative feature extraction, UPerNet–ConvNeXt-Tiny, sensitivity, resources, boundaries and small-weed diagnostics.
- [S7-initialized JNDA and final evaluation](docs/s7_initialized_reproduction.md): the separate initialization-controlled route and its five-system final evaluation interface.
- The PhenoBench native training and evaluation commands are included in the original reproduction guide, with their separate data and initialization requirements.

Run commands from the repository root and use a separate output workspace for each reproduction. The fixed data partitions, seed lists, optimizer settings, endpoint roles and comparison directions are part of the experiment definition. Preserve the separation between development-based selection and evaluation on the primary and final partitions.

The complete supplementary seed values, paired contrasts and development curves are available in `reference_results/`. These reference tables include favorable and unfavorable comparisons and are not used as inputs to training or model selection.

## Verification

The release retains scientific function definitions from the frozen implementations and provides explicit path and asset configuration. The documented software checks use synthetic CPU inputs and source equivalence checks. They do not rerun the study's training or score new predictions on its evaluation partitions.

```bash
python scripts/check_release.py
python scripts/supplementary/check_synthetic.py
python -m unittest discover -s tests -p "test_*.py"
```

Source provenance and the recorded reproduction settings are documented alongside each implementation. Hardware-specific latency must be measured on the reproduction platform; no clock-frequency correction is applied.

## License and citation

JNDA code is released under the [MIT license](LICENSE). Third-party software, pretrained weights and datasets retain their respective terms; see [third-party notices](THIRD_PARTY_NOTICES.md). In particular, the code license does not grant additional rights to the CropAndWeed data.

Use `CITATION.cff` and cite the associated study and the original dataset/model works when using this implementation.
