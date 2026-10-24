# Third party materials

The MIT license in this repository applies to the JNDA code and its documentation. It does not replace the licenses of dependencies, datasets, or pretrained model weights.

## Diffusers runtime

The separately distributed release asset `diffusers-0.40.0.dev0-py3-none-any.whl` contains the frozen Hugging Face Diffusers runtime used in the study. Its 996 installed library source files are unchanged. The distribution includes its upstream Apache License 2.0 text; a copy is also provided in `third_party/licenses/diffusers-LICENSE.txt`. The wheel contains a relative-path source manifest for verifying individual files.

Upstream project: https://github.com/huggingface/diffusers

## Other software dependencies

Transformers, PEFT and Accelerate are maintained by Hugging Face under their respective upstream Apache License 2.0 terms. PyTorch, TorchVision, NumPy, SciPy, Pillow and the other installation dependencies retain their own licenses. The frozen environment lists versions; installation obtains these packages from their providers.

- https://github.com/huggingface/transformers
- https://github.com/huggingface/peft
- https://github.com/huggingface/accelerate
- https://github.com/pytorch/pytorch
- https://github.com/pytorch/vision
- https://github.com/numpy/numpy
- https://github.com/scipy/scipy
- https://github.com/python-pillow/Pillow

The SegFormer and UPerNet implementations use pretrained specialist checkpoints identified in the reproduction instructions and model catalog. Their source model architectures and checkpoint providers are credited there. The JNDA license does not relicense those checkpoints.

## Datasets

Dataset images and annotations are obtained from the original providers and are not bundled here. The tables in `data/` identify the fixed study samples and label mappings.

CropAndWeed is distributed by its authors for non-commercial purposes under the terms at https://github.com/cropandweed/cropandweed-dataset/blob/main/LICENCE . Its label names and mappings are attributed to that dataset. Consult those terms when obtaining or using its images and annotations; the MIT code license does not grant additional data rights.

Reference: Daniel Steininger, Andreas Trondl, Gerardus Croonen, Julia Simon and Verena Widhalm. The CropAndWeed Dataset: A Multi-Modal Learning Approach for Efficient Crop and Weed Manipulation. WACV 2023, pp. 3729–3738.

PhenoBench is distributed under CC BY-SA 4.0: https://www.phenobench.org/dataset.html . The PhenoBench-derived validation selection and coverage table is supplied with attribution to that dataset and subject to those source-data terms.

Reference: Jan Weyler, Federico Magistri, Elias Marks, Yue Linn Chong, Matteo Sodano, Gianmarco Roggiolani, Nived Chebrolu, Cyrill Stachniss and Jens Behley. PhenoBench—A Large Dataset and Benchmarks for Semantic Image Interpretation in the Agricultural Domain. IEEE TPAMI 46(12), 9583–9594, 2024.

## Pretrained generative model

The study uses Black Forest Labs FLUX.2 [klein] 4B Base, revision `8c44a2fbef88fae175da65df054db0f901aa9747`. Its repository declares Apache License 2.0. Obtain the model and its license from https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B . No base-model weights are redistributed in this repository.
