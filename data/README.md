# Data manifests

These files contain the study's sample assignments and label definitions. Download the images and source semantic labels from the original dataset providers.

- `cropandweed_partitions.csv`: 8,034 CropAndWeed samples in the fixed train, development, primary and final partitions. Row order within each partition is retained.
- `cropandweed_class_mapping.csv`: the 101-entry source-label mapping used by the three-class task.
- `phenobench_validation_selection.csv`: the fixed 500-image public-validation selection used for the PhenoBench discovery evaluation.

The training and evaluation manifests are prepared from these tables and your local image and label files. See the reproduction instructions for the required directory layout and preparation command. The published tables do not contain machine-specific file paths.

The data partitions and sample order are part of the study protocol. Changing them produces a different experiment. Dataset files remain subject to their original licenses.
