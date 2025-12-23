# Benchmark Images

This directory contains images from the Natural Scenes Dataset (NSD) shared1000 subset for benchmarking HED annotation models.

## Selection Methodology

Images were randomly selected from the NSD shared1000 dataset (1000 images shared across all NSD subjects) using the following reproducible procedure:

```python
import random
random.seed(42)  # Fixed seed for reproducibility
selected = random.sample(available_images, 18)
```

Plus 2 pre-existing images (shared0001, shared0060), for a total of 20 images.

## Source

- **Dataset**: Natural Scenes Dataset (NSD)
- **Subset**: shared1000 (images shown to all 8 NSD subjects)
- **Location**: `/Volumes/S1/Datasets/NSD_stimuli/shared1000/`
- **Reference**: Allen et al. (2022). A massive 7T fMRI dataset to bridge cognitive neuroscience and artificial intelligence. Nature Neuroscience.

## Usage

The benchmark script (`model_benchmark.py`) automatically discovers all images in this directory. To add more images:

1. Copy images (jpg, jpeg, or png) to this directory
2. Run the benchmark - new images will be included automatically

## Current Images (n=20)

Selected with seed=42 on 2025-12-22:

| Image | NSD ID | Selection |
|-------|--------|-----------|
| shared0001_nsd02951 | nsd02951 | Pre-existing |
| shared0027_nsd04157 | nsd04157 | Random (seed=42) |
| shared0034_nsd04691 | nsd04691 | Random (seed=42) |
| shared0060_nsd06432 | nsd06432 | Pre-existing |
| shared0092_nsd08319 | nsd08319 | Random (seed=42) |
| shared0107_nsd09148 | nsd09148 | Random (seed=42) |
| shared0117_nsd09979 | nsd09979 | Random (seed=42) |
| shared0145_nsd11943 | nsd11943 | Random (seed=42) |
| shared0231_nsd19200 | nsd19200 | Random (seed=42) |
| shared0253_nsd20821 | nsd20821 | Random (seed=42) |
| shared0284_nsd22810 | nsd22810 | Random (seed=42) |
| shared0435_nsd32308 | nsd32308 | Random (seed=42) |
| shared0561_nsd42225 | nsd42225 | Random (seed=42) |
| shared0607_nsd44721 | nsd44721 | Random (seed=42) |
| shared0657_nsd47688 | nsd47688 | Random (seed=42) |
| shared0695_nsd50812 | nsd50812 | Random (seed=42) |
| shared0757_nsd55108 | nsd55108 | Random (seed=42) |
| shared0761_nsd55527 | nsd55527 | Random (seed=42) |
| shared0762_nsd55650 | nsd55650 | Random (seed=42) |
| shared0916_nsd66422 | nsd66422 | Random (seed=42) |
