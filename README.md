# SAM-UNet

SAM-UNet pairs a frozen Segment Anything (SAM) image encoder with a lightweight
U-Net style decoder that is adapted for medical image segmentation. The project
contains utilities for training on the SA-Med2D-16M collection as well as
zero-shot evaluation on external datasets.

## Sources:
- code: https://huggingface.co/papers/2408.09886
- dataset: https://huggingface.co/datasets/OpenGVLab/SA-Med2D-20M


## Repository structure

```
├── sam_unet/                  # Core python package
│   ├── config.py              # Central configuration (paths, hyper-parameters)
│   ├── dataset.py             # PyTorch dataset objects and collate function
│   ├── models/                # SAM-UNet model definition and wrappers
│   ├── utils/                 # Metrics, losses, transforms and helpers
│   ├── train_on_single_gpu.py # Training entry-point for 1 GPU
│   ├── train_on_multi_gpu.py  # DDP training script
│   └── test_*.py              # Evaluation utilities
├── datasets/                  # Scripts for preparing example datasets
│   ├── SAMed2D16M/            # Train/test split helpers for SA-Med2D-16M
│   └── zero-shot-dataset/     # Converters for public zero-shot benchmarks
└── README.md
```

## Installation

1. **Create a Python environment** (Python ≥ 3.10 is required):

   ```bash
   conda create -n sam-unet python=3.10
   conda activate sam-unet
   ```

2. **Install PyTorch** with CUDA that matches your system. Example for CUDA 11.8:

   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Install project dependencies**:

   ```bash
   pip install -U numpy opencv-python nibabel matplotlib scikit-image scikit-learn tqdm pillow scipy
   pip install -e .
   ```

4. (Optional) **Install additional tooling** for experiments:

   ```bash
   pip install tensorboard
   ```

## Configuration

All runtime settings live in `sam_unet/config.py`. Update the following keys
before training or evaluation:

- `data_directory`: list of folders that contain prepared datasets. Each folder
  is expected to follow the layout described in [Dataset structure](#dataset-structure).
- `root_dir`: absolute path that prefixes the relative paths stored in the JSON
  mapping files generated during preprocessing.
- `checkpoint_path`: path to the original SAM checkpoint (`vit_b` variant
  matching the chosen image resolution).
- `work_dir`: output directory for logs and trained weights.

You can keep several dataset folders in `data_directory` to mix modalities. The
training/evaluation scripts will iterate over the concatenated samples.

## Dataset structure

Every dataset directory must follow this 2D slice layout:

```
<dataset_name>/
├── train/
│   ├── images/                         # 8-bit PNG or JPG slices
│   ├── masks/                          # Binary PNG masks aligned with images
│   └── image2label_train.json          # {"images/<file>": ["masks/<file>", ...], ...}
└── test/
    ├── images/
    ├── masks/
    └── label2image_test.json           # {"masks/<file>": "images/<file>", ...}
```

`image2label_train.json` supports multiple masks per image, enabling training on
multi-instance annotations. All file paths stored in the JSONs are **relative to
`config_dict['root_dir']`** so that the same data can be shared across machines.

## Preparing datasets

### SA-Med2D-16M

1. Download the processed SA-Med2D-16M dataset.
2. Update `root_dir` in `sam_unet/config.py` to the directory that contains the
   raw JSON mapping files from the release.
3. Use `datasets/SAMed2D16M/split_to_train_test.py` to build per-modality
   train/test splits and copy files into a new `train_test/<modality>` folder
   (edit `root_dir` and `output_dir` at the bottom of the script).
4. Set `data_directory` to the list of generated modality folders inside
   `train_test`.

### General recipe for a new dataset

1. Convert each 3D volume (NIfTI, DICOM, etc.) into 2D slices.
   - Select slices with meaningful annotations (e.g., >0.5% foreground pixels).
   - Normalize the image intensities to `[0, 255]` and save them as 8-bit PNGs.
     Replicate grayscale channels to RGB so the SAM encoder receives 3 channels.
   - Export the corresponding masks as 0/255 PNGs.
2. Group the PNGs into `train/images`, `train/masks`, `test/images`, and
   `test/masks`. Create JSON mapping files using the templates in
   `datasets/SAMed2D16M/split_to_train_test.py` (`image2label_train.json`) and
   `datasets/zero-shot-dataset/HNTSMRG24.py` (`label2image.json`).
3. Set `root_dir` to the directory that contains the dataset folder and add the
   dataset path to `data_directory`.
4. Verify that `sam_unet/dataset.py` can load a sample by running:

   ```bash
   python - <<'PY'
   from sam_unet.dataset import TestingDataset
   from sam_unet.config import config_dict
   ds = TestingDataset(config_dict['data_directory'][0])
   print(len(ds), ds[0]['image'].shape, ds[0]['labels'].shape)
   PY
   ```

### HECKTOR 2022 (Head & Neck Tumor Segmentation)

The HECKTOR 2022 challenge provides co-registered CT/PET volumes with manual
GTV masks. The steps below outline how to convert the dataset into the format
required by SAM-UNet for **testing/evaluation**.

1. **Download and unpack the HECKTOR 2022 training set.** Each patient folder
   contains `ct.nii.gz`, `pt.nii.gz`, and `seg.nii.gz` under `imagesTr/` and the
   corresponding metadata.
2. **Choose an imaging modality.** SAM-UNet expects 3-channel 8-bit slices. A
   common choice is to use the CT modality. PET can also be used; ensure the
   same modality is used consistently across all slices.
3. **Generate PNG slices and masks** (example script):

   ```python
   import nibabel as nib
   import numpy as np
   import os
   from pathlib import Path
   from tqdm import tqdm
   import json
   from PIL import Image

   hecktor_root = Path('/path/to/hecktor2022')  # directory containing patient folders
   target_root = Path('/path/to/hecktor_slices/test')
   (target_root / 'images').mkdir(parents=True, exist_ok=True)
   (target_root / 'masks').mkdir(parents=True, exist_ok=True)

   label2image = {}
   keep_ratio = 100 / (256 * 256)  # match the filtering used in HNTSMRG24

   for patient_dir in tqdm(sorted(hecktor_root.iterdir())):
       ct = nib.load(patient_dir / 'ct.nii.gz').get_fdata()
       seg = nib.load(patient_dir / 'seg.nii.gz').get_fdata().astype(np.uint8)

       for k in range(seg.shape[-1]):
           mask_slice = seg[..., k]
           if np.count_nonzero(mask_slice) / mask_slice.size < keep_ratio:
               continue  # skip slices with tiny tumors

           img_slice = ct[..., k]
           img_slice = np.clip(img_slice, -200, 300)  # optional HU windowing
           img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-6)
           img_uint8 = (img_slice * 255).astype(np.uint8)
           img_rgb = np.stack([img_uint8]*3, axis=-1)

           base = f'{patient_dir.name}_slice_{k:03d}.png'
           img_rel = f'images/{base}'
           mask_rel = f'masks/{base}'

           Image.fromarray(img_rgb).save(target_root / img_rel)
           Image.fromarray((mask_slice > 0).astype(np.uint8) * 255).save(target_root / mask_rel)

           label2image[mask_rel] = img_rel

   with open(target_root / 'label2image_test.json', 'w') as f:
       json.dump(label2image, f, indent=2)
   ```

   - Adjust the HU windowing range for CT or normalization for PET as needed.
   - Remove non-tumor labels if `seg.nii.gz` contains multiple classes.
4. **Register the dataset**:
   - Update `config_dict['root_dir']` to the parent directory of
     `hecktor_slices`.
   - Append `'/path/to/hecktor_slices'` to `config_dict['data_directory']` (or
     pass it directly to the evaluation script).
5. **Run inference** on the prepared dataset:

   ```bash
   python sam_unet/test_model_zero_shot.py \
       --checkpoint /path/to/trained_checkpoint.pth \
       --dataset /path/to/hecktor_slices
   ```

   Modify the evaluation script arguments as needed (batch size, metrics).

## Training

### Single GPU

```bash
python sam_unet/train_on_single_gpu.py \
    --run_name sa-med16m \
    --epochs 30 \
    --batch_size 16 \
    --lr 1e-4 \
    --device cuda:0
```

- Models and logs are written to `${work_dir}/model_train/<run_name>/`.
- Use `--resume` to load a saved checkpoint.

### Distributed training

Use `sam_unet/train_on_multi_gpu.py` with `torchrun` (edit the script for your
cluster’s launch configuration):

```bash
torchrun --nproc_per_node=4 sam_unet/train_on_multi_gpu.py --config sam_unet/config.py
```

## Evaluation

- `sam_unet/test_model_on_test_set_total.py`: evaluate a checkpoint on the
  configured test split.
- `sam_unet/test_model_zero_shot.py`: run zero-shot inference on an external
  dataset prepared as described above.
- `sam_unet/test_model_of_diff_mod_on_test_set.py`: compare checkpoints trained
  on different modalities.

Each script reports Dice/IoU metrics defined in `sam_unet/utils/metrics.py`. The
metrics operate on the post-processed binary masks produced by the model.

## Tips

- Ensure that all masks are strictly binary (0/255). The dataloaders assert this
  condition and will fail otherwise.
- The `ResizeLongestSide` transform pads inputs to `img_size` (default 256).
  Update the value in `config.py` if you retrain with higher resolutions, and
  make sure to load the matching SAM checkpoint (e.g., `vit_b_512`).
- Use `sam_unet/utils/utils.get_logger` to capture both console and file logs
  for reproducibility.

## Citation

If you use SAM-UNet in your research, please cite the original authors of SAM
and SA-Med2D as applicable.
