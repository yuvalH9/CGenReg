# Cosmos-Transfer1 Generation for C-GenReg

This directory contains only the C-GenReg scripts needed to run the Cosmos-Transfer1 generation. Cosmos-Transfer1 and its checkpoints must be installed separately.

## Cosmos-Transfer1 version

We use the following Cosmos-Transfer1 revision:

- Repository: <https://github.com/nvidia-cosmos/cosmos-transfer1>
- Commit: `e4055e39ee9c53165e85275bdab84ed20909714a`
- Commit date: June 26, 2025
- Entry point: `cosmos_transfer1/diffusion/inference/transfer.py`

These are the main package versions we used in our cosmos-transfer environment (might differ from original instructions):

| Component | Version |
| --- | --- |
| Python | 3.12.11 |
| CUDA toolkit | 12.4.1 |
| PyTorch | 2.6.0+cu124 |
| torchvision | 0.21.0+cu124 |
| torchaudio | 2.6.0+cu124 |
| xformers | 0.0.29.post2 |
| Transformers | 4.53.0 |
| vLLM | 0.8.5 |
| FlashInfer | 0.2.5+cu124torch2.6 |
| Transformer Engine | 2.4.0 |
| NumPy | 2.2.6 |
| decord | 0.6.0 |

## Install Cosmos-Transfer1

Follow the official [Cosmos-Transfer1 installation instructions](https://github.com/nvidia-cosmos/cosmos-transfer1/blob/e4055e39ee9c53165e85275bdab84ed20909714a/INSTALL.md), including its Linux, Python, CUDA, and compiled-dependency requirements.

Clone the official repository and check out the revision used here before following its installation steps:

```bash
export COSMOS_ROOT=/path/to/cosmos-transfer1

git clone https://github.com/nvidia-cosmos/cosmos-transfer1.git $COSMOS_ROOT
cd $COSMOS_ROOT
git checkout --detach e4055e39ee9c53165e85275bdab84ed20909714a
git submodule update --init --recursive
```

After installing, verify the checkout and core runtime:

```bash
cd $COSMOS_ROOT
git rev-parse HEAD
PYTHONPATH=$COSMOS_ROOT python scripts/test_environment.py
python -V
python -c 'import torch; print(torch.__version__, torch.version.cuda)'
```

## Download checkpoints

Follow the official [Cosmos-Transfer1 checkpoint instructions](https://github.com/nvidia-cosmos/cosmos-transfer1/blob/e4055e39ee9c53165e85275bdab84ed20909714a/examples/inference_cosmos_transfer1_7b.md#download-checkpoints). The generation script uses `$COSMOS_ROOT/checkpoints` by default. Set `CHECKPOINT_DIR` when the downloaded checkpoints are stored elsewhere.

## Prepare spec files

Use the following checked-in JSONL batch specs to generate RGB videos from the depth videos:

- `specs/3dmatch_test.json`: for the 3DMatch test set.
- `specs/scannet_superglue_test.json`: for ScanNet SuperGlue test set.

They preserve the absolute depth-video paths used for the original generation.
Rewrite only the directory (currently `/my_data_path`) while preserving each video basename.

```bash
export CGENREG_ROOT=/path/to/cgenreg
export CGENREG_OUTPUT=/path/to/cgenreg_outputs

mkdir -p $CGENREG_OUTPUT/generation_specs $CGENREG_OUTPUT/generated_videos

python $CGENREG_ROOT/cosmos_wfm/scripts/rewrite_spec_paths.py \
  --input-spec $CGENREG_ROOT/cosmos_wfm/specs/3dmatch_test.json \
  --depth-video-dir $CGENREG_OUTPUT/3DMatch_input_depth_vids \
  --output-spec $CGENREG_OUTPUT/generation_specs/3dmatch_test.json \
  --check-files

python $CGENREG_ROOT/cosmos_wfm/scripts/rewrite_spec_paths.py \
  --input-spec $CGENREG_ROOT/cosmos_wfm/specs/scannet_superglue_test.json \
  --depth-video-dir $CGENREG_OUTPUT/ScanNetSuperGlue_input_depth_vids \
  --output-spec $CGENREG_OUTPUT/generation_specs/scannet_superglue_test.json \
  --check-files
```

Create the depth videos first by following the [Depth Videos instructions in the data preprocessing guide](../preprocess/DATA_PREPROCESS.md#depth-videos).
`--check-files` verifies that every batch entry has a matching video before a costly generation run.

## Generate 3DMatch videos

Activate the Cosmos environment created with the official installation instructions, then run:

```bash
COSMOS_ROOT=$COSMOS_ROOT \
NUM_GPU=2 \
BATCH_INPUT_PATH=$CGENREG_OUTPUT/generation_specs/3dmatch_test.json \
VIDEO_SAVE_FOLDER=$CGENREG_OUTPUT/generated_videos/3dmatch_rgb_gen_vids \
bash $CGENREG_ROOT/cosmos_wfm/scripts/generate_cosmos_videos.sh
```

## Generate ScanNet SuperGlue videos

```bash
COSMOS_ROOT=$COSMOS_ROOT \
NUM_GPU=2 \
BATCH_INPUT_PATH=$CGENREG_OUTPUT/generation_specs/scannet_superglue_test.json \
VIDEO_SAVE_FOLDER=$CGENREG_OUTPUT/generated_videos/scannet_test_superglue_rgb_gen_vids \
bash $CGENREG_ROOT/cosmos_wfm/scripts/generate_cosmos_videos.sh
```

The same joint script is used for both datasets. It reproduces the original generation parameters: depth control, control weight 1.0 from the batch specs, `sigma_max=80`, `fps=30`, and two GPUs by default. Set `CUDA_VISIBLE_DEVICES` to choose physical GPUs. No training is involved.

## Generated files and inference layout

The commands above produce the same directory names and layout as the published generated-RGB archives:

```text
$CGENREG_OUTPUT/
  generation_specs/
    3dmatch_test.json
    scannet_superglue_test.json
  generated_videos/
    3dmatch_rgb_gen_vids/
      video_<index>/
        output.mp4
        prompt.txt
    scannet_test_superglue_rgb_gen_vids/
      video_<index>/
        output.mp4
        prompt.txt
```

The corresponding downloaded archives are named `3dmatch_rgb_gen_vids.zip` and `scannet_test_superglue_rgb_gen_vids.zip`. Extract them under `$CGENREG_OUTPUT/generated_videos` without renaming their top-level directories. Preserve empty `video_<index>` directories because their positions are part of the inference index mapping. `output.mp4` is the generated RGB video consumed by C-GenReg; `prompt.txt` records the prompt used for that video.

Pass these roots to CGenReg inference:

```text
3DMatch:          --gen_vid_path <CGENREG_OUTPUT>/generated_videos/3dmatch_rgb_gen_vids
ScanNet SuperGlue: --gen_vid_path <CGENREG_OUTPUT>/generated_videos/scannet_test_superglue_rgb_gen_vids
--gen_spec_config <CGENREG_OUTPUT>/generation_specs
```
