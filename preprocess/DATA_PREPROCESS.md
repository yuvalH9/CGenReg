# Data Preprocessing

The preprocessing stage creates the inputs required by C-GenReg: cached GeoTransformer features, temporal depth videos for Cosmos-Transfer1, and generated RGB videos.

If you prefer not to run the preprocessing code, the preprocessed data can be downloaded directly:

- Generated RGB videos: **[Download link to be added]**
- GeoTransformer features: **[Download link to be added]**

## GeoTransformer Features

Precomputed GeoTransformer features may be downloaded instead of installing GeoTransformer and running this stage. After extracting an archive, pass the directory containing the feature `.pkl` files to the corresponding inference argument.

Create 3DMatch features:

```bash
cd $CGENREG_ROOT
python preprocess/create_geotransformer_features.py \
  --benchmark 3dmatch \
  --threedmatch_ply_path $DATA_ROOT/3DMatch/test \
  --feat_save_path $OUTPUT_ROOT/3DMatch_GeoTrans_Feat \
  --ckpt_path $GEOTRANSFORMER_CKPT
```

Create ScanNet SuperGlue features:

```bash
cd $CGENREG_ROOT
python preprocess/create_geotransformer_features.py \
  --benchmark scannet_superglue \
  --scannet_depth_path $DATA_ROOT/ScanNetSuperGlue \
  --feat_save_path $OUTPUT_ROOT/Scannet_Superglue_GeoTrans_feat \
  --ckpt_path $GEOTRANSFORMER_CKPT
```

Both commands write their cached features to a `geotransformer_feat_3dmatch` subdirectory.

## Depth Videos

Create temporal-concat depth videos for 3DMatch:

```bash
cd $CGENREG_ROOT
python preprocess/create_3dmatch_depth_videos.py \
  --depth_root $DATA_ROOT/3DMatch/raw/test_raw \
  --output_path $OUTPUT_ROOT/3DMatch_input_depth_vids
```

Create temporal-concat depth videos for ScanNet SuperGlue:

```bash
cd $CGENREG_ROOT
python preprocess/create_scannet_superglue_depth_videos.py \
  --depth_root $DATA_ROOT/ScanNetSuperGlue/ScannetSuperGlue \
  --output_path $OUTPUT_ROOT/ScanNetSuperGlue_input_depth_vids
```

The scripts always use the temporal-concat layout expected by Cosmos-Transfer1. Depth inversion is part of the preprocessing implementation and does not require an additional flag.

## Cosmos-Transfer1 Generation

Follow the dedicated [Cosmos-Transfer1 generation guide](../cosmos_wfm/COSMOS_GEN.md) to install the pinned external repository, rewrite the included batch specs, and generate RGB videos for either benchmark.
