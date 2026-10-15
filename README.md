# LikenessForge

**Identity-preserving image repair for the DanceFlow ecosystem.**

LikenessForge is a local-first, modular CLI for taking an existing image whose composition is already right and repairing the facial likeness of a selected person using an identity LoRA. Face detection, face segmentation, image modification and LoRA training are independent plugin families, so the pipeline can evolve as smaller image models improve.

## What it does

The core repair workflow is:

1. locate every face and assign a stable index;
2. select the face to repair (`--face 0`, `--face 1`, or `--face largest`);
3. segment only that face;
4. crop a square context region around it at the modifier model's preferred resolution;
5. run the selected image modifier with the selected LoRA;
6. resize the generated crop back into place;
7. composite **only the segmented face region**, with dilation + Gaussian feathering, so no square crop edge is visible.

`-v/--verbose` prints the exact pipeline steps and writes the original image, numbered face preview, masks, crops, model output and final composite plus `trace.json` into a timestamped debug directory.

## Built-in plugins

| Type | Plugin | Model | Typical use |
|---|---|---|---|
| detector | `opencv-haar` | bundled OpenCV cascade | zero-download CPU fallback |
| detector | `mtcnn` | facenet-pytorch MTCNN | stronger compact face detection + landmarks |
| segmenter | `soft-face` | none | fast elliptical mask fallback |
| segmenter | `sam2-tiny` | `facebook/sam2.1-hiera-tiny` | learned promptable segmentation |
| modifier | `sd15-inpaint-lowvram` | `stable-diffusion-v1-5/stable-diffusion-inpainting` | **8 GB-safe baseline**, 512px |
| modifier | `sdxl-inpaint` | `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` | better fidelity/detail; CPU offload supported |
| modifier | `flux2-klein-4b` | `black-forest-labs/FLUX.2-klein-4B` | unified generation/editing + LoRAs |
| modifier | `flux2-klein-4b-lowvram` | same | 4-bit quantization when supported + CPU offload |
| trainer | `sd15-lora` | SD 1.5 base | lightest identity LoRA training path |
| trainer | `sdxl-lora` | SDXL base | higher-quality identity LoRA |
| trainer | `flux2-klein-lora` | FLUX.2 Klein Base 4B | official Klein DreamBooth-LoRA training |

Model files are downloaded on first use and cached under the platform cache directory (or `--cache-dir`). The inference model license still applies independently of LikenessForge's MIT code license.

### 8 GB GPUs

Start with `sd15-inpaint-lowvram`. `sdxl-inpaint` is also usable with model CPU offload and VAE tiling, although it is slower. `flux2-klein-4b-lowvram` requests bitsandbytes 4-bit quantisation of the transformer/text encoder and CPU offload; exact support depends on GPU/CUDA/bitsandbytes versions. BFL also publishes FP8/NVFP4 Klein weights, but those are hardware/runtime-specific and are intentionally not silently substituted by the generic plugin.

FLUX.2 Klein LoRA **training** is a different constraint from inference: the official BFL/Hugging Face guidance puts normal 4B LoRA training around 24 GB VRAM. LikenessForge exposes the official trainer and an offload path, but the SD 1.5/SDXL trainers are the practical local choices on smaller cards.

## Install

Core CLI:

```bash
python -m pip install -e .
```

A practical 8 GB setup:

```bash
python -m pip install -e '.[faces,sam,diffusion,lowvram,training]'
```

Use the appropriate PyTorch CUDA wheel/index for your machine before installing the extras when necessary.

## Inspect plugins and models

```bash
likenessforge plugins
likenessforge models list
likenessforge models download sam2-tiny
likenessforge models download sd15-inpaint-lowvram
```

## Identify the face index

```bash
likenessforge faces photo.jpg --detector mtcnn --preview faces.png
```

Example output:

```text
0: box=(218, 96, 488, 383) score=0.9987
1: box=(702, 122, 914, 362) score=0.9941
```

## Repair a face

Small/8 GB-friendly path:

```bash
likenessforge repair input.png repaired.png \
  --detector mtcnn \
  --segmenter sam2-tiny \
  --modifier sd15-inpaint-lowvram \
  --face 0 \
  --lora ./loras/kieran.safetensors \
  --lora-scale 0.9 \
  --seed 42 \
  -v
```

SDXL:

```bash
likenessforge repair input.png repaired.png \
  --detector mtcnn --segmenter sam2-tiny --modifier sdxl-inpaint \
  --face 0 --lora ./loras/person.safetensors --strength 0.52
```

FLUX.2 Klein low-VRAM path:

```bash
likenessforge repair input.png repaired.png \
  --detector mtcnn --segmenter sam2-tiny \
  --modifier flux2-klein-4b-lowvram \
  --face 0 --lora ./loras/person-klein.safetensors \
  --reference clear-reference.jpg \
  --steps 4 --guidance 4.0 -v
```

The `--reference` option is repeatable for modifier backends that support multiple references.

## Train a LoRA

LikenessForge first uses the **selected detector + selected segmenter** to reduce each source photo to a face-focused square at the chosen trainer's preferred resolution. The region outside the face is blurred by default so identity features dominate training.

```bash
likenessforge train-lora ./authorised-photos ./training/kieran \
  --detector mtcnn \
  --segmenter sam2-tiny \
  --trainer sdxl-lora \
  --trigger lf_kieran \
  --steps 900 --rank 16 \
  --low-vram -v
```

To inspect the prepared data before spending GPU time:

```bash
likenessforge prepare-training ./authorised-photos ./prepared \
  --detector mtcnn --segmenter sam2-tiny --trainer sdxl-lora \
  --trigger lf_kieran -v
```

The trainer adapters download the matching **official Hugging Face Diffusers v0.41.0 DreamBooth-LoRA script** into LikenessForge's tool cache and launch it with `accelerate`. The supported trainer is therefore replaceable independently from the rest of the pipeline.

### FLUX.2 Klein LoRA

```bash
likenessforge train-lora ./authorised-photos ./training/kieran-klein \
  --detector mtcnn --segmenter sam2-tiny \
  --trainer flux2-klein-lora --trigger lf_kieran --steps 1000
```

For genuinely low-VRAM local training, prefer `sd15-lora`; FLUX.2 Klein 4B training remains substantially more demanding than inference.

## Configuration file

See `examples/likenessforge.yaml`:

```bash
likenessforge repair input.png output.png --config examples/likenessforge.yaml --face 1 --lora person.safetensors -v
```

Plugin settings can be strings or mappings with `name` and `options`.

## Third-party/custom plugins

Plugins can be distributed using Python entry points, or loaded directly without packaging:

```bash
likenessforge repair in.png out.png --modifier my_package.my_backend:MyModifier
```

Implement the relevant abstract class in `likenessforge.plugins.base`.

## Debugging

Verbose mode is deliberately exhaustive:

```bash
likenessforge repair in.png out.png ... -v --debug-dir ./debug
```

A run contains files similar to:

```text
debug/20261015T014500Z/
  001-original.png
  002-faces.png
  003-full-face-mask.png
  004-face-crop.png
  005-face-crop-mask.png
  006-generated-face.png
  007-composited.png
  trace.json
```

`trace.json` records the selected plugin/model, face index/coordinates, crop box, resolution, seed, LoRA, modifier options and blend settings.

## Privacy and responsible use

Training images and LoRA weights remain local unless an underlying model repository must be downloaded. Do not commit private portraits or trained identity weights. Use likeness repair and identity training only on images/identities you are authorised to work with, and respect the licence/usage terms of every downloaded model.
