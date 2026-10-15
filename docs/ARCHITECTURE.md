# Architecture

LikenessForge is deliberately split into four model/plugin families.

1. **Detector** – locates faces and returns stable numbered bounding boxes/landmarks.
2. **Segmenter** – converts the chosen face box into a pixel mask. It can be a tiny geometric fallback or a learned segmenter such as SAM 2.1 Tiny.
3. **Modifier** – edits the cropped face region. It receives an RGB crop, mask, prompt, optional LoRA, seed and optional reference images.
4. **Trainer** – prepares/launches a model-family-specific LoRA training job. The core workflow first uses the selected detector and segmenter to make model-sized face crops.

Third-party packages can register plugins with Python entry-point groups:

- `likenessforge.detectors`
- `likenessforge.segmenters`
- `likenessforge.modifiers`
- `likenessforge.trainers`

For local experiments, a CLI/config plugin name may also be `some.module:ClassName`.

## Repair data flow

`input -> detect -> choose face -> segment -> square context crop -> modifier + LoRA -> resize -> feathered face-mask composite -> output`

The generated crop is never pasted as a square. Only the selected face mask is composited, after dilation and Gaussian feathering. This is what removes the obvious rectangular boundary.

## Training data flow

`training photos -> detect -> choose face -> segment -> tighten around mask -> resize to trainer preferred size -> suppress background -> captions -> trainer`

This keeps the training set focused on identity and avoids wasting capacity on unrelated background detail.
