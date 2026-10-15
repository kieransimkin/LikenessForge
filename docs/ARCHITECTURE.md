# Architecture

LikenessForge is deliberately split into four model/plugin families.

1. **Detector** – locates faces and returns stable numbered bounding boxes/landmarks. The recommended built-in detector is OpenCV YuNet, whose tiny ONNX model is independent of the PyTorch/Torchvision stack used by diffusion backends.
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


## Desktop GUI

The Qt GUI in `likenessforge.gui` is deliberately an orchestration layer, not a second implementation of the image pipeline. It creates the same registered detector/segmenter/modifier/trainer plugins and calls `RepairWorkflow` / `TrainingWorkflow` in `TaskThread` workers so model downloads, inference and training cannot block the Qt event loop.

`ModelStore` accepts a structured progress callback. Hugging Face's `tqdm_class` hook is adapted into `ModelProgress` events, allowing GUI, CLI or future HTTP frontends to observe model checks, per-file byte progress, completion and failures without coupling the store to Qt.

The GUI exposes global plugin selection plus two workflow tabs:

- **Train likeness LoRA** accepts an explicit list of reference image paths. `TrainingWorkflow` still owns segmentation/cropping and writes `likenessforge-training.json` with the trainer/model/LoRA-family metadata.
- **Repair photograph** uses face detection first so the user can select a stable face index, then invokes the normal repair/crop/modify/feathered-composite pipeline.

Training subprocess output is streamed from `accelerate` through the trainer plugin callback. `DebugSink` also supports an event callback, so exact workflow steps and saved intermediate artefacts can be shown live without changing the core workflow.

## Bootstrap and CI

`.python-version` pins the default uv-managed development/runtime interpreter. `start-gui.sh`, `start-gui.ps1` and `start-gui.bat` bootstrap uv when absent, let uv install Python, sync the `all` extra, and launch the GUI. The GitHub test workflow intentionally installs only `gui` + `dev` extras because plugin registration and GUI construction are testable without downloading multi-gigabyte model runtimes or weights.
