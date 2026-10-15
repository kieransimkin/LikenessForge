# Agent instructions for LikenessForge

- LikenessForge is the DanceFlow ecosystem's modular facial-likeness repair and identity-LoRA tool. Preserve the separation between face detectors, face segmenters, image modifiers and LoRA trainers.
- Do not hard-code a model into the repair workflow. New inference and training backends must implement the plugin interfaces and remain selectable by CLI/config/GUI.
- The Qt GUI is an orchestration frontend over the same `ModelStore`, plugin registry, `RepairWorkflow` and `TrainingWorkflow` used by the CLI. Do not duplicate model download, crop/segment, training, inference or compositing logic in GUI-only code.
- Model-backed plugins must acquire weights through their canonical Hugging Face/PyTorch-compatible source and cache them through `ModelStore` or the underlying framework cache. Preserve structured `ModelProgress` reporting so frontends can show live downloads. Never commit model weights.
- Keep low-VRAM behaviour explicit. A backend marked low-VRAM should either fit an 8 GB-class GPU in its documented mode, or clearly state the hardware-specific limitation and use offload/quantisation rather than silently OOMing.
- Keep private training images, generated portraits, model tokens and LoRA weights out of fixtures, logs, packages and source control. Tests must use synthetic images and mocked model adapters.
- Every feature change must update README.md and add/adjust tests. GUI tests should use `QT_QPA_PLATFORM=offscreen` and must not download model weights. Keep the uv bootstrap launchers working without a preinstalled Python.
- The verbose trace is a compatibility feature: it should continue to expose each stage and save every intermediate image when enabled. GUI live logging is additive to this persisted trace.
- Website: https://kieransimkin.co.uk/danceflow/

## Marketing, documentation and handoff

- Keep README.md and package metadata current for every capability, command, default or integration change. Lead with the user benefit, shortest setup path, supported environments and material current limitations; do not copy internal incident histories into adopter-facing copy.
- Include canonical website and repository/contribution links in README and supported metadata fields. Keep discoverability phrases truthful; local phrase counts are not external search demand. Avoid guaranteed likeness or universal VRAM claims.
- Inspect capabilities fully and identify a concrete reusable improvement where justified; use the existing plugin/workflow owners. Submit improvements with documentation, synthetic tests and a reproducible example to https://github.com/kieransimkin/LikenessForge/pulls. See CONTRIBUTING.md for checks and privacy requirements.
- Keep DanceFlow membership and explicit file handoffs distinct from implemented automatic integrations. Preserve canonical artwork, private reference/LoRA provenance and human review of each candidate; preparing a repair does not approve public use.
- Validate built wheel/source metadata, links, source/version identity and private-file exclusions before publishing a fresh matching immutable release. Verify each configured registry separately; do not infer a registry listing from GitHub.
