# LikenessForge image preparation in DanceFlow

[DanceFlow](https://kieransimkin.co.uk/danceflow/) · [LikenessForge source and current commands](https://github.com/kieransimkin/LikenessForge)

Use LikenessForge when an existing composition is suitable but a selected person's facial likeness needs repair, or when authorised reference images need face-focused preparation and identity LoRA training. It prepares still-image candidates; it does not identify people or certify an accurate likeness.

## Choose and record the inputs

Inspect the actual source image and approved identity references. Preserve canonical cover artwork and original photographs unchanged. Record private source/reference paths, hashes, intended use and approval state. Keep portraits, trained identity weights, debug traces and rights evidence out of public repositories and releases. Do not turn permission to prepare an image into permission to publish it.

Inspect `likenessforge plugins`, `models list` and the selected plugin documentation. Choose the detector, segmenter, modifier and trainer explicitly; record model revisions when available, LoRA architecture, settings, seed and hardware. Pair compatible trainer/modifier families. First use may download models and helper scripts; respect the selected resource scope and model terms. Low-VRAM modes depend on the actual GPU/runtime and workload.

## Prepare and review a repair

Inspect the numbered output from `faces` before choosing a face index or `largest`. Use `repair` for a known recipe or `repair-wizard` for a bounded comparison. Review both face and full-image comparison sheets against the inspected references: likeness, expression, lighting, blend boundaries, other people, unchanged composition, text, crop, alpha and phone-size use. Keep every original, candidate and explicit rejection distinct; select a candidate only after human review. A completed run or comparison sheet is not approval or a quality guarantee.

For training, inspect `prepare-training` crops before starting `train-lora` or `train-wizard`. Keep the validation photograph outside the training set. Compare training variants with the same holdout/repair settings. Retain the training manifest, plugin/model/LoRA identities, settings, seeds and hashes. Automated synthetic/offscreen tests establish orchestration, not actual GPU fit or likeness quality.

## Hand the selected asset to the next tool

Keep the selected still and its lineage as an explicit file handoff. Use RasterMoves for applicable upscaling/refinement, PixelCue for optional media tagging and ThumbMoves for cached thumbnails; none certifies identity or public-use rights. Every derivative gets its own hash and review. Use DanceMoves only for subsequent web motion and its lifecycle/accessibility; LikenessForge adds no clock, sensor or page runtime. Review integration documentation before composing tools; no automatic bridge or MCP repair/training adapter is implied.

For campaign/EPK use, retain the selected candidate alongside canonical references, assess visual coherence, verify alpha/crop and transformed bounds where relevant, then follow the destination's existing approval, scheduling, metadata, responsive and playback checks. Generated portraits remain artist-authored promotional assets, not independent publicity. Record prepared, approved, uploaded, scheduled and verified-public states separately. An unchanged canonical artwork must not be silently replaced by a repaired derivative.
