from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from PIL import Image

from likenessforge import __version__
from likenessforge.config import load_config
from likenessforge.debug import DebugSink
from likenessforge.model_store import ModelStore
from likenessforge.registry import create, describe_plugins
from likenessforge.workflow import RepairWorkflow, TrainingWorkflow


def _add_common(parser):
    parser.add_argument("--config", help="YAML/JSON/TOML configuration file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose trace + save every intermediate image")
    parser.add_argument("--debug-dir", default="likenessforge-debug", help="Root for verbose intermediate artefacts")


def _plugin(args, cfg, store, kind):
    name = getattr(args, kind, None) or getattr(cfg, kind).name
    opts = dict(getattr(cfg, kind).options)
    if kind == "modifier":
        opts.setdefault("low_vram", getattr(args, "low_vram", None) or cfg.low_vram)
    return create(kind, name, store, **opts)


def _logging(verbose):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def build_parser():
    p = argparse.ArgumentParser(prog="likenessforge", description="Modular facial-likeness repair and LoRA training")
    p.add_argument("--version", action="version", version=f"LikenessForge {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    gui = sub.add_parser("gui", help="Launch the optional Qt desktop GUI")
    _add_common(gui)

    plugins = sub.add_parser("plugins", help="List available detector/segmenter/modifier/trainer plugins")
    _add_common(plugins)
    plugins.add_argument("--json", action="store_true")

    models = sub.add_parser("models", help="Inspect or download model-backed plugins")
    _add_common(models)
    models.add_argument("action", choices=["list", "download"])
    models.add_argument("name", nargs="?", help="Plugin name for download")

    faces = sub.add_parser("faces", help="Detect and index faces in an image")
    _add_common(faces)
    faces.add_argument("image")
    faces.add_argument("--detector")
    faces.add_argument("--preview", help="Write annotated preview image")

    repair = sub.add_parser("repair", help="Repair facial likeness in an existing image")
    _add_common(repair)
    repair.add_argument("image")
    repair.add_argument("output")
    repair.add_argument("--detector")
    repair.add_argument("--segmenter")
    repair.add_argument("--modifier")
    repair.add_argument("--face", default="0", help="Face index from `faces`, or 'largest'")
    repair.add_argument("--lora", help="Local LoRA path or Hugging Face LoRA repo")
    repair.add_argument("--lora-scale", type=float, default=1.0)
    repair.add_argument("--reference", action="append", default=[], help="Optional identity reference image (repeatable)")
    repair.add_argument("--prompt")
    repair.add_argument("--negative-prompt")
    repair.add_argument("--crop-margin", type=float)
    repair.add_argument("--feather", type=int)
    repair.add_argument("--seed", type=int, default=0)
    repair.add_argument("--steps", type=int)
    repair.add_argument("--guidance", type=float)
    repair.add_argument("--strength", type=float)
    repair.add_argument("--low-vram", choices=["auto", "off", "cpu_offload", "bnb4"], default=None)

    wizard = sub.add_parser("repair-wizard", help="Try a guided set of repair settings and build comparison sheets")
    _add_common(wizard)
    wizard.add_argument("image")
    wizard.add_argument("output", help="Directory for wizard candidates and comparison sheets")
    wizard.add_argument("--detector")
    wizard.add_argument("--segmenter")
    wizard.add_argument("--modifier")
    wizard.add_argument("--face", default="largest", help="Face index from `faces`, or 'largest'")
    wizard.add_argument("--lora", help="Local LoRA path or Hugging Face LoRA repo")
    wizard.add_argument("--reference", action="append", default=[], help="Optional identity reference image (repeatable)")
    wizard.add_argument("--preset", choices=["quick", "balanced", "thorough"], default="balanced")
    wizard.add_argument("--seed", type=int, default=0)
    wizard.add_argument("--prompt")
    wizard.add_argument("--negative-prompt")
    wizard.add_argument("--low-vram", choices=["auto", "off", "cpu_offload", "bnb4"], default=None)

    train = sub.add_parser("train-lora", help="Crop/segment a face dataset and train a LoRA")
    _add_common(train)
    train.add_argument("images", help="Directory of authorised training images")
    train.add_argument("output", help="Training output directory")
    train.add_argument("--detector")
    train.add_argument("--segmenter")
    train.add_argument("--trainer")
    train.add_argument("--trigger", required=True, help="Unique subject token, e.g. lf_kieran")
    train.add_argument("--face", default="largest", help="Face index or 'largest' for each training image")
    train.add_argument("--background", choices=["blur", "neutral", "black", "original"], default="blur")
    train.add_argument("--steps", type=int, default=800)
    train.add_argument("--rank", type=int, default=16)
    train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument("--low-vram", action="store_true")
    train.add_argument("--extra-arg", action="append", default=[], help="Pass an extra argument to the trainer; repeatable")

    train_wizard = sub.add_parser(
        "train-wizard",
        help="Train several LoRA variants and compare them on a validation photograph",
    )
    _add_common(train_wizard)
    train_wizard.add_argument("images", help="Directory of authorised training images")
    train_wizard.add_argument("validation", help="Holdout photograph used to compare the trained LoRAs")
    train_wizard.add_argument("output", help="Directory for candidate trainings and comparison sheets")
    train_wizard.add_argument("--detector")
    train_wizard.add_argument("--segmenter")
    train_wizard.add_argument("--trainer")
    train_wizard.add_argument("--modifier")
    train_wizard.add_argument("--trigger", required=True, help="Unique subject token, e.g. lf_kieran")
    train_wizard.add_argument("--face", default="largest", help="Training face index or 'largest'")
    train_wizard.add_argument(
        "--validation-face",
        default="largest",
        help="Validation face index or 'largest'",
    )
    train_wizard.add_argument("--preset", choices=["quick", "balanced", "thorough"], default="balanced")
    train_wizard.add_argument("--seed", type=int, default=0, help="Fixed validation-repair seed")
    train_wizard.add_argument("--low-vram", action="store_true", help="Use trainer low-VRAM/offload options")
    train_wizard.add_argument("--prompt")
    train_wizard.add_argument("--negative-prompt")

    prep = sub.add_parser("prepare-training", help="Only prepare segmented/cropped LoRA training images")
    _add_common(prep)
    prep.add_argument("images")
    prep.add_argument("output")
    prep.add_argument("--detector")
    prep.add_argument("--segmenter")
    prep.add_argument("--trainer")
    prep.add_argument("--trigger", required=True)
    prep.add_argument("--face", default="largest")
    prep.add_argument("--background", choices=["blur", "neutral", "black", "original"], default="blur")
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    _logging(args.verbose)
    cfg = load_config(getattr(args, "config", None))
    store = ModelStore()
    debug = DebugSink(args.verbose, args.debug_dir)

    if args.command == "gui":
        try:
            from likenessforge.gui.app import main as gui_main
        except ImportError as exc:
            parser.error(f"GUI dependencies are missing ({exc}); install likenessforge[gui]")
        return gui_main([])

    if args.command == "plugins":
        rows = describe_plugins()
        if args.json:
            print(json.dumps(rows, indent=2))
        else:
            for row in rows:
                marker = " [low-vram]" if row["low_vram"] else ""
                model = f" | {row['model_id']}" if row["model_id"] else ""
                print(f"{row['kind']:9} {row['name']:28}{marker} | {row['description']}{model}")
        return 0

    if args.command == "models":
        rows = [r for r in describe_plugins() if r["model_id"]]
        if args.action == "list":
            for row in rows:
                print(f"{row['kind']:9} {row['name']:28} {row['model_id']}")
            return 0
        if not args.name:
            parser.error("models download requires a plugin name")
        matches = [r for r in rows if r["name"] == args.name]
        if not matches:
            parser.error(f"unknown model-backed plugin: {args.name}")
        path = store.ensure_hf_snapshot(matches[0]["model_id"])
        print(path)
        return 0

    detector = (
        _plugin(args, cfg, store, "detector")
        if args.command
        in {"faces", "repair", "repair-wizard", "train-lora", "train-wizard", "prepare-training"}
        else None
    )

    if args.command == "faces":
        image = Image.open(args.image).convert("RGB")
        workflow = RepairWorkflow(detector, None, None, debug)
        faces = workflow.inspect_faces(image)
        for face in faces:
            print(f"{face.index}: box={face.box.as_tuple()} score={face.score:.4f}")
        if args.preview:
            from likenessforge.image_ops import annotate_faces

            annotate_faces(image, faces).save(args.preview)
        return 0 if faces else 2

    if args.command == "repair":
        segmenter = _plugin(args, cfg, store, "segmenter")
        modifier = _plugin(args, cfg, store, "modifier")
        image = Image.open(args.image).convert("RGB")
        refs = [Image.open(p).convert("RGB") for p in args.reference]
        options = {k: v for k, v in {"steps": args.steps, "guidance": args.guidance, "strength": args.strength}.items() if v is not None}
        workflow = RepairWorkflow(detector, segmenter, modifier, debug)
        result = workflow.repair(
            image=image,
            face_selector=args.face,
            lora=args.lora,
            prompt=args.prompt or cfg.prompt,
            negative_prompt=args.negative_prompt or cfg.negative_prompt,
            crop_margin=args.crop_margin if args.crop_margin is not None else cfg.crop_margin,
            feather=args.feather if args.feather is not None else cfg.feather,
            seed=args.seed,
            lora_scale=args.lora_scale,
            reference_images=refs,
            modifier_options=options,
        )
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        result.save(args.output)
        print(args.output)
        if debug.root:
            print(f"debug: {debug.root}")
        return 0

    if args.command == "repair-wizard":
        segmenter = _plugin(args, cfg, store, "segmenter")
        modifier = _plugin(args, cfg, store, "modifier")
        image = Image.open(args.image).convert("RGB")
        refs = [Image.open(p).convert("RGB") for p in args.reference]
        output_dir = Path(args.output)
        workflow = RepairWorkflow(detector, segmenter, modifier, debug)

        def progress(index, total, label):
            print(f"[{index}/{total}] {label}")

        run = workflow.wizard(
            image=image,
            output_dir=output_dir,
            face_selector=args.face,
            lora=args.lora,
            prompt=args.prompt or cfg.prompt,
            negative_prompt=args.negative_prompt or cfg.negative_prompt,
            preset=args.preset,
            seed=args.seed,
            reference_images=refs,
            progress=progress,
        )
        print(run.comparison_face_path)
        print(run.comparison_full_path)
        print(run.manifest_path)
        return 0

    if args.command == "train-wizard":
        segmenter = _plugin(args, cfg, store, "segmenter")
        trainer = _plugin(args, cfg, store, "trainer")
        modifier = _plugin(args, cfg, store, "modifier")
        if trainer.info.lora_family != modifier.info.lora_family:
            parser.error(
                f"trainer {trainer.info.name} produces {trainer.info.lora_family}, but modifier "
                f"{modifier.info.name} expects {modifier.info.lora_family}"
            )
        validation = Image.open(args.validation).convert("RGB")
        output_dir = Path(args.output)
        training_workflow = TrainingWorkflow(detector, segmenter, trainer, debug)
        repair_workflow = RepairWorkflow(detector, segmenter, modifier, debug)

        def progress(index, total, label):
            print(f"[{index}/{total}] {label}")

        run = training_workflow.wizard(
            source_dir=Path(args.images),
            validation_image=validation,
            repair_workflow=repair_workflow,
            output_dir=output_dir,
            trigger=args.trigger,
            selector=args.face,
            validation_face_selector=args.validation_face,
            preset=args.preset,
            seed=args.seed,
            low_vram=args.low_vram,
            prompt=args.prompt or cfg.prompt,
            negative_prompt=args.negative_prompt or cfg.negative_prompt,
            progress=progress,
        )
        print(run.comparison_face_path)
        print(run.comparison_full_path)
        print(run.manifest_path)
        return 0

    if args.command in {"train-lora", "prepare-training"}:
        segmenter = _plugin(args, cfg, store, "segmenter")
        trainer = _plugin(args, cfg, store, "trainer")
        workflow = TrainingWorkflow(detector, segmenter, trainer, debug)
        if args.command == "prepare-training":
            count = workflow.prepare_dataset(
                Path(args.images), Path(args.output), args.trigger, args.face, args.background
            )
            print(f"prepared {count} images in {args.output}")
            return 0
        weights = workflow.train(
            Path(args.images),
            Path(args.output),
            args.trigger,
            selector=args.face,
            background=args.background,
            steps=args.steps,
            rank=args.rank,
            lr=args.learning_rate,
            low_vram=args.low_vram,
            extra_args=args.extra_arg,
        )
        print(weights)
        if debug.root:
            print(f"debug: {debug.root}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
