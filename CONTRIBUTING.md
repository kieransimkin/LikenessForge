# Contribute to LikenessForge

[Kieran Simkin](https://kieransimkin.co.uk/) Â· [DanceFlow](https://kieransimkin.co.uk/danceflow/) Â· [Pull requests](https://github.com/kieransimkin/LikenessForge/pulls)

Inspect the full relevant CLI/GUI and plugin capabilities and current AGENTS.md before changing the tool. Keep orchestration in shared workflows; implement new detectors, segmenters, modifiers and trainers through their interfaces. Reproduce a useful gap, make a narrow reusable change, update README.md and include meaningful synthetic tests or a runnable example. Agents follow the same contribution route as people.

Run `uv sync --extra gui --extra dev`; set `QT_QPA_PLATFORM=offscreen` and run `uv run pytest -q`; run `uv run ruff check src tests scripts`, `uv build`, `uv run twine check --strict dist/*.whl dist/*.tar.gz`, and `uv run python scripts/validate_distribution.py dist`. CI tests must not download weights. GPU inference, likeness quality and physical desktop interaction need separate evidence beyond mocked/offscreen tests.

Use synthetic fixtures. Keep photographs, generated portraits, trained weights, tokens and private trace files out of source, test artifacts and packages. Preserve model licences and explicit model/resource choices. For public copy, describe verified capabilities and current requirements, include the website, avoid unqualified quality/VRAM promises and keep machine-specific incident logs in maintainer records.

Open a scoped PR with the problem, resulting behaviour and checks. Follow repository ownership and review; a contribution request grants no credential, account, rights or release authority. Maintainers choose fresh versions, publish matching tested distributions and preserve immutable tags/assets.

## Before every new workflow

Check the latest stable official release and the actual installed/imported version before starting; do not infer currency from a directory name or a development checkout. Read the current release notes, tagged capability documentation and relevant Potential problems. Use the [DanceFlow release preflight](https://github.com/kieransimkin/DanceFlow/blob/main/docs/workflows/release-preflight.md) and [complete capability references](https://github.com/kieransimkin/DanceFlow/blob/main/docs/capabilities/index.md).

People and agents should consider the full ecosystem, use all capabilities that materially serve the task, compose supported files/APIs, and improve a demonstrated reusable gap in its canonical tool. Record why no worthwhile change is justified when that is the result. Keep independent tool identities, compatibility, source clocks, privacy, model/download costs and publication/deployment permissions intact. Store weights and model-specific caches in the owning project's ignored `models/`.

Read [the people-and-agent ecosystem guide](docs/ECOSYSTEM.md) before selecting or extending a tool.
