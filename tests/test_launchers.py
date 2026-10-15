from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_gui_launchers_bootstrap_uv_and_use_managed_environment():
    shell = (ROOT / "start-gui.sh").read_text(encoding="utf-8")
    powershell = (ROOT / "start-gui.ps1").read_text(encoding="utf-8")
    batch = (ROOT / "start-gui.bat").read_text(encoding="utf-8")

    assert "astral.sh/uv/install.sh" in shell
    assert '"$UV_BIN" python install' in shell
    assert '"$UV_BIN" sync --extra all' in shell
    assert "likenessforge-gui" in shell
    assert 'LIKENESSFORGE_PROJECT_ROOT="$SCRIPT_DIR"' in shell

    assert "astral.sh/uv/install.ps1" in powershell
    assert "python install" in powershell
    assert "sync --extra all" in powershell
    assert "likenessforge-gui" in powershell
    assert "$env:LIKENESSFORGE_PROJECT_ROOT = $PSScriptRoot" in powershell

    assert "start-gui.ps1" in batch
    assert "ExecutionPolicy Bypass" in batch


def test_python_version_is_pinned_for_uv_bootstrap():
    assert (ROOT / ".python-version").read_text(encoding="utf-8").strip() == "3.12"
