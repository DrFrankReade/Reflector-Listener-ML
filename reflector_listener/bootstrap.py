"""Explicit desktop startup helpers; importing this module has no side effects."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _extended_windows_path(path: Path) -> str:
    """Return an extended path for the uncommon cross-drive fallback."""
    value = path.resolve().as_posix()
    if value.startswith("//?/"):
        return value
    if value.startswith("//"):
        return "//?/UNC/" + value.lstrip("/")
    return "//?/" + value


def _tcl_runtime_path(path: Path, working_directory: Path | None = None) -> str:
    """Avoid Tcl's faulty normalization of absolute profile-junction paths."""
    start = Path.cwd() if working_directory is None else Path(working_directory)
    try:
        return Path(os.path.relpath(path.resolve(), start.resolve())).as_posix()
    except ValueError:  # different Windows drives cannot form a relative path
        return _extended_windows_path(path)


def _configure_tk_runtime(
    base_prefix: Path | None = None,
    environment: dict[str, str] | None = None,
    working_directory: Path | None = None,
) -> bool:
    """Point Tcl/Tk at the scripts supplied by the active Python runtime.

    Relative paths matter in restricted Windows environments where Tcl can
    incorrectly normalize absolute profile junctions such as ``AppData``.
    Supplying them before importing tkinter keeps GUI discovery anchored to
    the selected Python installation. An extended path is used only when
    Python is on another drive from the process working directory.
    """
    if getattr(sys, "frozen", False):
        return True
    if os.name != "nt":
        return False
    base = Path(sys.base_prefix) if base_prefix is None else Path(base_prefix)
    tcl_library = base / "tcl" / "tcl8.6"
    tk_library = base / "tcl" / "tk8.6"
    if not (
        tcl_library.joinpath("init.tcl").is_file()
        and tk_library.joinpath("tk.tcl").is_file()
    ):
        return False
    target = os.environ if environment is None else environment

    def set_library(name: str, path: Path) -> None:
        target[name] = _tcl_runtime_path(path, working_directory)

    set_library("TCL_LIBRARY", tcl_library)
    set_library("TK_LIBRARY", tk_library)
    return True


def prepare_startup(app_dir: Path) -> bool:
    """Configure Tk and offer dependency installation only on explicit launch."""
    from dependencies import ensure_dependencies

    _configure_tk_runtime()
    return ensure_dependencies(app_dir=app_dir)
