"""Dependency validation and automatic satisfaction for Reflector Messenger."""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any, NamedTuple, Sequence


PIP_TIMEOUT_SECONDS = 300


class Dependency(NamedTuple):
    name: str
    import_name: str
    pip_spec: str
    description: str


# Core required dependencies for Reflector Messenger
DEFAULT_DEPENDENCIES: list[Dependency] = [
    Dependency(
        name="Pillow",
        import_name="PIL",
        pip_spec="Pillow>=10.0",
        description="Image processing and RS-MS1 picture support",
    ),
]

if sys.version_info < (3, 11):
    DEFAULT_DEPENDENCIES.append(
        Dependency(
            name="tomli",
            import_name="tomli",
            pip_spec="tomli>=1.1.0",
            description="TOML configuration parser for Python < 3.11",
        )
    )


def check_missing_dependencies(
    dependencies: Sequence[Dependency] = DEFAULT_DEPENDENCIES,
) -> list[Dependency]:
    """Return a list of dependencies that cannot be imported."""
    missing: list[Dependency] = []
    for dep in dependencies:
        try:
            spec = importlib.util.find_spec(dep.import_name)
            if spec is None:
                missing.append(dep)
        except (ImportError, ValueError, AttributeError):
            missing.append(dep)
    return missing


def install_dependencies(
    specs: Sequence[str],
    requirements_file: Path | None = None,
) -> tuple[bool, str]:
    """Install packages using pip. Returns (success, output_or_error_message)."""
    if requirements_file and requirements_file.is_file():
        cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
    else:
        cmd = [sys.executable, "-m", "pip", "install", *specs]

    try:
        # Bound both attempts so a stalled installer cannot strand startup.
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=PIP_TIMEOUT_SECONDS)
        if proc.returncode == 0:
            importlib.invalidate_caches()
            return True, proc.stdout

        # If failed with permission denied, retry with --user.
        error_output = (proc.stderr or "") + (proc.stdout or "")
        if any(err in error_output for err in ("Access is denied", "Permission denied", "[WinError 5]")):
            proc = subprocess.run(
                cmd + ["--user"], capture_output=True, text=True, timeout=PIP_TIMEOUT_SECONDS
            )
            if proc.returncode == 0:
                importlib.invalidate_caches()
                return True, proc.stdout
        return False, proc.stderr or proc.stdout or f"pip exited with status {proc.returncode}"
    except subprocess.TimeoutExpired:
        return False, f"pip installation timed out after {PIP_TIMEOUT_SECONDS} seconds."
    except OSError as exc:
        return False, f"Unable to run pip installation: {exc}"


def ensure_dependencies(
    app_dir: Path | None = None,
    interactive: bool = True,
    dependencies: Sequence[Dependency] = DEFAULT_DEPENDENCIES,
) -> bool:
    """Ensure all required dependencies are installed.

    Interactive calls may offer installation with the user's consent. A
    noninteractive call only checks availability. Returns False on missing
    dependencies, declined installation, or installation failure; callers own
    their application's exit status.
    """
    if getattr(sys, "frozen", False):
        return True

    if app_dir is None:
        app_dir = Path(__file__).resolve().parent

    missing = check_missing_dependencies(dependencies)
    if not missing:
        return True
    if not interactive:
        return False

    req_file = app_dir / "requirements.txt"
    gui_result = _ensure_dependencies_gui(missing, dependencies, req_file)
    if gui_result is not None:
        return gui_result
    return _ensure_dependencies_cli(missing, dependencies, req_file)


def _ensure_dependencies_gui(
    missing: Sequence[Dependency], dependencies: Sequence[Dependency], req_file: Path
) -> bool | None:
    """Own the temporary Tk root; None requests the terminal fallback."""
    try:
        import tkinter as tk
        from tkinter import messagebox, ttk
    except ImportError:
        return None
    try:
        root = tk.Tk()
    except tk.TclError:
        # Tk can be installed while no graphical display is available.
        return None

    try:
        root.withdraw()
        root.attributes("-topmost", True)
        if not _confirm_gui_installation(root, messagebox, missing):
            return False
        success, message = _install_with_progress(root, tk, ttk, missing, req_file)
        return _report_gui_installation(root, messagebox, dependencies, success, message)
    finally:
        root.destroy()


def _confirm_gui_installation(root: Any, messagebox: Any, missing: Sequence[Dependency]) -> bool:
    items_text = "\n".join(f"  • {dep.name} ({dep.description})" for dep in missing)
    prompt_text = (
        "Reflector Messenger requires additional component(s) to run:\n\n"
        f"{items_text}\n\n"
        "Would you like to download and install them now automatically?"
    )
    if messagebox.askyesno(
        "Reflector Messenger - Missing Components", prompt_text, parent=root
    ):
        return True
    messagebox.showinfo(
        "Reflector Messenger",
        "The application cannot start without the required components.\n\n"
        "To install them manually, run:\n"
        "python -m pip install -r requirements.txt",
        parent=root,
    )
    return False


def _create_installation_progress(root: Any, tk: Any, ttk: Any) -> tuple[Any, Any]:
    """Build the temporary progress UI without starting installation."""
    progress_win = tk.Toplevel(root)
    progress_win.title("Installing Components")
    progress_win.geometry("420x150")
    progress_win.resizable(False, False)
    progress_win.attributes("-topmost", True)

    progress_win.update_idletasks()
    x = (progress_win.winfo_screenwidth() - 420) // 2
    y = (progress_win.winfo_screenheight() - 150) // 2
    progress_win.geometry(f"420x150+{x}+{y}")

    frame = ttk.Frame(progress_win, padding=20)
    frame.pack(fill="both", expand=True)
    lbl_title = ttk.Label(
        frame,
        text="Downloading and installing required components...",
        font=("Segoe UI", 10, "bold"),
    )
    lbl_title.pack(anchor="w", pady=(0, 6))
    lbl_desc = ttk.Label(
        frame,
        text="This may take a moment depending on your internet connection.",
        font=("Segoe UI", 9),
    )
    lbl_desc.pack(anchor="w", pady=(0, 12))
    progress_bar = ttk.Progressbar(frame, mode="indeterminate")
    progress_bar.pack(fill="x", pady=(0, 10))
    progress_bar.start(15)
    return progress_win, progress_bar


def _install_missing_dependencies(missing: Sequence[Dependency], req_file: Path) -> tuple[bool, str]:
    return install_dependencies(
        [dep.pip_spec for dep in missing],
        requirements_file=req_file if req_file.is_file() else None,
    )


def _install_with_progress(
    root: Any, tk: Any, ttk: Any, missing: Sequence[Dependency], req_file: Path
) -> tuple[bool, str]:
    """Run pip off the UI thread and always retire the progress window."""
    progress_win, progress_bar = _create_installation_progress(root, tk, ttk)
    result = (False, "Dependency installation did not return a result.")

    def install() -> None:
        nonlocal result
        try:
            result = _install_missing_dependencies(missing, req_file)
        except Exception as exc:
            # Failures outside subprocess creation must reach the GUI too.
            result = False, f"Dependency installation failed: {exc}"

    try:
        worker = threading.Thread(target=install)
        worker.start()
        while worker.is_alive():
            root.update()
            time.sleep(0.05)
        return result
    finally:
        progress_bar.stop()
        progress_win.destroy()


def _report_gui_installation(
    root: Any, messagebox: Any, dependencies: Sequence[Dependency], success: bool, message: str
) -> bool:
    if not success:
        messagebox.showerror(
            "Installation Failed",
            "Failed to automatically install the required components.\n\n"
            f"Details:\n{message}\n\n"
            "Please check your internet connection or install manually using:\n"
            "python -m pip install -r requirements.txt",
            parent=root,
        )
        return False
    if check_missing_dependencies(dependencies):
        messagebox.showerror(
            "Installation Incomplete",
            "Installation finished, but some components could not be loaded.\n\n"
            "Please try running:\npython -m pip install -r requirements.txt",
            parent=root,
        )
        return False
    return True


def _confirm_cli_installation() -> bool:
    if sys.stdin is None or not sys.stdin.isatty():
        print("Installation requires confirmation. Run: python -m pip install -r requirements.txt")
        return False
    try:
        answer = input("Would you like to install them now? [Y/n]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    if answer not in ("", "y", "yes"):
        print("Aborted. You can install manually with: python -m pip install -r requirements.txt")
        return False
    return True


def _ensure_dependencies_cli(
    missing: Sequence[Dependency], dependencies: Sequence[Dependency], req_file: Path
) -> bool:
    """Offer a terminal installation and report the verified outcome."""
    print("=======================================================")
    print("Reflector Messenger - Missing Dependencies")
    print("=======================================================")
    for dep in missing:
        print(f"  - {dep.name}: {dep.description}")
    print()

    if not _confirm_cli_installation():
        return False

    print("Installing dependencies via pip...")
    success, msg = _install_missing_dependencies(missing, req_file)
    if not success:
        print(f"[ERROR] Failed to install dependencies:\n{msg}")
        print("Please check your internet connection or run: python -m pip install -r requirements.txt")
        return False
    if check_missing_dependencies(dependencies):
        print("[ERROR] Installation finished, but required components could not be loaded.")
        return False
    print("Dependencies successfully installed!")
    return True


if __name__ == "__main__":
    if "--check" in sys.argv:
        missing = check_missing_dependencies()
        if missing:
            print("Missing dependencies:")
            for m in missing:
                print(f"  - {m.name} ({m.pip_spec})")
            sys.exit(1)
        else:
            print("All dependencies are satisfied.")
            sys.exit(0)

    sys.exit(0 if ensure_dependencies(interactive=True) else 1)
