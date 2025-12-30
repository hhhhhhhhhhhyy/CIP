import sys
import os
from pathlib import Path

def resource_path(relative_path: str) -> str:
    """
    Get absolute path to resource, works for dev and for PyInstaller.
    """
    # PyInstaller bundled runtime path
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)

    # dev environment
    return os.path.join(os.path.abspath("."), relative_path)


def load_prompt(filename: str) -> str:
    """
    Load the content of a prompt text file and return it as a string.
    Automatically supports PyInstaller bundled files.
    """
    real_path = resource_path(filename)
    prompt_path = Path(real_path)

    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    with prompt_path.open("r", encoding="utf-8") as f:
        return f.read()