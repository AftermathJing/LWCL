from __future__ import annotations

import re
import subprocess
from pathlib import Path


FORBIDDEN_SUFFIXES = {".pem", ".key", ".p12", ".pfx", ".pt", ".pth", ".ckpt", ".safetensors"}
FORBIDDEN_PARTS = {".idea", ".venv", "__pycache__", "data/raw", "data/processed", "outputs", "checkpoints"}
SECRET_PATTERN = re.compile(
    r"(api[_-]?key|access[_-]?token|secret[_-]?key|BEGIN (RSA |OPENSSH )?PRIVATE" + r" KEY)",
    re.I,
)


def main() -> None:
    tracked = subprocess.check_output(["git", "ls-files"], text=True, encoding="utf-8").splitlines()
    failures: list[str] = []
    for relative in tracked:
        normalized = relative.replace("\\", "/")
        path = Path(relative)
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            failures.append(f"forbidden artifact: {relative}")
        if any(part in normalized for part in FORBIDDEN_PARTS):
            failures.append(f"forbidden path: {relative}")
        if path.is_file() and path.stat().st_size < 2_000_000:
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if SECRET_PATTERN.search(content):
                failures.append(f"possible secret marker: {relative}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"pre-push audit passed for {len(tracked)} tracked files")


if __name__ == "__main__":
    main()
