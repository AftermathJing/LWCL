from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_PARTS = {"data", "outputs", "logs", "checkpoints", ".idea", ".venv", "secrets"}
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors", ".pem", ".key", ".p12", ".pfx"}
SECRET_PATTERN = re.compile(r"(api[_-]?key|access[_-]?token|private[_-]?key)\s*[:=]\s*['\"][^'\"]+", re.I)


def main() -> None:
    tracked = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
    failures = []
    for relative in tracked:
        path = Path(relative)
        if FORBIDDEN_PARTS & set(path.parts) or path.suffix.lower() in FORBIDDEN_SUFFIXES:
            failures.append(f"forbidden tracked path: {relative}")
            continue
        full = ROOT / path
        if full.exists() and full.stat().st_size > 20 * 1024 * 1024:
            failures.append(f"tracked file exceeds 20 MiB: {relative}")
        if full.suffix.lower() in {".py", ".yaml", ".yml", ".md", ".toml", ".json"}:
            text = full.read_text(encoding="utf-8", errors="ignore")
            if SECRET_PATTERN.search(text):
                failures.append(f"possible inline secret: {relative}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"pre-push audit passed for {len(tracked)} tracked files")


if __name__ == "__main__":
    main()
