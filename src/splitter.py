from pathlib import Path
import re, sys

ROOT = Path("data")
OTHER = re.compile(r"_(other|virus)_", re.I)

for i in ["train", "val", "test"]:
    pneumonia_dir = ROOT / i / "pneumonia"
    if not pneumonia_dir.exists():
        continue
    for file in pneumonia_dir.glob("*.jpeg"):
        first = OTHER.search(file.name)
        dest = ROOT / i / (
            "bacterial_pneumonia" if first and first.group(1).lower()=="bacteria"
            else "viral_pneumonia"
        )
        dest.mkdir(parents=True, exist_ok=True)
        file.rename(dest / file.name)
    try: pneumonia_dir.rmdir()
    except OSError: pass