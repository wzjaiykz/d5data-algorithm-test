#!/usr/bin/env python3
"""Rebuild report/report.pdf from report/report.md.

Requirements:
  * Pandoc 3.x (https://pandoc.org)
  * Typst 0.13+ (https://typst.app)

Both must be on PATH, or pass --pandoc/--typst. The committed report.pdf is the
authoritative artifact; this script is for reproducibility and small layout
tweaks. It also avoids the UTF-8/GBK corruption that PowerShell's
Get-Content | Set-Content pipeline can introduce on Chinese Windows.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "report"


def find_tool(explicit: str | None, name: str, env_var: str) -> str:
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise SystemExit(f"{name} not found at {path}")
        return str(path)
    env_value = os.environ.get(env_var)
    if env_value:
        return env_value
    found = shutil.which(name)
    if found:
        return found
    raise SystemExit(
        f"{name} not found. Install it, put it on PATH, or pass --{name} / set {env_var}."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pandoc", default=None)
    parser.add_argument("--typst", default=None)
    parser.add_argument(
        "--png",
        action="store_true",
        help="Also render page images and print the page count.",
    )
    args = parser.parse_args(argv)

    pandoc = find_tool(args.pandoc, "pandoc", "PANDOC")
    typst = find_tool(args.typst, "typst", "TYPST")

    markdown = REPORT_DIR / "report.md"
    body_path = REPORT_DIR / "report_body.typ"
    preamble_path = REPORT_DIR / "preamble.typ"
    typst_path = REPORT_DIR / "report.typ"
    pdf_path = REPORT_DIR / "report.pdf"

    if not markdown.exists() or not preamble_path.exists():
        raise SystemExit("report/report.md or report/preamble.typ is missing")

    subprocess.run(
        [pandoc, str(markdown), "-t", "typst", "-o", str(body_path)],
        check=True,
    )
    preamble = preamble_path.read_text(encoding="utf-8").rstrip()
    body = body_path.read_text(encoding="utf-8").lstrip("\ufeff")
    typst_path.write_text(preamble + "\n\n" + body, encoding="utf-8", newline="\n")

    subprocess.run([typst, "compile", str(typst_path), str(pdf_path)], check=True)
    print(f"wrote {pdf_path} ({pdf_path.stat().st_size} bytes)")

    if args.png:
        for old in REPORT_DIR.glob("report-*.png"):
            old.unlink()
        subprocess.run(
            [typst, "compile", str(typst_path), str(REPORT_DIR / "report-{p}.png"), "--ppi", "110"],
            check=True,
        )
        pages = sorted(REPORT_DIR.glob("report-*.png"))
        print(f"rendered {len(pages)} page image(s): {[p.name for p in pages]}")
        if len(pages) > 2:
            print("WARNING: report is longer than 2 pages", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
