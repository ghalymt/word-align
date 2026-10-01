"""Print one version's section of CHANGELOG.md (the GitHub release notes).

    python scripts/release_notes.py 2.1.0 > notes.md
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

CHANGELOG = Path(__file__).resolve().parent.parent / "CHANGELOG.md"


def release_notes(version: str, text: str) -> str:
    """The body under ``## <version>`` up to the next ``## `` heading."""
    heading = re.compile(rf"^## {re.escape(version)}(\s|$).*$", re.M)
    match = heading.search(text)
    if match is None:
        raise KeyError(f"CHANGELOG.md has no '## {version}' section")
    rest = text[match.end():]
    following = re.search(r"^## ", rest, re.M)
    return (rest[:following.start()] if following else rest).strip() + "\n"


def unwrap(markdown: str) -> str:
    """Join hard-wrapped lines: GitHub release notes keep every newline."""
    out = []
    for line in markdown.splitlines():
        stripped = line.strip()
        starts_block = (not stripped or stripped.startswith(("- ", "* ", "#", "|", "```"))
                        or stripped[:1].isdigit() and ". " in stripped[:4])
        if out and out[-1].strip() and not out[-1].lstrip().startswith("#") \
                and not starts_block:
            out[-1] = out[-1].rstrip() + " " + stripped
        else:
            out.append(line)
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    try:
        sys.stdout.write(unwrap(release_notes(
            args[0].lstrip("v"), CHANGELOG.read_text(encoding="utf-8"))))
    except KeyError as exc:
        print(f"error: {exc.args[0]}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
