#!/usr/bin/env python3
"""Regenerate the rights tables of metadata/LICENSES.md from the CSV files.

    python scripts/build_licenses.py           # rewrite the generated blocks
    python scripts/build_licenses.py --check   # exit 1 if LICENSES.md is out of date

metadata/images.csv and metadata/negatives.csv are the single source of truth.
Only the text between the markers

    <!-- BEGIN GENERATED:name -->  ...  <!-- END GENERATED:name -->

is rewritten (blocks: unrestricted, with-conditions, local-only, negatives).
Everything else in LICENSES.md is yours to edit. Frames extracted from a video
are grouped on one line per parent video.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMAGES_CSV = ROOT / "metadata" / "images.csv"
NEGATIVES_CSV = ROOT / "metadata" / "negatives.csv"
LICENSES_MD = ROOT / "metadata" / "LICENSES.md"

FOLDER_NAMES = {
    "public": "data/unrestricted",
    "public_with_conditions": "data/with-conditions",
    "local_only": "data/local-only",
}
BLOCKS = ["unrestricted", "with-conditions", "local-only", "negatives"]


def read_rows(path: Path):
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8-sig") as f:
        return [r for r in csv.DictReader(f) if any((v or "").strip() for v in r.values())]


# ----------------------------------------------------------------- helpers --
def cell(text: str) -> str:
    return (text or "").replace("|", "\\|").replace("\n", " ").strip()


def link(label: str, url: str) -> str:
    url = (url or "").strip()
    label = cell(label)
    if not url:
        return label
    safe = url.replace(" ", "%20").replace("(", "%28").replace(")", "%29")
    return f"[{label}]({safe})"


def conditions(license_name: str) -> str:
    tokens = re.findall(r"[A-Z0-9]+", license_name.upper())
    parts = []
    if "BY" in tokens:
        parts.append("Attribution")
    if "NC" in tokens:
        parts.append("Non-commercial use only")
    if "SA" in tokens:
        parts.append("ShareAlike")
    if "ND" in tokens:
        parts.append("No derivatives: keep unmodified")
    return "; ".join(parts) or "See the licence"


def id_ranges(ids) -> str:
    """['COEL-000016', 'COEL-000017', 'COEL-000020'] -> 'COEL-000016 to COEL-000017, COEL-000020'."""
    parsed = sorted((int(i.rsplit("-", 1)[1]), i) for i in ids)
    runs, start = [], 0
    for k in range(1, len(parsed) + 1):
        if k == len(parsed) or parsed[k][0] != parsed[k - 1][0] + 1:
            first, last = parsed[start][1], parsed[k - 1][1]
            runs.append(first if first == last else f"{first} to {last}")
            start = k
    return ", ".join(runs)


def make_groups(rows):
    """Standalone rows alone; derived rows (parent_id) grouped by parent."""
    singles, derived = [], {}
    for r in sorted(rows, key=lambda r: r["image_id"]):
        if (r.get("parent_id") or "").strip():
            derived.setdefault(r["parent_id"], []).append(r)
        else:
            singles.append([r])
    groups = singles + list(derived.values())
    return sorted(groups, key=lambda g: g[0]["image_id"])


def files_cell(group, parents, with_folder: bool) -> str:
    folder = FOLDER_NAMES.get(group[0]["availability"], "")
    extensions = sorted({Path(r["filename"]).suffix for r in group})
    if len(group) == 1 and not group[0].get("parent_id"):
        return f"`{folder}/{group[0]['filename']}`" if with_folder else f"`{group[0]['image_id']}`"
    parent_id = group[0]["parent_id"]
    parent_name = parents.get(parent_id, parent_id)
    where = f"`{folder}/`: " if with_folder else ""
    return (f"{where}{id_ranges(r['image_id'] for r in group)} "
            f"({len(group)} item{'s' if len(group) > 1 else ''}, {'/'.join(extensions)}), "
            f"taken from {parent_id} ({parent_name})")


def changes(group, parents) -> str:
    mods = {(r.get("modifications") or "").strip() for r in group} - {""}
    parent_id = (group[0].get("parent_id") or "").strip()
    parent_name = parents.get(parent_id, parent_id)
    if mods and all(m.startswith("Frame ") for m in mods):
        return f"Changes: frames extracted from {parent_name}"
    if len(mods) == 1:
        return f"Changes: {next(iter(mods))}"
    if parent_id:
        return f"Changes: derived from {parent_name} (see the modifications column of the CSV)"
    return ""


def table(header, rows) -> str:
    if not rows:
        return "_None yet._"
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


# ------------------------------------------------------------------ blocks --
def render(images, negatives) -> dict[str, str]:
    parents = {r["image_id"]: r["filename"] for r in images}
    blocks = {}

    rows = []
    for g in make_groups([r for r in images if r["availability"] == "public"]):
        r = g[0]
        rows.append([files_cell(g, parents, True), cell(r["creator"]),
                     link(r["license"], r["license_url"]), link("source", r["source_url"]),
                     cell(r.get("verified_on", ""))])
    blocks["unrestricted"] = table(
        ["Files", "Creator / rights holder", "Legal basis", "Evidence", "Verified on"], rows)

    rows = []
    for g in make_groups([r for r in images if r["availability"] == "public_with_conditions"]):
        r = g[0]
        cond = "; ".join(p for p in (conditions(r["license"]), changes(g, parents)) if p)
        rows.append([files_cell(g, parents, True), cell(r["creator"]), link("source", r["source_url"]),
                     link(r["license"], r["license_url"]), cell(cond), cell(r.get("verified_on", ""))])
    blocks["with-conditions"] = table(
        ["Files", "Creator", "Source", "Licence", "Required conditions", "Verified on"], rows)

    rows = []
    for g in make_groups([r for r in images if r["availability"] == "local_only"]):
        r = g[0]
        rows.append([files_cell(g, parents, False), link("source", r["source_url"]),
                     cell(r["license"] or "unverified"), cell(changes(g, parents))])
    blocks["local-only"] = table(["Items", "Source", "Licence status", "Note"], rows)

    rows = []
    neg_parents = parents
    for g in make_groups(negatives):
        r = g[0]
        cond = "; ".join(p for p in (conditions(r["license"]), changes(g, neg_parents)) if p)
        rows.append([files_cell(g, neg_parents, True), cell(r["creator"]),
                     link(r["license"], r["license_url"]), link("source", r["source_url"]), cell(cond)])
    blocks["negatives"] = table(
        ["Files", "Creator", "Licence", "Source", "Required conditions"], rows)
    return blocks


def apply_blocks(text: str, blocks: dict[str, str]) -> str:
    for name in BLOCKS:
        pattern = re.compile(
            rf"(<!-- BEGIN GENERATED:{re.escape(name)} -->)(.*?)(<!-- END GENERATED:{re.escape(name)} -->)",
            re.DOTALL)
        if not pattern.search(text):
            raise ValueError(f"markers for block '{name}' not found in LICENSES.md")
        text = pattern.sub(lambda m: f"{m.group(1)}\n{blocks[name]}\n{m.group(3)}", text)
    return text


def expected_text() -> tuple[str, str]:
    """Return (current text, regenerated text). Raises ValueError if markers are missing."""
    current = LICENSES_MD.read_text(encoding="utf-8")
    new = apply_blocks(current, render(read_rows(IMAGES_CSV), read_rows(NEGATIVES_CSV)))
    return current, new


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="do not write, exit 1 if out of date")
    args = parser.parse_args()
    try:
        current, new = expected_text()
    except (OSError, ValueError) as error:
        sys.exit(f"error: {error}")
    if args.check:
        if current != new:
            sys.exit("metadata/LICENSES.md is out of date: run python scripts/build_licenses.py")
        print("metadata/LICENSES.md is up to date")
        return
    if current == new:
        print("metadata/LICENSES.md already up to date")
        return
    LICENSES_MD.write_text(new, encoding="utf-8")
    print("metadata/LICENSES.md updated")


if __name__ == "__main__":
    main()
