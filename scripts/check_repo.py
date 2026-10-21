#!/usr/bin/env python3
"""Check the integrity of the dataset repository.

    python scripts/check_repo.py              # report errors and warnings
    python scripts/check_repo.py --fix-hashes # also fill missing sha256 values

Exit code 1 if there is at least one error, so it can run in a pre-commit hook
or in CI. Warnings never fail the check.

What is checked
---------------
* metadata/images.csv (COEL-) and metadata/negatives.csv (NEG-, optional):
  exact header, well-formed rows, unique ids, filenames, allowed values,
  licence / licence_url consistency, parent_id references;
* availability rules: public -> CC0 / public domain only; public_with_conditions
  -> verified rights, source and creator; ND images must be unmodified;
* files: every public file exists in the folder matching its availability,
  no orphan file, sha256 matches (placeholders can be filled with --fix-hashes);
* metadata/LICENSES.md is up to date (see scripts/build_licenses.py);
* Git: nothing from data/local-only/, raw/ or staging/ is tracked, those
  folders are ignored, and local_only files never appeared in the history.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

IMAGES_CSV = ROOT / "metadata" / "images.csv"
NEGATIVES_CSV = ROOT / "metadata" / "negatives.csv"

FOLDERS = {
    "public": ROOT / "data" / "unrestricted",
    "public_with_conditions": ROOT / "data" / "with-conditions",
    "local_only": ROOT / "data" / "local-only",
}
COLUMNS = [
    "image_id", "filename", "species", "live_status", "source_url", "creator",
    "license", "license_url", "rights_verified", "sha256", "source_group",
    "parent_id", "verified_on", "modifications", "availability",
]
NEG_COLUMNS = [
    "image_id", "filename", "source_url", "creator", "license", "license_url",
    "rights_verified", "sha256", "source_group", "parent_id", "verified_on",
    "modifications", "availability", "description",
]
PLACEHOLDER_HASH = "REPLACE_WITH_SHA256"
SPECIES = {"Latimeria chalumnae", "Latimeria menadoensis"}
LIVE_STATUS = {"live"}
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
ID_RES = {"positive": re.compile(r"^COEL-\d{6}$"), "negative": re.compile(r"^NEG-\d{6}$")}
IGNORED_FILES = {".DS_Store", ".gitkeep"}


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)


def license_slug(name: str):
    s = name.strip().upper()
    if s.startswith("CC0"):
        return "publicdomain/zero/1.0"
    m = re.fullmatch(r"CC (BY(?:-(?:NC|SA|ND))*) (\d\.\d)", s)
    return f"{m.group(1).lower()}/{m.group(2)}" if m else None


def is_public_domain(name: str) -> bool:
    s = name.strip().upper()
    return s.startswith("CC0") or s.startswith("PUBLIC DOMAIN")


def read_rows(path: Path, columns, report: Report, required: bool):
    if not path.exists():
        if required:
            report.error(f"{path.relative_to(ROOT)} not found")
        return None if required else []
    name = path.relative_to(ROOT)
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header != columns:
            missing = [c for c in columns if c not in (header or [])]
            extra = [c for c in (header or []) if c not in columns]
            report.error(f"{name}: header differs from the expected schema "
                         f"(missing: {missing or 'none'}, unexpected: {extra or 'none'}, or wrong order)")
            return None
        rows = []
        for line_number, row in enumerate(reader, start=2):
            if not any(cell.strip() for cell in row):
                report.warn(f"{name} line {line_number}: blank row")
                continue
            if len(row) != len(columns):
                report.error(f"{name} line {line_number}: {len(row)} fields instead of "
                             f"{len(columns)} (a field with a comma must be in double quotes)")
                return None
            rows.append(dict(zip(columns, row)))
    return rows


def write_rows(path: Path, columns, rows) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    shutil.copy2(path, path.with_name(path.name + ".bak"))
    os.replace(tmp, path)


def check_rows(rows, kind: str, image_ids: set[str], report: Report, fix_hashes: bool) -> bool:
    """Return True if some sha256 values were filled (rows modified in place)."""
    seen: set[str] = set()
    changed = False
    for r in rows:
        iid = r["image_id"]
        tag = iid or "<no id>"
        if not ID_RES[kind].match(iid):
            report.error(f"{tag}: image_id must look like {'COEL' if kind == 'positive' else 'NEG'}-000001")
        if iid in seen:
            report.error(f"{tag}: duplicate image_id")
        seen.add(iid)
        if not r["filename"].startswith(iid + "."):
            report.error(f"{tag}: filename '{r['filename']}' does not start with the image_id")
        if kind == "positive":
            if r["species"] not in SPECIES:
                report.error(f"{tag}: species '{r['species']}' (expected one of {sorted(SPECIES)})")
            if r["live_status"] not in LIVE_STATUS:
                report.warn(f"{tag}: live_status '{r['live_status']}' (only {sorted(LIVE_STATUS)} expected)")
        elif not r["description"].strip():
            report.warn(f"{tag}: empty description")
        availability = r["availability"]
        if availability not in FOLDERS:
            report.error(f"{tag}: availability '{availability}' (expected one of {', '.join(FOLDERS)})")
            continue
        if r["parent_id"] and r["parent_id"] not in image_ids:
            report.error(f"{tag}: parent_id {r['parent_id']} does not exist in images.csv")
        if not r["source_group"].strip():
            report.warn(f"{tag}: empty source_group (needed to keep related images together in splits)")
        if r["verified_on"] and not DATE_RE.match(r["verified_on"]):
            report.error(f"{tag}: verified_on '{r['verified_on']}' is not YYYY-MM-DD")

        lic, url = r["license"], r["license_url"]
        nd = "ND" in re.findall(r"[A-Z0-9]+", lic.upper())
        slug = license_slug(lic)
        if availability != "local_only":
            if r["rights_verified"].lower() != "true":
                report.error(f"{tag}: published but rights_verified is not true")
            for field in ("creator", "source_url", "license", "license_url"):
                if not r[field].strip() or r[field].strip().lower() == "unverified":
                    report.error(f"{tag}: published but {field} is empty or 'unverified'")
            if not r["verified_on"]:
                report.warn(f"{tag}: empty verified_on")
            if availability == "public" and not is_public_domain(lic):
                report.error(f"{tag}: availability 'public' requires CC0 or public domain, found '{lic}'")
            if availability == "public_with_conditions" and is_public_domain(lic):
                report.warn(f"{tag}: licence '{lic}' has no conditions: could be 'public'")
            if nd and (r["modifications"].strip() or r["parent_id"]):
                report.error(f"{tag}: licence '{lic}' forbids derivatives but the image is modified or derived")
            if slug and slug not in url.lower():
                report.warn(f"{tag}: license_url does not look like the licence '{lic}'")
        elif lic.strip().lower() not in ("", "unverified") and slug and slug not in url.lower():
            report.warn(f"{tag}: license_url does not look like the licence '{lic}'")

        path = FOLDERS[availability] / r["filename"]
        placeholder = r["sha256"].strip() in ("", PLACEHOLDER_HASH)
        if not path.exists():
            if availability == "local_only":
                report.warn(f"{tag}: local file missing on this machine ({path.relative_to(ROOT)})")
            else:
                report.error(f"{tag}: file missing ({path.relative_to(ROOT)})")
            if placeholder:
                (report.warn if availability == "local_only" else report.error)(f"{tag}: sha256 is a placeholder")
            continue
        actual = sha256_of(path)
        if placeholder:
            if fix_hashes:
                r["sha256"] = actual
                changed = True
            else:
                report.error(f"{tag}: sha256 is a placeholder (run with --fix-hashes)")
        elif r["sha256"].strip() != actual:
            report.error(f"{tag}: sha256 does not match the file {path.relative_to(ROOT)}")
    return changed


def check_orphans(all_rows, report: Report) -> None:
    expected = {(r["availability"], r["filename"]) for r in all_rows if r["availability"] in FOLDERS}
    for availability, folder in FOLDERS.items():
        if not folder.is_dir():
            continue
        for path in sorted(folder.iterdir()):
            if path.name in IGNORED_FILES or path.is_dir():
                continue
            if (availability, path.name) not in expected:
                elsewhere = [a for a, n in expected if n == path.name]
                hint = f" (listed with availability '{elsewhere[0]}')" if elsewhere else " (not listed in a CSV)"
                report.error(f"orphan file {path.relative_to(ROOT)}{hint}")


def check_licenses_md(report: Report) -> None:
    try:
        import build_licenses
        current, new = build_licenses.expected_text()
    except FileNotFoundError:
        report.warn("metadata/LICENSES.md not found")
        return
    except ValueError as error:
        report.warn(f"metadata/LICENSES.md: {error} (see scripts/build_licenses.py)")
        return
    if current != new:
        report.warn("metadata/LICENSES.md is out of date: run python scripts/build_licenses.py")


def check_git(all_rows, report: Report) -> None:
    if git("rev-parse", "--git-dir").returncode != 0:
        report.warn("not a Git repository: Git checks skipped")
        return
    tracked = set(git("ls-files").stdout.splitlines())
    for prefix in ("data/local-only/", "raw/", "staging/"):
        leaked = sorted(t for t in tracked if t.startswith(prefix))
        if leaked:
            report.error(f"tracked by Git but must stay local: {', '.join(leaked)}")
    for sample in ("data/local-only/x.jpg", "raw/x.mp4", "staging/x/y.jpg"):
        if git("check-ignore", "-q", sample).returncode != 0:
            report.error(f".gitignore does not ignore {sample.split('/x')[0]}/")
    for r in all_rows:
        if r["availability"] != "local_only" or not r["image_id"]:
            continue
        out = git("log", "--all", "--format=%h %s", "--", f":(glob)**/{r['image_id']}.*").stdout.strip()
        if out:
            commits = ", ".join(line.split(" ", 1)[0] for line in out.splitlines())
            report.warn(f"{r['image_id']} is local_only but is in the Git history (commits {commits}): "
                        "deleting a file does not remove it from past commits")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fix-hashes", action="store_true",
                        help="fill empty or placeholder sha256 values for files present locally")
    parser.add_argument("--no-git", action="store_true", help="skip the Git checks")
    args = parser.parse_args()

    report = Report()
    images = read_rows(IMAGES_CSV, COLUMNS, report, required=True)
    negatives = read_rows(NEGATIVES_CSV, NEG_COLUMNS, report, required=False)
    if images is not None and negatives is not None:
        image_ids = {r["image_id"] for r in images}
        changed_img = check_rows(images, "positive", image_ids, report, args.fix_hashes)
        changed_neg = check_rows(negatives, "negative", image_ids, report, args.fix_hashes)
        all_rows = images + negatives
        check_orphans(all_rows, report)
        check_licenses_md(report)
        if not args.no_git:
            check_git(all_rows, report)
        if changed_img:
            write_rows(IMAGES_CSV, COLUMNS, images)
        if changed_neg:
            write_rows(NEGATIVES_CSV, NEG_COLUMNS, negatives)
        if changed_img or changed_neg:
            print("sha256 values filled (previous versions kept as .bak files)\n")
        counts: dict[str, int] = {}
        for r in all_rows:
            counts[r["availability"]] = counts.get(r["availability"], 0) + 1
        print("Rows: " + ", ".join(f"{n} {k}" for k, n in sorted(counts.items()))
              + f" (total {len(all_rows)}: {len(images)} positive, {len(negatives)} negative)\n")

    for message in report.errors:
        print(f"ERROR   {message}")
    for message in report.warnings:
        print(f"WARNING {message}")
    print(f"\n{len(report.errors)} error(s), {len(report.warnings)} warning(s)")
    sys.exit(1 if report.errors else 0)


if __name__ == "__main__":
    main()
