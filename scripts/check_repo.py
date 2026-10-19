#!/usr/bin/env python3
"""Check the integrity of the dataset repository.

    python scripts/check_repo.py              # report errors and warnings
    python scripts/check_repo.py --fix-hashes # also fill missing sha256 values

Exit code 1 if there is at least one error, so it can run in a pre-commit hook
or in CI. Warnings never fail the check.

What is checked
---------------
* metadata/images.csv: exact header, well-formed rows, unique COEL ids,
  filenames, allowed values, licence/licence_url consistency;
* availability rules: public -> CC0 / public domain only; public_with_conditions
  -> verified rights, source and creator; ND images must be unmodified;
* files: every public file exists in the folder matching its availability,
  no orphan file, sha256 matches (placeholders can be filled with --fix-hashes);
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
IMAGES_CSV = ROOT / "metadata" / "images.csv"

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
PLACEHOLDER_HASH = "REPLACE_WITH_SHA256"
SPECIES = {"Latimeria chalumnae", "Latimeria menadoensis"}
LIVE_STATUS = {"live"}
ID_RE = re.compile(r"^COEL-\d{6}$")
FILE_RE = re.compile(r"^(COEL-\d{6})\.[A-Za-z0-9]+$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
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


def read_rows(report: Report):
    if not IMAGES_CSV.exists():
        report.error("metadata/images.csv not found")
        return None
    with open(IMAGES_CSV, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header != COLUMNS:
            missing = [c for c in COLUMNS if c not in (header or [])]
            extra = [c for c in (header or []) if c not in COLUMNS]
            report.error(f"images.csv header differs from the expected schema "
                         f"(missing: {missing or 'none'}, unexpected: {extra or 'none'}, "
                         f"or wrong order)")
            return None
        rows = []
        for line_number, row in enumerate(reader, start=2):
            if not any(cell.strip() for cell in row):
                report.warn(f"images.csv line {line_number}: blank row")
                continue
            if len(row) != len(COLUMNS):
                report.error(f"images.csv line {line_number}: {len(row)} fields instead of "
                             f"{len(COLUMNS)} (a field with a comma must be in double quotes)")
                return None
            rows.append(dict(zip(COLUMNS, row)))
    return rows


def check_rows(rows, report: Report, fix_hashes: bool) -> bool:
    """Return True if some sha256 values were filled (rows modified in place)."""
    ids = {r["image_id"] for r in rows}
    seen: set[str] = set()
    changed = False
    for r in rows:
        iid = r["image_id"]
        tag = iid or "<no id>"
        if not ID_RE.match(iid):
            report.error(f"{tag}: image_id must look like COEL-000001")
        if iid in seen:
            report.error(f"{tag}: duplicate image_id")
        seen.add(iid)
        if not r["filename"].startswith(iid + "."):
            report.error(f"{tag}: filename '{r['filename']}' does not start with the image_id")
        if r["species"] not in SPECIES:
            report.error(f"{tag}: species '{r['species']}' (expected one of {sorted(SPECIES)})")
        if r["live_status"] not in LIVE_STATUS:
            report.warn(f"{tag}: live_status '{r['live_status']}' (only {sorted(LIVE_STATUS)} expected)")
        availability = r["availability"]
        if availability not in FOLDERS:
            report.error(f"{tag}: availability '{availability}' (expected one of {', '.join(FOLDERS)})")
            continue
        if r["parent_id"] and r["parent_id"] not in ids:
            report.error(f"{tag}: parent_id {r['parent_id']} does not exist")
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

        # files and hashes
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


def check_orphans(rows, report: Report) -> None:
    expected = {(r["availability"], r["filename"]) for r in rows if r["availability"] in FOLDERS}
    for availability, folder in FOLDERS.items():
        if not folder.is_dir():
            continue
        for path in sorted(folder.iterdir()):
            if path.name in IGNORED_FILES or path.is_dir():
                continue
            if (availability, path.name) not in expected:
                elsewhere = [a for a, n in expected if n == path.name]
                hint = f" (listed with availability '{elsewhere[0]}')" if elsewhere else " (not listed in images.csv)"
                report.error(f"orphan file {path.relative_to(ROOT)}{hint}")


def check_git(rows, report: Report) -> None:
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
    for r in rows:
        if r["availability"] != "local_only" or not ID_RE.match(r["image_id"]):
            continue
        out = git("log", "--all", "--format=%h %s", "--", f":(glob)**/{r['image_id']}.*").stdout.strip()
        if out:
            commits = ", ".join(line.split(" ", 1)[0] for line in out.splitlines())
            report.warn(f"{r['image_id']} is local_only but is in the Git history (commits {commits}): "
                        "deleting a file does not remove it from past commits")


def write_rows(rows) -> None:
    fd, tmp = tempfile.mkstemp(dir=IMAGES_CSV.parent, suffix=".tmp")
    with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    shutil.copy2(IMAGES_CSV, IMAGES_CSV.with_name(IMAGES_CSV.name + ".bak"))
    os.replace(tmp, IMAGES_CSV)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fix-hashes", action="store_true",
                        help="fill empty or placeholder sha256 values for files present locally")
    parser.add_argument("--no-git", action="store_true", help="skip the Git checks")
    args = parser.parse_args()

    report = Report()
    rows = read_rows(report)
    if rows is not None:
        changed = check_rows(rows, report, args.fix_hashes)
        check_orphans(rows, report)
        if not args.no_git:
            check_git(rows, report)
        if changed:
            write_rows(rows)
            print("sha256 values filled in metadata/images.csv (previous version kept as images.csv.bak)\n")
        counts: dict[str, int] = {}
        for r in rows:
            counts[r["availability"]] = counts.get(r["availability"], 0) + 1
        print("Rows: " + ", ".join(f"{n} {k}" for k, n in sorted(counts.items())) + f" (total {len(rows)})\n")

    for message in report.errors:
        print(f"ERROR   {message}")
    for message in report.warnings:
        print(f"WARNING {message}")
    print(f"\n{len(report.errors)} error(s), {len(report.warnings)} warning(s)")
    sys.exit(1 if report.errors else 0)


if __name__ == "__main__":
    main()