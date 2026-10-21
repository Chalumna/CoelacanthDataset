#!/usr/bin/env python3
"""Extract frames from a video of the dataset, in two steps.

The video must already be a row of metadata/images.csv (for example
COEL-000014). Frames inherit its metadata (creator, licence, source_group,
availability...) and get `parent_id` = the video's image_id.

    1. Stage candidate frames (nothing is added to the dataset yet):

        python scripts/extract_frames.py stage --parent COEL-000014 --count 30 --dry-run
        python scripts/extract_frames.py stage --parent COEL-000014 --count 30

    2. Open staging/COEL-000014/ and sort the frames:

       * video where every frame shows a coelacanth: just DELETE the bad
         frames and leave the others where they are (they are all positives);
       * video mixing shots with and without a coelacanth: move each frame to
         positive/ (coelacanth visible) or negative/ (no coelacanth: empty
         scene, other fish, shark...), and DELETE the rest (title cards...).
         As soon as one frame is in positive/ or negative/, every frame must
         be sorted or deleted.

    3. Commit what is left:

        python scripts/extract_frames.py commit --parent COEL-000014

       Positives get the next free COEL-xxxxxx numbers and are appended to
       metadata/images.csv. Negatives get NEG-xxxxxx numbers and are appended
       to metadata/negatives.csv. Files go to the folder matching the parent's
       `availability` (public -> data/unrestricted, public_with_conditions ->
       data/with-conditions, local_only -> data/local-only).

Staging first means rejected frames never consume an ID or touch the CSV files.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "raw"
STAGING_DIR = ROOT / "staging"
IMAGES_CSV = ROOT / "metadata" / "images.csv"
NEGATIVES_CSV = ROOT / "metadata" / "negatives.csv"

# availability value -> folder holding the files
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
INHERITED = ["species", "live_status", "source_url", "creator", "license",
             "license_url", "rights_verified", "source_group", "verified_on",
             "availability"]

NEG_COLUMNS = [
    "image_id", "filename", "source_url", "creator", "license", "license_url",
    "rights_verified", "sha256", "source_group", "parent_id", "verified_on",
    "modifications", "availability", "description",
]
NEG_INHERITED = ["source_url", "creator", "license", "license_url",
                 "rights_verified", "source_group", "verified_on", "availability"]
NEG_DESCRIPTION = "No coelacanth visible"

PLACEHOLDER_HASH = "REPLACE_WITH_SHA256"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
ID_WIDTH = 6
FRAME_FILE = re.compile(r"^(COEL-\d+)_f(\d+)\.(jpg|png)$")
FRAME_NOTE = re.compile(r"^Frame (\d+) ")


def fail(message: str):
    sys.exit(f"error: {message}")


# --------------------------------------------------------------------- CSV --
def read_csv(path: Path, columns, required: bool = True):
    if not path.exists():
        if required:
            fail(f"{path} not found")
        return []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header != columns:
            missing = [c for c in columns if c not in (header or [])]
            extra = [c for c in (header or []) if c not in columns]
            fail(f"{path.name}: header differs from the expected schema "
                 f"(missing: {missing or 'none'}, unexpected: {extra or 'none'})")
        rows = []
        for line_number, row in enumerate(reader, start=2):
            if not any(cell.strip() for cell in row):
                continue
            if len(row) != len(columns):
                fail(f"{path.name}, line {line_number}: expected {len(columns)} fields, "
                     f"found {len(row)}. A field containing a comma must be wrapped "
                     "in double quotes.")
            rows.append(dict(zip(columns, row)))
    return rows


def write_csvs(tables) -> None:
    """Write several CSV files: temp files first, then replace. Keeps .bak copies."""
    temps = []
    try:
        for path, columns, rows in tables:
            fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
            with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
            temps.append((path, tmp))
    except BaseException:
        for _, tmp in temps:
            Path(tmp).unlink(missing_ok=True)
        raise
    for path, tmp in temps:
        if path.exists():
            shutil.copy2(path, path.with_name(path.name + ".bak"))
        os.replace(tmp, path)


# ------------------------------------------------------------------ helpers --
def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_placeholder(value: str) -> bool:
    return value.strip() in ("", PLACEHOLDER_HASH)


def next_number(prefix: str, rows) -> int:
    """Highest <prefix>-number found in the CSV, dataset folders or raw/, plus one."""
    pattern = re.compile(rf"{prefix}-(\d+)")
    numbers = []
    for row in rows:
        match = pattern.fullmatch(row["image_id"].strip())
        if match:
            numbers.append(int(match.group(1)))
    for folder in (*FOLDERS.values(), RAW_DIR):
        if folder.is_dir():
            for path in folder.iterdir():
                match = pattern.match(path.stem)
                if match:
                    numbers.append(int(match.group(1)))
    return max(numbers, default=0) + 1


def get_parent(rows, parent_id: str):
    parent = next((r for r in rows if r["image_id"] == parent_id), None)
    if parent is None:
        fail(f"{parent_id} not found in images.csv")
    if Path(parent["filename"]).suffix.lower() not in VIDEO_EXTENSIONS:
        fail(f"{parent_id} is not a video ({parent['filename']})")
    if parent["availability"] not in FOLDERS:
        fail(f"{parent_id}: unknown availability '{parent['availability']}' "
             f"(expected one of {', '.join(FOLDERS)})")
    if not parent["source_group"].strip():
        fail(f"{parent_id} has no source_group: fill it first, frames inherit it "
             "and it is what keeps them together in train/test splits")
    if parent["availability"] != "local_only":
        if parent["rights_verified"].strip().lower() != "true":
            fail(f"{parent_id}: rights_verified is not true, so frames cannot be published")
        if "ND" in re.findall(r"[A-Z0-9]+", parent["license"].upper()):
            fail(f"{parent_id}: licence '{parent['license']}' forbids derivatives (ND); "
                 "frames are adaptations. Keep it local_only or ask for permission.")
    return parent


def locate_video(parent) -> Path:
    candidates = [RAW_DIR / parent["filename"], FOLDERS[parent["availability"]] / parent["filename"]]
    found = [p for p in candidates if p.exists()]
    if not found:
        fail(f"video not found: looked in {', '.join(str(c.relative_to(ROOT)) for c in candidates)}")
    if len(found) == 2 and sha256_of(found[0]) != sha256_of(found[1]):
        fail("raw/ and the dataset folder hold two different files with this name")
    return found[0]


def select_indices(total, fps, start_s, end_s, every_s, every_frames, count):
    first = max(0, int(round(start_s * fps)))
    last = total - 1 if end_s is None else min(total - 1, int(round(end_s * fps)))
    if first > last:
        fail("the selected time range is empty")
    if count:
        indices = [(first + last) // 2] if count == 1 else [
            round(first + i * (last - first) / (count - 1)) for i in range(count)]
    elif every_frames:
        indices = list(range(first, last + 1, every_frames))
    else:
        indices = list(range(first, last + 1, max(1, int(round(every_s * fps)))))
    return sorted(set(indices))


def sharpness(frame) -> float:
    return float(cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


# -------------------------------------------------------------------- stage --
def cmd_stage(args) -> None:
    rows = read_csv(IMAGES_CSV, COLUMNS)
    neg_rows = read_csv(NEGATIVES_CSV, NEG_COLUMNS, required=False)
    parent = get_parent(rows, args.parent)
    video = locate_video(parent)
    video_hash = sha256_of(video)
    if not is_placeholder(parent["sha256"]) and parent["sha256"].strip() != video_hash:
        fail("the video does not match the sha256 recorded in images.csv")

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        fail("OpenCV cannot open this video; convert it first, e.g. "
             "ffmpeg -i input.avi -c:v libx264 -crf 18 output.mp4")
    fps = args.fps or cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if not fps or fps != fps or fps <= 0:
        fail("unknown frame rate: pass --fps")
    if total <= 0:
        cap = cv2.VideoCapture(str(video))
        total = 0
        while cap.grab():
            total += 1
        cap.release()

    indices = select_indices(total, fps, args.start, args.end,
                             args.every_seconds, args.every_frames, args.count)
    stage_dir = STAGING_DIR / args.parent
    ext = args.format

    print(f"Parent    : {args.parent} ({video.relative_to(ROOT)})")
    print(f"Video     : {total} frames at {fps:g} fps ({total / fps:.1f} s)")
    print(f"Selection : {len(indices)} candidate frames -> {stage_dir.relative_to(ROOT)}/")
    print(f"After commit: {FOLDERS[parent['availability']].relative_to(ROOT)}/ "
          f"(availability: {parent['availability']}), positives start at "
          f"COEL-{next_number('COEL', rows):0{ID_WIDTH}d}, negatives at "
          f"NEG-{next_number('NEG', neg_rows):0{ID_WIDTH}d}")
    if args.dry_run:
        print("Dry run: nothing written.")
        return

    if stage_dir.exists() and any(stage_dir.iterdir()):
        if not args.clean:
            fail(f"{stage_dir.relative_to(ROOT)}/ is not empty: commit it, or use --clean to discard it")
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    wanted, last_wanted = set(indices), indices[-1]
    staged, skipped = [], []
    cap = cv2.VideoCapture(str(video))
    index = 0
    while index <= last_wanted:
        if not cap.grab():
            print(f"warning: video ended early at frame {index}")
            break
        if index in wanted:
            ok, frame = cap.retrieve()
            if ok and args.min_sharpness and sharpness(frame) < args.min_sharpness:
                skipped.append(index)
            elif ok:
                path = stage_dir / f"{args.parent}_f{index:06d}.{ext}"
                params = ([cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality] if ext == "jpg"
                          else [cv2.IMWRITE_PNG_COMPRESSION, 6])
                if not cv2.imwrite(str(path), frame, params):
                    fail(f"could not write {path}")
                staged.append(index)
        index += 1
    cap.release()

    (stage_dir / "positive").mkdir(exist_ok=True)
    (stage_dir / "negative").mkdir(exist_ok=True)
    (stage_dir / "manifest.json").write_text(json.dumps({
        "parent_id": args.parent, "video": video.name, "video_sha256": video_hash,
        "fps": fps, "total_frames": total, "format": ext,
        "jpeg_quality": args.jpeg_quality, "frames": staged,
    }, indent=2))
    print(f"Staged {len(staged)} frames." + (f" Skipped {len(skipped)} blurry ones." if skipped else ""))
    print(f"Next: open {stage_dir.relative_to(ROOT)}/ and delete the frames you do not want.")
    print("If the video also has shots WITHOUT a coelacanth, move each frame to positive/ or negative/.")
    print(f"Then run: python scripts/extract_frames.py commit --parent {args.parent}")


# ------------------------------------------------------------------- commit --
def collect_frames(stage_dir: Path, parent_id: str, ext: str):
    """Return (positives, negatives) as sorted lists of (frame_index, path)."""
    def scan(folder: Path):
        found = []
        if folder.is_dir():
            for path in sorted(folder.iterdir()):
                m = FRAME_FILE.match(path.name)
                if path.is_file() and m and m.group(1) == parent_id and m.group(3) == ext:
                    found.append((int(m.group(2)), path))
        return sorted(found)

    top = scan(stage_dir)
    positives = scan(stage_dir / "positive")
    negatives = scan(stage_dir / "negative")
    if positives or negatives:
        if top:
            fail(f"{len(top)} frame(s) are still directly in {stage_dir.relative_to(ROOT)}/: "
                 "move each one to positive/ or negative/, or delete it")
        both = {i for i, _ in positives} & {i for i, _ in negatives}
        if both:
            fail(f"frames found in both positive/ and negative/: {sorted(both)}")
        return positives, negatives
    return top, []  # nothing sorted: every remaining frame is a positive


def already_extracted(rows, parent_id: str) -> set[int]:
    return {int(m.group(1)) for r in rows if r["parent_id"] == parent_id
            and (m := FRAME_NOTE.match(r["modifications"]))}


def cmd_commit(args) -> None:
    rows = read_csv(IMAGES_CSV, COLUMNS)
    neg_rows = read_csv(NEGATIVES_CSV, NEG_COLUMNS, required=False)
    parent = get_parent(rows, args.parent)
    stage_dir = STAGING_DIR / args.parent
    manifest_path = stage_dir / "manifest.json"
    if not manifest_path.exists():
        fail(f"nothing staged for {args.parent} (run the stage command first)")
    manifest = json.loads(manifest_path.read_text())

    video = locate_video(parent)
    video_hash = sha256_of(video)
    if video_hash != manifest["video_sha256"]:
        fail("the video changed since the frames were staged")
    if not is_placeholder(parent["sha256"]) and parent["sha256"].strip() != video_hash:
        fail("the video does not match the sha256 recorded in images.csv")

    ext, fps, quality = manifest["format"], manifest["fps"], manifest["jpeg_quality"]
    positives, negatives = collect_frames(stage_dir, args.parent, ext)

    done = already_extracted(rows, args.parent) | already_extracted(neg_rows, args.parent)
    skipped = sorted({i for i, _ in positives + negatives} & done)
    if skipped:
        print(f"warning: frames already in the dataset are skipped: {skipped}")
    positives = [(i, p) for i, p in positives if i not in done]
    negatives = [(i, p) for i, p in negatives if i not in done]
    if not positives and not negatives:
        fail("no frame left to commit")

    dest = FOLDERS[parent["availability"]]
    pos_number, neg_number = next_number("COEL", rows), next_number("NEG", neg_rows)
    note_tail = (f"; saved as JPEG (quality {quality})" if ext == "jpg" else "; saved as PNG")

    if positives:
        print(f"Positives: {len(positives)} frames, COEL-{pos_number:0{ID_WIDTH}d} -> "
              f"COEL-{pos_number + len(positives) - 1:0{ID_WIDTH}d}")
    if negatives:
        print(f"Negatives: {len(negatives)} frames, NEG-{neg_number:0{ID_WIDTH}d} -> "
              f"NEG-{neg_number + len(negatives) - 1:0{ID_WIDTH}d}")
    print(f"Destination: {dest.relative_to(ROOT)}/")
    if args.dry_run:
        print("Dry run: nothing written.")
        return

    dest.mkdir(parents=True, exist_ok=True)
    moved, new_pos, new_neg = [], [], []

    def add(frame_index, path, prefix, number, kind):
        image_id = f"{prefix}-{number:0{ID_WIDTH}d}"
        target = dest / f"{image_id}.{ext}"
        if target.exists():
            fail(f"{target} already exists")
        shutil.move(str(path), str(target))
        moved.append((target, path))
        columns, inherited = (COLUMNS, INHERITED) if kind == "positive" else (NEG_COLUMNS, NEG_INHERITED)
        row = {c: "" for c in columns}
        row.update({k: parent[k] for k in inherited})
        row.update({
            "image_id": image_id,
            "filename": target.name,
            "sha256": sha256_of(target),
            "parent_id": args.parent,
            "modifications": (f"Frame {frame_index} (t = {frame_index / fps:.3f} s) "
                              f"taken from {parent['filename']}{note_tail}"),
        })
        if kind == "negative":
            row["description"] = NEG_DESCRIPTION
        return row

    try:
        for frame_index, path in positives:
            new_pos.append(add(frame_index, path, "COEL", pos_number, "positive"))
            pos_number += 1
        for frame_index, path in negatives:
            new_neg.append(add(frame_index, path, "NEG", neg_number, "negative"))
            neg_number += 1
        if is_placeholder(parent["sha256"]):
            parent["sha256"] = video_hash
            print(f"Recorded the sha256 of {parent['filename']} in images.csv")
        tables = [(IMAGES_CSV, COLUMNS, rows + new_pos)]
        if new_neg:
            tables.append((NEGATIVES_CSV, NEG_COLUMNS, neg_rows + new_neg))
        write_csvs(tables)
    except BaseException:
        for target, original in moved:  # roll back: put the frames back in staging
            shutil.move(str(target), str(original))
        raise

    shutil.rmtree(stage_dir)
    print(f"Done: {len(new_pos)} positive(s) added to images.csv"
          + (f", {len(new_neg)} negative(s) added to negatives.csv." if new_neg else "."))
    print("Next: python scripts/build_licenses.py, python scripts/check_repo.py, then commit.")


# --------------------------------------------------------------------- main --
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("stage", help="write candidate frames to staging/ (dataset untouched)")
    s.add_argument("--parent", required=True, help="image_id of the video, e.g. COEL-000014")
    sel = s.add_mutually_exclusive_group()
    sel.add_argument("--count", type=int, help="N frames evenly spaced over the selected range")
    sel.add_argument("--every-seconds", type=float, help="one frame every S seconds of video time (default 1.0)")
    sel.add_argument("--every-frames", type=int, help="one frame every N frames")
    s.add_argument("--start", type=float, default=0.0, help="start time in seconds")
    s.add_argument("--end", type=float, default=None, help="end time in seconds")
    s.add_argument("--fps", type=float, help="override the frame rate used for timestamps")
    s.add_argument("--format", choices=["jpg", "png"], default="jpg")
    s.add_argument("--jpeg-quality", type=int, default=95)
    s.add_argument("--min-sharpness", type=float, default=0.0,
                   help="skip frames whose Laplacian variance is below this value")
    s.add_argument("--clean", action="store_true", help="discard frames already staged for this parent")
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(func=cmd_stage)

    c = sub.add_parser("commit", help="add the staged frames that are left to the dataset")
    c.add_argument("--parent", required=True)
    c.add_argument("--dry-run", action="store_true")
    c.set_defaults(func=cmd_commit)
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "stage" and args.every_seconds is None and not args.every_frames and not args.count:
        args.every_seconds = 1.0
    args.func(args)


if __name__ == "__main__":
    main()
