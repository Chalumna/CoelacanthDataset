#!/usr/bin/env python3
"""Extract frames from a video of the dataset, in two steps.

The video must already be a row of metadata/images.csv (for example
COEL-000014). Frames inherit its metadata (creator, licence, source_group,
availability...) and get `parent_id` = the video's image_id.

    1. Stage candidate frames (nothing is added to the dataset yet):

        python scripts/extract_frames.py stage --parent COEL-000014 --count 30 --dry-run
        python scripts/extract_frames.py stage --parent COEL-000014 --count 30

    2. Open staging/COEL-000014/ and DELETE the frames you do not want
       (coelacanth not visible, cut off, blurry...).

    3. Commit what is left: frames get the next free COEL-xxxxxx numbers, are
       moved to the folder matching the parent's `availability`
       (public -> data/unrestricted, public_with_conditions ->
       data/with-conditions, local_only -> data/local-only) and are appended
       to metadata/images.csv.

        python scripts/extract_frames.py commit --parent COEL-000014

Staging first means rejected frames never consume an ID or touch the CSV.
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

PLACEHOLDER_HASH = "REPLACE_WITH_SHA256"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
ID_PATTERN = re.compile(r"COEL-(\d+)")
ID_WIDTH = 6
FRAME_FILE = re.compile(r"^(COEL-\d+)_f(\d+)\.(jpg|png)$")
FRAME_NOTE = re.compile(r"^Frame (\d+) ")


def fail(message: str):
    sys.exit(f"error: {message}")


# --------------------------------------------------------------------- CSV --
def read_images():
    if not IMAGES_CSV.exists():
        fail(f"{IMAGES_CSV} not found")
    with open(IMAGES_CSV, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header != COLUMNS:
            missing = [c for c in COLUMNS if c not in (header or [])]
            extra = [c for c in (header or []) if c not in COLUMNS]
            fail(f"images.csv header differs from the expected schema "
                 f"(missing: {missing or 'none'}, unexpected: {extra or 'none'})")
        rows = []
        for line_number, row in enumerate(reader, start=2):
            if not any(cell.strip() for cell in row):
                continue
            if len(row) != len(COLUMNS):
                fail(f"images.csv, line {line_number}: expected {len(COLUMNS)} fields, "
                     f"found {len(row)}. A field containing a comma must be wrapped "
                     "in double quotes.")
            rows.append(dict(zip(COLUMNS, row)))
    return rows


def write_images(rows) -> None:
    """Write atomically, keeping a .bak copy (ignored by Git) of the previous file."""
    fd, tmp_name = tempfile.mkstemp(dir=IMAGES_CSV.parent, suffix=".tmp")
    with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    shutil.copy2(IMAGES_CSV, IMAGES_CSV.with_name(IMAGES_CSV.name + ".bak"))
    os.replace(tmp_name, IMAGES_CSV)


# ------------------------------------------------------------------ helpers --
def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_placeholder(value: str) -> bool:
    return value.strip() in ("", PLACEHOLDER_HASH)


def next_number(rows) -> int:
    """Highest COEL number found in the CSV, in dataset folders or in raw/, plus one."""
    numbers = []
    for row in rows:
        match = ID_PATTERN.fullmatch(row["image_id"].strip())
        if match:
            numbers.append(int(match.group(1)))
    for folder in (*FOLDERS.values(), RAW_DIR):
        if folder.is_dir():
            for path in folder.iterdir():
                match = ID_PATTERN.match(path.stem)
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
    rows = read_images()
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
          f"(availability: {parent['availability']}), starting at "
          f"COEL-{next_number(rows):0{ID_WIDTH}d}")
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

    (stage_dir / "manifest.json").write_text(json.dumps({
        "parent_id": args.parent, "video": video.name, "video_sha256": video_hash,
        "fps": fps, "total_frames": total, "format": ext,
        "jpeg_quality": args.jpeg_quality, "frames": staged,
    }, indent=2))
    print(f"Staged {len(staged)} frames." + (f" Skipped {len(skipped)} blurry ones." if skipped else ""))
    print(f"Next: open {stage_dir.relative_to(ROOT)}/, delete the frames you do not want, then run:")
    print(f"  python scripts/extract_frames.py commit --parent {args.parent}")


# ------------------------------------------------------------------- commit --
def cmd_commit(args) -> None:
    rows = read_images()
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
    already = {int(m.group(1)) for r in rows if r["parent_id"] == args.parent
               and (m := FRAME_NOTE.match(r["modifications"]))}
    candidates = []
    for path in sorted(stage_dir.iterdir()):
        m = FRAME_FILE.match(path.name)
        if m and m.group(1) == args.parent and m.group(3) == ext:
            candidates.append((int(m.group(2)), path))
    candidates.sort()
    duplicates = [i for i, _ in candidates if i in already]
    candidates = [(i, p) for i, p in candidates if i not in already]
    if duplicates:
        print(f"warning: frames already in the dataset are skipped: {duplicates}")
    if not candidates:
        fail("no frame left to commit")

    dest = FOLDERS[parent["availability"]]
    number = next_number(rows)
    note_tail = (f"; saved as JPEG (quality {quality})" if ext == "jpg" else "; saved as PNG")
    print(f"Committing {len(candidates)} frames: COEL-{number:0{ID_WIDTH}d} -> "
          f"COEL-{number + len(candidates) - 1:0{ID_WIDTH}d} into {dest.relative_to(ROOT)}/")
    if args.dry_run:
        print("Dry run: nothing written.")
        return

    dest.mkdir(parents=True, exist_ok=True)
    moved, new_rows = [], []
    try:
        for frame_index, path in candidates:
            image_id = f"COEL-{number:0{ID_WIDTH}d}"
            target = dest / f"{image_id}.{ext}"
            if target.exists():
                fail(f"{target} already exists")
            shutil.move(str(path), str(target))
            moved.append((target, path))
            row = {c: "" for c in COLUMNS}
            row.update({k: parent[k] for k in INHERITED})
            row.update({
                "image_id": image_id,
                "filename": target.name,
                "sha256": sha256_of(target),
                "parent_id": args.parent,
                "modifications": (f"Frame {frame_index} (t = {frame_index / fps:.3f} s) "
                                  f"taken from {parent['filename']}{note_tail}"),
            })
            new_rows.append(row)
            number += 1
        if is_placeholder(parent["sha256"]):
            parent["sha256"] = video_hash
            print(f"Recorded the sha256 of {parent['filename']} in images.csv")
        write_images(rows + new_rows)
    except BaseException:
        for target, original in moved:  # roll back: put the frames back in staging
            shutil.move(str(target), str(original))
        raise

    shutil.rmtree(stage_dir)
    print(f"Done: {new_rows[0]['image_id']} -> {new_rows[-1]['image_id']} added to images.csv.")
    print("Next: python scripts/check_repo.py, then commit.")


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