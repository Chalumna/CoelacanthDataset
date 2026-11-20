#!/usr/bin/env python3
"""Look for shortcuts a detector could learn instead of the coelacanth.

    python scripts/audit_images.py              # summary + staging/audit/audit.csv
    python scripts/audit_images.py --sheets     # also contact sheets of images with text or borders

For every image of the collection present on this machine (positives from
metadata/images.csv, negatives from metadata/negatives.csv, videos skipped), it
measures what a network can pick up without looking at the fish: dimensions
and aspect ratio, file format, burned-in text or logos, uniform borders,
brightness, colour. The summary compares positives and negatives per origin.
A feature that differs sharply between positives and negatives (all positives
are 16:9 PNG, all negatives 4:3 JPEG...) is a shortcut to neutralise before
training: augmentation, cropping, masking or more varied sources.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import image_checks as ic  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FOLDERS = {
    "public": ROOT / "data" / "unrestricted",
    "public_with_conditions": ROOT / "data" / "with-conditions",
    "local_only": ROOT / "data" / "local-only",
    "external": ROOT / "data" / "external",
}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
OUT_DIR = ROOT / "staging" / "audit"
FIELDS = ["image_id", "label", "origin", "source_group", "format", "width", "height", "aspect",
          "text", "borders", "brightness", "saturation", "hue"]


def rows_of(path: Path):
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def measure(path: Path):
    image = ic.load_bgr(path)
    if image is None:
        return None
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(cv2.resize(image, (256, max(1, round(256 * h / w)))), cv2.COLOR_BGR2HSV)
    borders = ic.find_borders(image)
    return {
        "width": w, "height": h, "aspect": round(w / h, 2),
        "text": ic.describe_text(ic.find_text(image)),
        "borders": ",".join(map(str, borders)) if any(borders) else "",
        "brightness": round(float(hsv[..., 2].mean()), 1),
        "saturation": round(float(hsv[..., 1].mean()), 1),
        "hue": round(float(np.median(hsv[..., 0])) * 2),  # degrees
    }


def summarize(records) -> None:
    groups = defaultdict(list)
    for r in records:
        groups[(r["label"], r["origin"])].append(r)
    print(f"{'label':9} {'origin':28} {'n':>5} {'groups':>6} {'formats':18} {'median size':>12} "
          f"{'aspects':16} {'text':>5} {'border':>6} {'bright':>6} {'sat':>5} {'hue':>4}")
    for (label, origin), rs in sorted(groups.items()):
        formats = defaultdict(int)
        aspects = defaultdict(int)
        for r in rs:
            formats[r["format"]] += 1
            aspects[r["aspect"]] += 1
        fmt = " ".join(f"{k}:{v}" for k, v in sorted(formats.items(), key=lambda kv: -kv[1])[:3])
        asp = " ".join(f"{k}" for k, _ in sorted(aspects.items(), key=lambda kv: -kv[1])[:3])
        size = f"{int(np.median([r['width'] for r in rs]))}x{int(np.median([r['height'] for r in rs]))}"
        pct = lambda key: f"{100 * sum(1 for r in rs if r[key]) / len(rs):.0f}%"  # noqa: E731
        print(f"{label:9} {origin[:28]:28} {len(rs):5d} {len({r['source_group'] for r in rs}):6d} {fmt[:18]:18} "
              f"{size:>12} {asp[:16]:16} {pct('text'):>5} {pct('borders'):>6} "
              f"{np.mean([r['brightness'] for r in rs]):6.0f} {np.mean([r['saturation'] for r in rs]):5.0f} "
              f"{np.median([r['hue'] for r in rs]):4.0f}")


def sheets(records) -> None:
    flagged = [r for r in records if r["text"] or r["borders"]]
    for old in OUT_DIR.glob("flagged-*.jpg"):
        old.unlink()
    tile_w, tile_h, cols, per = 320, 200, 5, 30
    for s, start in enumerate(range(0, len(flagged), per), 1):
        chunk = flagged[start:start + per]
        sheet = np.full((((len(chunk) + cols - 1) // cols) * (tile_h + 22), cols * tile_w, 3), 255, np.uint8)
        for i, r in enumerate(chunk):
            image = ic.load_bgr(r["_path"])
            sc = min(tile_w / image.shape[1], tile_h / image.shape[0])
            for item in r["text"].split("|") if r["text"] else []:
                x0, y0, x1, y1 = (int(v) for v in item.split("@")[1].split(","))
                cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 255), max(2, int(2 / sc)))
            small = cv2.resize(image, (int(image.shape[1] * sc), int(image.shape[0] * sc)))
            y, x = (i // cols) * (tile_h + 22), (i % cols) * tile_w
            sheet[y:y + small.shape[0], x:x + small.shape[1]] = small
            what = ("text " if r["text"] else "") + ("border" if r["borders"] else "")
            cv2.putText(sheet, f"{r['image_id']} {what}", (x + 3, y + tile_h + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 180), 1)
        cv2.imwrite(str(OUT_DIR / f"flagged-{s:03d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"{len(flagged)} flagged images on contact sheets in {OUT_DIR.relative_to(ROOT)}/")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sheets", action="store_true", help="contact sheets of images with text or borders")
    args = parser.parse_args()
    annex = {r["image_id"]: r["dataset"] for r in rows_of(ROOT / "metadata" / "negatives_external.csv")}
    records = []
    for label, csv_path in (("positive", ROOT / "metadata" / "images.csv"),
                            ("negative", ROOT / "metadata" / "negatives.csv")):
        for row in rows_of(csv_path):
            path = FOLDERS.get(row["availability"], ROOT / "missing") / row["filename"]
            if path.suffix.lower() not in IMAGE_EXT or not path.exists():
                continue
            m = measure(path)
            if m is None:
                continue
            origin = (f"external/{annex[row['image_id']]}" if row["image_id"] in annex
                      else row["availability"])
            records.append({"image_id": row["image_id"], "label": label, "origin": origin,
                            "source_group": row["source_group"], "format": path.suffix.lower().lstrip("."),
                            **m, "_path": path})
    if not records:
        sys.exit("no image found on this machine")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "audit.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    summarize(records)
    print(f"\nPer-image results: {(OUT_DIR / 'audit.csv').relative_to(ROOT)}")
    if args.sheets:
        sheets(records)


if __name__ == "__main__":
    main()
