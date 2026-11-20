#!/usr/bin/env python3
"""Image checks shared by the dataset scripts.

* find_borders()   uniform black or white bands (letterbox, technical strips);
* find_text()      burned-in text: timecodes, depth counters, logos, credits;
* dhash()          64-bit perceptual hash, to catch near-duplicates;
* hamming()        distance between two dhash values.

These are heuristics. find_text() was tuned on the frames of COEL-000004,
whose "Sakaue et al." credit it finds on every frame, and on the other images
of the collection (about 1 false alarm in 60 images, on very regular textures
such as comb-like corals). It flags an image for review; it does not prove
there is text.
"""
from __future__ import annotations

import cv2
import numpy as np

WORK_WIDTH = 960


def to_gray(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr.ndim == 2:
        return image_bgr
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


# ------------------------------------------------------------------ borders --
def find_borders(image_bgr: np.ndarray, *, max_std: float = 3.0, dark: float = 16.0,
                 bright: float = 240.0, min_frac: float = 0.01, jump: float = 10.0):
    """Thickness (top, bottom, left, right) of uniform dark or bright bands.

    A band counts only if it is nearly flat (std <= max_std, the level of
    compression noise), very dark or very bright, at least min_frac of the
    image, and ends with a sharp change. This keeps dark open water, which
    fades out gradually, from being taken for a letterbox.
    """
    gray = to_gray(image_bgr).astype(np.float32)
    h, w = gray.shape

    def band(lines) -> int:
        n, level = 0, None
        for line in lines:
            m, s = float(line.mean()), float(line.std())
            flat = s <= max_std and (m <= dark or m >= bright)
            if flat and (level is None or abs(m - level) <= 4):
                level = m if level is None else level
                n += 1
            else:
                if n and level is not None and abs(m - level) < jump and s < 3 * max_std + 4:
                    return 0  # no sharp edge: gradual dark water, not a border
                break
        return n

    rows = band(gray[i] for i in range(h)), band(gray[h - 1 - i] for i in range(h))
    cols = band(gray[:, j] for j in range(w)), band(gray[:, w - 1 - j] for j in range(w))
    out = []
    for value, size in ((rows[0], h), (rows[1], h), (cols[0], w), (cols[1], w)):
        out.append(value if max(2, min_frac * size) <= value < size // 3 else 0)
    return tuple(out)


# --------------------------------------------------------------------- text --
def _components(response: np.ndarray):
    otsu, _ = cv2.threshold(response, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    mask = (response >= max(otsu, 40)).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    comps = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if not (7 <= h <= 40 and 2 <= w <= 40):
            continue
        if not (0.15 <= area / float(w * h) <= 0.85):
            continue
        if not (0.12 <= w / h <= 2.0):
            continue
        comps.append((x, y, w, h))
    return comps


def _lines(comps, min_chars: int):
    comps = sorted(comps)
    parent = list(range(len(comps)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, (x1, y1, w1, h1) in enumerate(comps):
        for j in range(i + 1, len(comps)):
            x2, y2, w2, h2 = comps[j]
            gap = x2 - (x1 + w1)
            if gap > 1.2 * max(h1, h2):
                break
            if gap < -0.3 * w1 or max(h1, h2) / min(h1, h2) > 1.5:
                continue
            tol = 0.25 * max(h1, h2)
            if abs((y1 + h1) - (y2 + h2)) > tol and abs(y1 - y2) > tol:
                continue
            parent[root(i)] = root(j)
    groups = {}
    for i, c in enumerate(comps):
        groups.setdefault(root(i), []).append(c)
    lines = []
    for g in groups.values():
        if len(g) < min_chars:
            continue
        heights = np.array([c[3] for c in g], float)
        x0, y0 = min(c[0] for c in g), min(c[1] for c in g)
        x1, y1 = max(c[0] + c[2] for c in g), max(c[1] + c[3] for c in g)
        if x1 - x0 < 3 * np.median(heights) or heights.std() / heights.mean() > 0.25:
            continue
        lines.append((len(g), (x0, y0, x1, y1)))
    return lines


def find_text(image_bgr: np.ndarray, min_chars: int = 6):
    """Return a list of (n_chars, (x0, y0, x1, y1)) text-like lines, in pixels.

    Looks for rows of 6 or more glyph-sized blobs of similar height, aligned on
    a common baseline, brighter (white text) or darker (black text) than their
    surroundings.
    """
    h0, w0 = image_bgr.shape[:2]
    scale = WORK_WIDTH / w0 if w0 > WORK_WIDTH else 1.0
    img = image_bgr if scale == 1.0 else cv2.resize(
        image_bgr, (round(w0 * scale), round(h0 * scale)), interpolation=cv2.INTER_AREA)
    gray = to_gray(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    found = []
    for op in (cv2.MORPH_TOPHAT, cv2.MORPH_BLACKHAT):
        for n, box in _lines(_components(cv2.morphologyEx(gray, op, kernel)), min_chars):
            found.append((n, tuple(round(v / scale) for v in box)))
    # merge the bright and dark detections of the same outlined text
    merged = []
    for n, box in sorted(found, key=lambda f: -f[0]):
        if not any(_overlap(box, other) > 0.5 for _, other in merged):
            merged.append((n, box))
    return merged


def _overlap(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])) or 1
    return inter / smaller


def describe_text(found) -> str:
    """'text@x0,y0,x1,y1' items joined by '|' (empty string if nothing)."""
    return "|".join("text@" + ",".join(str(v) for v in box) for _, box in found)


# -------------------------------------------------------------------- dhash --
def dhash(image_bgr: np.ndarray) -> str:
    """64-bit difference hash, as 16 hex characters."""
    small = cv2.resize(to_gray(image_bgr), (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


class HashIndex:
    """Near-duplicate lookup over many dhash values (vectorised with numpy)."""

    def __init__(self) -> None:
        self._values = np.zeros(0, dtype=np.uint64)
        self._labels: list[str] = []
        self._pending: list[int] = []

    def add(self, value: str, label: str) -> None:
        self._pending.append(int(value, 16))
        self._labels.append(label)

    def _flush(self) -> None:
        if self._pending:
            self._values = np.concatenate([self._values, np.array(self._pending, dtype=np.uint64)])
            self._pending = []

    def nearest(self, value: str, max_distance: int):
        """Return (label, distance) of the closest hash within max_distance, else None."""
        self._flush()
        if not len(self._values):
            return None
        x = np.bitwise_xor(self._values, np.uint64(int(value, 16)))
        # popcount on uint64, byte by byte
        counts = np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)
        i = int(np.argmin(counts))
        d = int(counts[i])
        return (self._labels[i], d) if d <= max_distance else None

    def __len__(self) -> int:
        return len(self._labels)


def load_bgr(path) -> np.ndarray | None:
    """Read an image file (jpg, png, webp, tif) as BGR, or None."""
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return image
