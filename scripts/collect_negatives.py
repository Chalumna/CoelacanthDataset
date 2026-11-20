#!/usr/bin/env python3
"""Collect negative examples (no coelacanth) from open underwater datasets.

Sources
-------
* fathomnet    FathomNet open data on AWS (s3://fathomnet-pds): deep-sea and
               ROV imagery, licence chosen by each contributor, per image.
* deepfish     DeepFish (Saleh et al. 2020), James Cook University, CC BY 4.0:
               fixed cameras over 20 coastal habitats of tropical Australia,
               frames with and without fish.
* reefsupport  Reef Support Marine Data (Hugging Face): reef and underwater
               imagery aggregated from several open sources, licence per row.

Where the images go
-------------------
Collected images get NEG-xxxxxx identifiers and `availability` = `external`:

* metadata/negatives.csv            one row per image, same schema as the other
                                    negatives (licence, creator, source_group...);
* metadata/negatives_external.csv   source-specific details: original record,
                                    dimensions, processing, category, content,
                                    perceptual hash...;
* data/external/                    the files. NEVER pushed to Git: anyone can
                                    rebuild them with the `restore` command.

Commands
--------
    python scripts/collect_negatives.py plan                     # what would be collected (metadata only)
    python scripts/collect_negatives.py collect --target fathomnet=100 --target deepfish=100 --target reefsupport=100
    python scripts/collect_negatives.py collect                  # default targets: 4000 / 4000 / 2000
    python scripts/collect_negatives.py restore                  # rebuild data/external/ from the CSV files
    python scripts/collect_negatives.py review --sample 200      # contact sheets in staging/review/
    python scripts/collect_negatives.py reject NEG-000123 --reason "watermark"

`collect` can be interrupted (Ctrl-C) and run again: images already in the CSV
files are kept, the next run picks up where the last one stopped. Selection is
deterministic for a given --seed.

What every image goes through
-----------------------------
1. licence check: the licence must be known and open (CC0, CC BY, CC BY-SA,
   CC BY-NC..., Apache-2.0, MIT, public domain). NC is kept unless
   --exclude-nc. ND images are kept only byte-for-byte unmodified (or skipped
   with --exclude-nd);
2. integrity: SHA-256 of the downloaded bytes compared with the source's;
3. size: shorter side >= --min-side;
4. borders: uniform black/white bands are cropped (ND images with borders are
   skipped);
5. burned-in text (timecodes, depth counters, credits, logos): skipped, unless
   --keep-text, in which case the location is recorded;
6. near-duplicates: perceptual hash compared with every image of the
   collection, positives included;
7. resize to --max-side, JPEG re-encoding, metadata stripped;
8. diversity: source groups (dive, habitat, survey site) are taken in turn,
   round-robin; DeepFish: at most --deepfish-per-clip frames per video clip.

The licence checks are automatic and only as good as the sources' metadata:
review a sample (`review` command) before relying on them.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import fnmatch
import hashlib
import io
import json
import os
import random
import re
import shutil
import sys
import tarfile
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
import image_checks as ic  # noqa: E402

COLLECTOR_VERSION = "1.0"
ROOT = Path(__file__).resolve().parent.parent
IMAGES_CSV = ROOT / "metadata" / "images.csv"
NEGATIVES_CSV = ROOT / "metadata" / "negatives.csv"
ANNEX_CSV = ROOT / "metadata" / "negatives_external.csv"
REJECTED_CSV = ROOT / "metadata" / "negatives_rejected.csv"
EXTERNAL_DIR = ROOT / "data" / "external"
PARTIAL_DIR = EXTERNAL_DIR / ".partial"
STAGING_DIR = ROOT / "staging"
DEFAULT_CACHE = ROOT / ".cache" / "negatives"
FOLDERS = {
    "public": ROOT / "data" / "unrestricted",
    "public_with_conditions": ROOT / "data" / "with-conditions",
    "local_only": ROOT / "data" / "local-only",
    "external": EXTERNAL_DIR,
}

NEG_COLUMNS = [
    "image_id", "filename", "source_url", "creator", "license", "license_url",
    "rights_verified", "sha256", "source_group", "parent_id", "verified_on",
    "modifications", "availability", "description",
]
IMAGES_COLUMNS = [
    "image_id", "filename", "species", "live_status", "source_url", "creator",
    "license", "license_url", "rights_verified", "sha256", "source_group",
    "parent_id", "verified_on", "modifications", "availability",
]
ANNEX_COLUMNS = [
    "image_id", "dataset", "source_record_id", "source_path", "source_sha256",
    "original_width", "original_height", "width", "height", "processing",
    "category", "content", "site", "depth_m", "imaging_type", "attribution",
    "dhash", "text_overlay", "collected_on", "collector_version",
]
REJECTED_COLUMNS = ["dataset", "source_record_id", "image_id", "reason", "rejected_on"]

DATASETS = ["fathomnet", "deepfish", "reefsupport"]
DATASET_LABELS = {
    "fathomnet": "FathomNet",
    "deepfish": "DeepFish",
    "reefsupport": "Reef Support Marine Data",
}
DEFAULT_TARGETS = {"fathomnet": 4000, "deepfish": 4000, "reefsupport": 2000}
CATEGORY_LABELS = {
    "empty_scene": "underwater scene without animals",
    "marine_life": "underwater scene with other marine life",
    "reef_benthic": "reef or seafloor scene (corals, benthic organisms)",
    "unspecified": "underwater scene",
}
COELACANTH = re.compile(r"latimeria|coelacanth|actinistia", re.IGNORECASE)
TODAY = dt.date.today().isoformat()


def fail(message: str):
    sys.exit(f"error: {message}")


def log(message: str = "") -> None:
    print(message, flush=True)


# =================================================================== licences
def normalize_license(raw: str | None, url: str | None = None):
    """Map a source licence string to (label, url, flags) or None if unusable.

    label follows the convention of the CSV files ("CC BY 4.0", "CC0 1.0"...),
    flags is a set among {"NC", "ND", "SA"}.
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.lower() in ("nan", "none", "unknown", "all rights reserved"):
        return None
    key = re.sub(r"[\s_]+", "-", s.upper())
    if key.startswith("CC0") or key.startswith("CC-ZERO"):
        return "CC0 1.0", "https://creativecommons.org/publicdomain/zero/1.0/", set()
    m = re.fullmatch(r"CC-(BY(?:-(?:NC|SA|ND))*)-(\d\.\d)(?:-[A-Z]{2,})?", key)
    if m:
        parts, version = m.group(1), m.group(2)
        flags = set(parts.split("-")[1:])
        return (f"CC {parts} {version}",
                f"https://creativecommons.org/licenses/{parts.lower()}/{version}/", flags)
    if key in ("APACHE-2.0", "APACHE-2"):
        return "Apache-2.0", "https://www.apache.org/licenses/LICENSE-2.0", set()
    if key == "MIT":
        return "MIT", "https://opensource.org/license/mit", set()
    if key in ("US-GOV-PD", "PDM", "PDM-1.0", "PUBLIC-DOMAIN", "PUBLICDOMAIN"):
        return ("Public domain (US Government work)" if key == "US-GOV-PD" else "Public domain",
                url or "https://creativecommons.org/publicdomain/mark/1.0/", set())
    return None


# ==================================================================== records
@dataclass
class Record:
    dataset: str
    record_id: str                 # stable identifier in the source
    source_path: str               # where the bytes are (S3 key, tar member, parquet shard)
    source_url: str
    license_raw: str
    creator: str
    attribution: str
    source_group: str
    category: str = "unspecified"
    content: str = ""
    site: str = ""
    depth_m: str = ""
    imaging_type: str = ""
    source_sha256: str = ""        # expected SHA-256 of the downloaded bytes, if known
    license_url_raw: str = ""
    sub_group: str = ""            # finer unit for diversity (video clip, dive)
    rank: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def key(self):
        return (self.dataset, self.record_id)


def det_rank(seed: int, *parts) -> float:
    """Deterministic pseudo-random number in [0, 1) for a record."""
    digest = hashlib.sha256(":".join([str(seed), *map(str, parts)]).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2 ** 64


def interleave(records, cap: int, existing: Counter):
    """Round-robin over source groups (best ranked first), at most `cap` per group."""
    by_group = defaultdict(list)
    for r in sorted(records, key=lambda r: r.rank):
        by_group[r.source_group].append(r)
    queues = sorted(by_group.values(), key=lambda q: q[0].rank)
    taken = Counter(existing)
    out, i = [], 0
    while queues:
        alive = []
        for q in queues:
            if i < len(q) and taken[q[i].source_group] < cap:
                out.append(q[i])
                taken[q[i].source_group] += 1
                alive.append(q)
        queues = alive
        i += 1
        queues = [q for q in queues if i < len(q)]
    return out


# ================================================================ HTTP helpers
_session = None


def http():
    global _session
    if _session is None:
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        _session = requests.Session()
        retry = Retry(total=5, backoff_factor=1.0, status_forcelist=(429, 500, 502, 503, 504))
        _session.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=32))
        _session.headers["User-Agent"] = f"Chalumna-CoelacanthDataset-collector/{COLLECTOR_VERSION}"
    return _session


def http_get(url: str, timeout: float = 60) -> bytes:
    response = http().get(url, timeout=timeout)
    response.raise_for_status()
    return response.content


def download_file(url: str, dest: Path, expected_size: int | None = None) -> Path:
    if dest.exists() and (expected_size is None or dest.stat().st_size == expected_size):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with http().get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in response.iter_content(1 << 20):
                f.write(chunk)
    os.replace(tmp, dest)
    return dest


def parallel_fetch(records, fetch_one, workers: int, window: int = 64):
    """Yield (record, bytes or None, error or None) in order, `workers` downloads at a time."""
    records = iter(records)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = []
        try:
            for r in records:
                pending.append((r, pool.submit(fetch_one, r)))
                if len(pending) >= window:
                    r0, fut = pending.pop(0)
                    yield _result(r0, fut)
            for r0, fut in pending:
                yield _result(r0, fut)
            pending = []
        finally:
            for _, fut in pending:
                fut.cancel()


def _result(record, future):
    try:
        return record, future.result(), None
    except Exception as error:  # network or source error: reported, not fatal
        return record, None, f"{type(error).__name__}: {error}"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def slug(text: str, n: int = 40) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")[:n] or "unknown"


# ==================================================================== sources
class Source:
    name = ""
    keep_order = False  # True: the source orders candidates itself (do not interleave)

    def __init__(self, args) -> None:
        self.args = args
        self.cache = Path(args.cache_dir) / self.name

    def candidates(self, target: int) -> list[Record]:
        raise NotImplementedError

    def fetch(self, records):
        """Yield (record, bytes or None, error or None)."""
        raise NotImplementedError

    def records_from_annex(self, rows) -> list[Record]:
        """Rebuild the records needed by fetch() from annex rows (for restore)."""
        out = []
        for neg, annex in rows:
            out.append(Record(
                dataset=self.name, record_id=annex["source_record_id"],
                source_path=annex["source_path"], source_url=neg["source_url"],
                license_raw=neg["license"], creator=neg["creator"],
                attribution=annex["attribution"], source_group=neg["source_group"],
                source_sha256=annex["source_sha256"]))
        return out

    def describe(self) -> str:
        return ""


# ---------------------------------------------------------------- FathomNet --
class FathomNet(Source):
    """FathomNet open data on AWS: Parquet metadata, one licence per image."""

    name = "fathomnet"
    base = "https://fathomnet-pds.s3.us-west-2.amazonaws.com"
    citation = "Katija et al. (2022), Sci. Rep. 12:15914, doi:10.1038/s41598-022-19939-2"
    non_biological = {
        "rock", "rocks", "sand", "sediment", "mud", "trash", "debris", "marine snow",
        "bubble", "bubbles", "laser", "shadow", "equipment", "rope", "anchor",
        "physical object", "geological feature", "substrate", "boulder", "cobble",
        "gravel", "bacterial mat", "detritus", "unknown",
    }

    def release(self) -> str:
        if getattr(self, "_release", None):
            return self._release
        if self.args.fathomnet_release:
            self._release = self.args.fathomnet_release
            return self._release
        text = http_get(f"{self.base}/metadata/releases.csv").decode("utf-8")
        releases = [r["release"] for r in csv.DictReader(io.StringIO(text)) if r.get("release")]
        if not releases:
            fail("FathomNet: metadata/releases.csv lists no release")
        self._release = max(releases)
        return self._release

    def list_keys(self, prefix: str) -> list[tuple[str, int]]:
        keys, token = [], None
        ns = "{http://s3.amazonaws.com/doc/2006-03-01/}"
        while True:
            params = {"list-type": "2", "prefix": prefix}
            if token:
                params["continuation-token"] = token
            response = http().get(f"{self.base}/", params=params, timeout=60)
            response.raise_for_status()
            root = ET.fromstring(response.content)
            for item in root.iter(f"{ns}Contents"):
                keys.append((item.find(f"{ns}Key").text, int(item.find(f"{ns}Size").text)))
            truncated = root.find(f"{ns}IsTruncated")
            token_el = root.find(f"{ns}NextContinuationToken")
            if truncated is None or truncated.text != "true" or token_el is None:
                return keys
            token = token_el.text

    def table(self, name: str, columns, required=()):
        import pandas as pd
        release = self.release()
        prefix = f"metadata/{name}/release={release}/"
        parts = [(k, s) for k, s in self.list_keys(prefix) if k.endswith(".parquet")]
        if not parts:
            fail(f"FathomNet: no Parquet file under {prefix}")
        frames = []
        for key, size in parts:
            path = download_file(f"{self.base}/{key}", self.cache / key, size)
            available = set(_parquet_columns(path))
            missing = [c for c in required if c not in available]
            if missing:
                fail(f"FathomNet {key}: missing columns {missing}; the data dictionary changed, "
                     "update the FathomNet adapter")
            frames.append(pd.read_parquet(path, columns=[c for c in columns if c in available]))
        table = pd.concat(frames, ignore_index=True)
        for c in columns:
            if c not in table.columns:
                table[c] = None
        return table

    def candidates(self, target: int) -> list[Record]:
        a = self.args
        log(f"FathomNet: release {self.release()}, reading metadata...")
        images = self.table("images", [
            "image_uuid", "s3_key", "thumbnail_s3_key", "sha256", "media_type", "width",
            "height", "image_license", "imaging_type", "recorded_timestamp", "depth_meters"],
            required=("image_uuid", "s3_key", "width", "height", "image_license"))
        boxes = self.table("boxes", ["image_uuid", "concept", "alt_concept"],
                           required=("image_uuid", "concept"))
        links = self.table("image_collections", ["image_uuid", "collection_uuid"])
        colls = self.table("collections", [
            "collection_uuid", "dataset_name", "owner_institution_code", "rights_holder",
            "bibliographic_citation"])
        n0 = len(images)

        concepts = defaultdict(set)
        excluded = set()
        for uuid, concept, alt in boxes[["image_uuid", "concept", "alt_concept"]].itertuples(index=False):
            for c in (concept, alt):
                if isinstance(c, str) and c.strip():
                    concepts[uuid].add(c.strip())
                    if COELACANTH.search(c):
                        excluded.add(uuid)
        coll_of = dict(links.drop_duplicates("image_uuid")[["image_uuid", "collection_uuid"]]
                       .itertuples(index=False))
        coll_info = {r.collection_uuid: r for r in colls.itertuples(index=False)}

        stats = Counter()
        out = []
        for r in images.itertuples(index=False):
            lic = normalize_license(r.image_license)
            if lic is None:
                stats["unknown licence"] += 1
                continue
            if not licence_allowed(lic[2], a):
                stats[f"licence excluded ({lic[0]})"] += 1
                continue
            if r.image_uuid in excluded:
                stats["coelacanth annotation"] += 1
                continue
            if min(_int(r.width), _int(r.height)) < a.min_side:
                stats["too small"] += 1
                continue
            depth = _float(r.depth_meters)
            if a.fathomnet_min_depth is not None and (depth is None or depth < a.fathomnet_min_depth):
                stats["outside depth range"] += 1
                continue
            if a.fathomnet_max_depth is not None and (depth is None or depth > a.fathomnet_max_depth):
                stats["outside depth range"] += 1
                continue
            if "ND" in lic[2] and str(r.media_type) not in ("image/jpeg", "image/png"):
                stats["ND licence, format cannot stay unmodified"] += 1
                continue
            coll = coll_of.get(r.image_uuid, "")
            info = coll_info.get(coll)
            holder = _first(getattr(info, "rights_holder", None), getattr(info, "owner_institution_code", None),
                            "FathomNet contributor")
            biblio = _first(getattr(info, "bibliographic_citation", None), "")
            day = str(r.recorded_timestamp)[:10].replace("-", "") if _str(r.recorded_timestamp) else ""
            group = f"fathomnet-{(coll or 'nocollection')[:8]}" + (f"-{day}" if day else "")
            names = sorted(concepts.get(r.image_uuid, ()))
            bio = [c for c in names if c.lower() not in self.non_biological]
            key = r.thumbnail_s3_key if a.fathomnet_thumbnails else r.s3_key
            if not _str(key):
                stats["no object key"] += 1
                continue
            out.append(Record(
                dataset=self.name, record_id=r.image_uuid, source_path=key,
                source_url=f"{self.base}/{key}", license_raw=r.image_license,
                creator=holder,
                attribution="; ".join(p for p in (holder, biblio, f"via FathomNet ({self.citation})") if p),
                source_group=group, sub_group=group,
                category="marine_life" if bio else ("empty_scene" if names else "unspecified"),
                content="; ".join(names)[:300],
                site=_first(getattr(info, "dataset_name", None), ""),
                depth_m="" if depth is None else f"{depth:.0f}",
                imaging_type=_str(r.imaging_type),
                source_sha256="" if a.fathomnet_thumbnails else _str(r.sha256).lower(),
                rank=det_rank(a.seed, self.name, r.image_uuid)))
        self._stats = (n0, stats)
        return out

    def describe(self) -> str:
        n0, stats = getattr(self, "_stats", (0, Counter()))
        return f"{n0} images in the release; filtered out: " + (
            ", ".join(f"{k}: {v}" for k, v in stats.most_common()) or "none")

    def fetch(self, records):
        return parallel_fetch(records, lambda r: http_get(f"{self.base}/{r.source_path}"),
                              self.args.workers)


# ----------------------------------------------------------------- DeepFish --
class DeepFish(Source):
    """DeepFish classification frames, read from the official tar archive."""

    name = "deepfish"
    keep_order = True
    repo = "Alzayats/DeepFish"
    filename = "DeepFish.tar"
    sha256 = "8acabb8a314fa8eb45fa9cc6829c84bdc127b1f8f605f2e152304e98b6dbb46c"
    url = "https://huggingface.co/datasets/Alzayats/DeepFish"
    creator = "Saleh et al. (2020), James Cook University"
    attribution = ("DeepFish, A. Saleh, M. Bradley, M. Sheaves, James Cook University, "
                   "doi:10.25903/5f617fb6d6e0e; Saleh et al. (2020), Sci. Rep., "
                   "doi:10.1038/s41598-020-71639-x; CC BY 4.0")
    frame_re = re.compile(r"^(?P<clip>.+?)_f(?P<frame>\d+)$")

    def tar_path(self) -> Path:
        if getattr(self, "_tar", None):
            return self._tar
        if self.args.deepfish_tar:
            path = Path(self.args.deepfish_tar).expanduser()
            if not path.exists():
                fail(f"--deepfish-tar: {path} not found")
        else:
            from huggingface_hub import hf_hub_download
            log(f"DeepFish: downloading {self.filename} (7.1 GB, resumable) into {self.cache}...")
            path = Path(hf_hub_download(self.repo, self.filename, repo_type="dataset",
                                        local_dir=str(self.cache)))
        marker = path.with_name(path.name + ".verified")
        stamp = f"{path.stat().st_size}:{int(path.stat().st_mtime)}"
        if not (marker.exists() and marker.read_text() == stamp):
            log("DeepFish: checking the SHA-256 of the archive (once)...")
            if sha256_file(path) != self.sha256:
                fail(f"{path}: SHA-256 differs from the published one ({self.sha256}); "
                     "delete it and download it again")
            marker.write_text(stamp)
        self._tar = path
        return path

    def index(self) -> list[dict]:
        if getattr(self, "_index", None) is not None:
            return self._index
        tar = self.tar_path()
        cached = self.cache / "classification_index.json"
        stamp = f"{tar.stat().st_size}:{int(tar.stat().st_mtime)}"
        if cached.exists():
            data = json.loads(cached.read_text())
            if data.get("stamp") == stamp:
                self._index = data["members"]
                return self._index
        log("DeepFish: indexing the archive (once)...")
        members = []
        with tarfile.open(tar, "r:") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                parts = m.name.split("/")
                if "Classification" not in parts:
                    continue
                i = parts.index("Classification")
                if len(parts) != i + 4:
                    continue
                stem, ext = os.path.splitext(parts[i + 3])
                if ext.lower() not in (".jpg", ".jpeg", ".png"):
                    continue
                members.append({"name": m.name, "offset": m.offset_data, "size": m.size,
                                "habitat": parts[i + 1], "label": parts[i + 2], "stem": stem})
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps({"stamp": stamp, "members": members}))
        self._index = members
        return members

    def candidates(self, target: int) -> list[Record]:
        a = self.args
        clips = defaultdict(list)
        for m in self.index():
            match = self.frame_re.match(m["stem"])
            clip, frame = (match.group("clip"), int(match.group("frame"))) if match else (m["stem"], 0)
            clips[(m["habitat"], m["label"], clip)].append((frame, m))
        pools = {"empty": [], "valid": []}
        for (habitat, label, clip), frames in clips.items():
            frames.sort(key=lambda f: f[0])
            k = min(a.deepfish_per_clip, len(frames))
            picks = [frames[round(i * (len(frames) - 1) / (k - 1))] if k > 1 else frames[len(frames) // 2]
                     for i in range(k)]
            seen = set()
            for frame, m in picks:
                if m["name"] in seen:
                    continue
                seen.add(m["name"])
                if label == "valid":
                    species = clip.split("_", 1)[1] if "_" in clip else clip
                    content = species.replace("_", " ")
                    category = "marine_life"
                else:
                    content, category = "no fish", "empty_scene"
                record_id = m["name"].split("DeepFish/", 1)[-1]
                pools["valid" if label == "valid" else "empty"].append(Record(
                    dataset=self.name, record_id=record_id, source_path=m["name"],
                    source_url=self.url, license_raw="CC-BY-4.0", creator=self.creator,
                    attribution=self.attribution, source_group=f"deepfish-{habitat}",
                    sub_group=f"{habitat}/{clip}", category=category, content=content,
                    site=f"habitat {habitat}", imaging_type="fixed camera frame (video)",
                    rank=det_rank(a.seed, self.name, record_id),
                    extra={"offset": m["offset"], "size": m["size"]}))
        self._stats = {k: len(v) for k, v in pools.items()}
        # habitats interleaved within each pool, then pools merged at the requested share
        empty = interleave(pools["empty"], cap_of(a), Counter())
        valid = interleave(pools["valid"], cap_of(a), Counter())
        share, out, ie, iv = a.deepfish_empty_share, [], 0, 0
        while ie < len(empty) or iv < len(valid):
            if iv >= len(valid) or (ie < len(empty) and ie <= share * (ie + iv)):
                out.append(empty[ie])
                ie += 1
            else:
                out.append(valid[iv])
                iv += 1
        return out

    def describe(self) -> str:
        s = getattr(self, "_stats", {})
        return (f"{s.get('empty', 0)} empty-scene and {s.get('valid', 0)} fish candidate frames "
                f"(at most {self.args.deepfish_per_clip} per clip)")

    def fetch(self, records):
        path = self.tar_path()
        offsets = None
        with open(path, "rb") as f:
            for r in records:
                try:
                    if "offset" not in r.extra:  # restore: look the member up
                        if offsets is None:
                            offsets = {m["name"]: m for m in self.index()}
                        m = offsets[r.source_path]
                        r.extra.update(offset=m["offset"], size=m["size"])
                    f.seek(r.extra["offset"])
                    yield r, f.read(r.extra["size"]), None
                except Exception as error:
                    yield r, None, f"{type(error).__name__}: {error}"


# -------------------------------------------------------------- Reef Support --
class ReefSupport(Source):
    """Reef Support Marine Data on Hugging Face (Parquet, licence per row)."""

    name = "reefsupport"
    repo = "reefsupport/marine-data"
    keep_order = True
    meta_columns = [
        "image_sha256", "source_id", "license", "license_url", "attribution", "rights_holder",
        "upstream_url", "upstream_id", "platform", "camera", "habitat", "depth_m",
        "privacy_flag", "face_score", "bytes_config", "bytes_sha256", "site_id", "survey_id",
        "region",
    ]

    def revision(self) -> str:
        if getattr(self, "_rev", None):
            return self._rev
        from huggingface_hub import HfApi
        self._rev = self.args.reefsupport_revision or HfApi().dataset_info(self.repo).sha
        return self._rev

    def files(self, kind: str) -> list[str]:
        from huggingface_hub import HfApi
        pattern = getattr(self.args, f"reefsupport_{kind}_glob")
        files = [f for f in HfApi().list_repo_files(self.repo, repo_type="dataset", revision=self.revision())
                 if fnmatch.fnmatch(f, pattern)]
        if not files:
            fail(f"Reef Support: no file matches {pattern!r} at revision {self.revision()}; "
                 f"check the repository layout and pass --reefsupport-{kind}-glob")
        return sorted(files)

    def open(self, path: str):
        from huggingface_hub import HfFileSystem
        if getattr(self, "_fs", None) is None:
            self._fs = HfFileSystem()
        return self._fs.open(f"datasets/{self.repo}@{self.revision()}/{path}", "rb")

    def metadata(self):
        import pandas as pd
        import pyarrow.parquet as pq
        cached = self.cache / f"metadata-{self.revision()[:12]}.parquet"
        if cached.exists():
            return pd.read_parquet(cached)
        frames = []
        for path in self.files("metadata"):
            with self.open(path) as f:
                pf = pq.ParquetFile(f)
                cols = [c for c in self.meta_columns if c in pf.schema_arrow.names]
                frames.append(pf.read(columns=cols).to_pandas())
        meta = pd.concat(frames, ignore_index=True).drop_duplicates("image_sha256")
        cached.parent.mkdir(parents=True, exist_ok=True)
        meta.to_parquet(cached)
        return meta

    def shard_index(self):
        """image_sha256 -> (shard, row group, split_group, row group bytes), from the images config."""
        import pyarrow.parquet as pq
        cached = self.cache / f"images-index-{self.revision()[:12]}.json"
        if cached.exists():
            return json.loads(cached.read_text())
        log("Reef Support: indexing the image shards (column 'image_sha256' only)...")
        index = {}
        for path in self.files("images"):
            with self.open(path) as f:
                pf = pq.ParquetFile(f)
                names = pf.schema_arrow.names
                cols = [c for c in ("image_sha256", "split_group", "source_id") if c in names]
                img_col = names.index("image") if "image" in names else None
                for g in range(pf.num_row_groups):
                    rg = pf.metadata.row_group(g)
                    size = sum(rg.column(j).total_compressed_size for j in range(rg.num_columns)
                               if img_col is None or rg.column(j).path_in_schema.startswith("image."))
                    t = pf.read_row_group(g, columns=cols).to_pydict()
                    for i, sha in enumerate(t["image_sha256"]):
                        index[sha] = [path, g, (t.get("split_group") or [""] * len(t["image_sha256"]))[i],
                                      size]
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps(index))
        return index

    def candidates(self, target: int) -> list[Record]:
        a = self.args
        log(f"Reef Support: revision {self.revision()[:12]}, reading metadata...")
        meta = self.metadata()
        index = self.shard_index()
        include = [p for p in a.reefsupport_include_sources.split(",") if p]
        exclude = [p for p in a.reefsupport_exclude_sources.split(",") if p]
        stats = Counter()
        eligible = []
        for r in meta.to_dict("records"):
            sha = r.get("image_sha256")
            src = _str(r.get("source_id"))
            if include and not any(fnmatch.fnmatch(src, p) for p in include):
                stats[f"source not included ({src})"] += 1
                continue
            if any(fnmatch.fnmatch(src, p) for p in exclude):
                stats[f"source excluded ({src})"] += 1
                continue
            if "bytes_config" in r and _str(r.get("bytes_config")) not in ("", "images"):
                stats["pixels not in the images config"] += 1
                continue
            if sha not in index:
                stats["not in the image shards"] += 1
                continue
            lic = normalize_license(r.get("license"), _str(r.get("license_url")))
            if lic is None:
                stats["unknown licence"] += 1
                continue
            if not licence_allowed(lic[2], a):
                stats[f"licence excluded ({lic[0]})"] += 1
                continue
            if _truthy(r.get("privacy_flag")) or (_float(r.get("face_score")) or 0) >= 0.5:
                stats["privacy flag (people)"] += 1
                continue
            shard, rg, split_group, _ = index[sha]
            site = _first(_str(r.get("site_id")), _str(r.get("survey_id")), split_group, "")
            group = f"reefsupport-{slug(src, 30)}-{slug(site, 30)}" if site else f"reefsupport-{slug(src, 30)}"
            eligible.append(Record(
                dataset=self.name, record_id=sha,
                source_path=f"{shard}#{sha}",
                source_url=_first(_str(r.get("upstream_url")), f"https://huggingface.co/datasets/{self.repo}"),
                license_raw=_str(r.get("license")), license_url_raw=_str(r.get("license_url")),
                creator=_first(_str(r.get("rights_holder")), _str(r.get("attribution")), src),
                attribution=_first(_str(r.get("attribution")), src) + f"; via Reef Support Marine Data "
                            f"(huggingface.co/datasets/{self.repo}, revision {self.revision()[:12]})",
                source_group=group, sub_group=group,
                category="marine_life" if src.startswith(("uiis", "usis", "suim")) else "reef_benthic",
                content=src, site=_str(r.get("habitat")) or _str(r.get("region")),
                depth_m=_str(r.get("depth_m")), imaging_type=_first(_str(r.get("platform")), _str(r.get("camera")), ""),
                source_sha256=_str(r.get("bytes_sha256")).lower(),
                rank=det_rank(a.seed, self.name, sha),
                extra={"shard": shard, "row_group": rg, "alt_sha256": sha}))
        # Pixels are stored by row groups of many images: reading one image means
        # downloading its whole row group. Pick a set of row groups that covers the
        # sources evenly and holds enough eligible images, then read them in order.
        by_rg = defaultdict(list)
        for r in eligible:
            by_rg[(r.extra["shard"], r.extra["row_group"])].append(r)
        rg_sources = defaultdict(lambda: defaultdict(list))
        for key, rs in by_rg.items():
            for src in {r.content for r in rs}:
                rg_sources[src][key] = rs
        wanted = int(target * a.reefsupport_overshoot)
        chosen, total = [], 0
        queues = {src: sorted(rgs, key=lambda k: det_rank(a.seed, "rg", *k)) for src, rgs in rg_sources.items()}
        while total < wanted and any(queues.values()):
            for src in sorted(queues):
                while queues[src] and queues[src][0] in chosen:
                    queues[src].pop(0)
                if queues[src]:
                    key = queues[src].pop(0)
                    chosen.append(key)
                    total += min(len(by_rg[key]), a.reefsupport_per_rowgroup)
                    if total >= wanted:
                        break
        sizes = {tuple(v[:2]): v[3] for v in index.values()}
        self._stats = (len(meta), stats, len(eligible), len(chosen), sum(sizes.get(k, 0) for k in chosen))
        out = []
        for key in chosen:
            out.extend(sorted(by_rg[key], key=lambda r: r.rank)[:a.reefsupport_per_rowgroup])
        return out

    def describe(self) -> str:
        n, stats, eligible, n_rg, size = getattr(self, "_stats", (0, Counter(), 0, 0, 0))
        filtered = ", ".join(f"{k}: {v}" for k, v in stats.most_common()) or "none"
        return (f"{n} rows in metadata, {eligible} eligible; filtered out: {filtered}. "
                f"{n_rg} row groups selected, about {size / 1e9:.1f} GB of image data to download")

    def records_from_annex(self, rows) -> list[Record]:
        out = super().records_from_annex(rows)
        index = None
        for r in out:
            shard, _, sha = r.source_path.partition("#")
            if index is None:
                index = self.shard_index()
            entry = index.get(sha)
            r.extra = {"shard": shard, "row_group": entry[1] if entry else None}
        return out

    def fetch(self, records):
        import pyarrow.parquet as pq
        groups = defaultdict(list)
        order = []
        for r in records:
            key = (r.extra.get("shard"), r.extra.get("row_group"))
            if key not in groups:
                order.append(key)
            groups[key].append(r)
        for shard, rg in order:
            rs = groups[(shard, rg)]
            if rg is None:
                for r in rs:
                    yield r, None, "image not found in the shard index"
                continue
            try:
                with self.open(shard) as f:
                    t = pq.ParquetFile(f).read_row_group(rg, columns=["image_sha256", "image"]).to_pydict()
                found = {}
                for sha, img in zip(t["image_sha256"], t["image"]):
                    found[sha] = img.get("bytes") if isinstance(img, dict) else img
            except Exception as error:
                for r in rs:
                    yield r, None, f"{type(error).__name__}: {error}"
                continue
            for r in rs:
                data = found.get(r.record_id)
                yield (r, data, None) if data else (r, None, "image bytes missing in the row group")


SOURCES = {"fathomnet": FathomNet, "deepfish": DeepFish, "reefsupport": ReefSupport}


# ================================================================== helpers --
def _str(v) -> str:
    if v is None:
        return ""
    try:
        if v != v:  # NaN
            return ""
    except Exception:
        pass
    return str(v).strip()


def _first(*values) -> str:
    for v in values:
        s = _str(v)
        if s:
            return s
    return ""


def _int(v) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _float(v):
    try:
        f = float(v)
        return None if f != f else f
    except (TypeError, ValueError):
        return None


def _truthy(v) -> bool:
    return _str(v).lower() not in ("", "0", "false", "none", "no", "ok", "nan")


def _parquet_columns(path: Path):
    import pyarrow.parquet as pq
    return pq.ParquetFile(path).schema_arrow.names


def cap_of(args) -> int:
    return args.max_per_group if getattr(args, "max_per_group", 0) else 10 ** 9


def licence_allowed(flags: set, args) -> bool:
    if "NC" in flags and args.exclude_nc:
        return False
    if "ND" in flags and args.exclude_nd:
        return False
    return True


# =============================================================== processing --
@dataclass
class Processed:
    data: bytes
    ext: str
    width: int
    height: int
    original_width: int
    original_height: int
    dhash: str
    text_overlay: str
    modifications: str
    processing: str


class Skip(Exception):
    pass


def decode(data: bytes):
    """Bytes -> (RGB numpy array, PIL format name). Handles EXIF rotation, 16-bit, alpha, CMYK."""
    try:
        im = Image.open(io.BytesIO(data))
        im.seek(0)
        fmt = im.format or ""
        im = ImageOps.exif_transpose(im)
    except Exception as error:
        raise Skip(f"cannot decode image ({type(error).__name__})")
    if im.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        arr = np.asarray(im, dtype=np.float64)
        lo, hi = np.percentile(arr, (0.5, 99.5))
        arr = np.clip((arr - lo) / max(hi - lo, 1e-9) * 255, 0, 255).astype(np.uint8)
        im = Image.fromarray(arr).convert("RGB")
    elif im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
        background = Image.new("RGB", im.size, (0, 0, 0))
        background.paste(im.convert("RGBA"), mask=im.convert("RGBA").split()[-1])
        im = background
    else:
        im = im.convert("RGB")
    return np.asarray(im), fmt


def process(data: bytes, record: Record, args, nd: bool, forced: dict | None = None) -> Processed:
    """Check and normalise one image. Raises Skip(reason) to reject it.

    forced (restore): {'crop': (l, t, r, b) or None, 'max_side': int, 'jpeg_quality': int,
    'mode': 'reencode' | 'original'}; detection steps are then skipped.
    """
    if record.source_sha256 and not forced:
        digest = sha256_bytes(data)
        if digest != record.source_sha256 and digest != record.extra.get("alt_sha256"):
            raise Skip("SHA-256 differs from the source's")
    rgb, fmt = decode(data)
    oh, ow = rgb.shape[:2]
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    if forced:
        mode = forced["mode"]
        crop = forced.get("crop")
        max_side, quality = forced["max_side"], forced["jpeg_quality"]
        text = ""
    else:
        if min(ow, oh) < args.min_side:
            raise Skip(f"too small ({ow}x{oh})")
        mode = "original" if nd else "reencode"
        top, bottom, left, right = ic.find_borders(bgr)
        crop = None
        if top or bottom or left or right:
            if nd:
                raise Skip("borders, but the ND licence forbids cropping")
            crop = (left, top, ow - right, oh - bottom)
        max_side, quality = args.max_side, args.jpeg_quality
    if crop:
        l, t, r, b = crop
        bgr = bgr[t:b, l:r]
    h, w = bgr.shape[:2]
    if not forced:
        if min(w, h) < args.min_side:
            raise Skip(f"too small after border crop ({w}x{h})")
        found = ic.find_text(bgr)
        if found and not args.keep_text:
            raise Skip("burned-in text or logo")
        text = ic.describe_text(found)

    if mode == "original":
        ext = {"JPEG": ".jpg", "PNG": ".png"}.get(fmt)
        if ext is None:
            raise Skip(f"ND licence: {fmt or 'this'} format cannot be kept unmodified")
        return Processed(data, ext, ow, oh, ow, oh, ic.dhash(bgr), text, "",
                         "mode=original")

    mods = []
    if crop:
        l, t, r, b = crop
        mods.append(f"uniform borders cropped (kept x {l}-{r}, y {t}-{b})")
    if max(w, h) > max_side:
        s = max_side / max(w, h)
        nw, nh = max(1, round(w * s)), max(1, round(h * s))
        bgr = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        mods.append(f"resized from {w}x{h} to {nw}x{nh}")
        w, h = nw, nh
    out = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(bgr[:, :, ::-1])).save(
        out, "JPEG", quality=quality, optimize=True, progressive=False, subsampling=0)
    mods.append(f"saved as JPEG (quality {quality}), metadata removed")
    processing = (f"mode=reencode;crop={','.join(map(str, crop)) if crop else ''};"
                  f"max_side={max_side};jpeg_quality={quality}")
    return Processed(out.getvalue(), ".jpg", w, h, ow, oh, ic.dhash(bgr), text,
                     _sentence("; ".join(mods)), processing)


def _sentence(text: str) -> str:
    return text[:1].upper() + text[1:]


def parse_processing(text: str) -> dict:
    fields = dict(p.split("=", 1) for p in text.split(";") if "=" in p)
    crop = tuple(int(v) for v in fields.get("crop", "").split(",")) if fields.get("crop") else None
    return {"mode": fields.get("mode", "reencode"), "crop": crop,
            "max_side": int(fields.get("max_side", 0) or 10 ** 9),
            "jpeg_quality": int(fields.get("jpeg_quality", 92) or 92)}


# ==================================================================== store --
def read_csv(path: Path, columns, required: bool = False):
    if not path.exists():
        if required:
            fail(f"{path.relative_to(ROOT)} not found")
        return []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header != columns:
            fail(f"{path.relative_to(ROOT)}: header differs from the expected schema")
        rows = []
        for n, row in enumerate(reader, start=2):
            if not any(c.strip() for c in row):
                continue
            if len(row) != len(columns):
                fail(f"{path.relative_to(ROOT)}, line {n}: {len(row)} fields instead of {len(columns)}")
            rows.append(dict(zip(columns, row)))
    return rows


def write_csvs(tables) -> None:
    """Write several CSV files: temp files first, then replace all. Keeps .bak copies."""
    temps = []
    try:
        for path, columns, rows in tables:
            path.parent.mkdir(parents=True, exist_ok=True)
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


class Store:
    """The negatives CSV files and data/external/, with batched, atomic commits."""

    def __init__(self, cache_dir: Path) -> None:
        self.images = read_csv(IMAGES_CSV, IMAGES_COLUMNS, required=True)
        self.negatives = read_csv(NEGATIVES_CSV, NEG_COLUMNS)
        self.annex = read_csv(ANNEX_CSV, ANNEX_COLUMNS)
        self.rejected = read_csv(REJECTED_CSV, REJECTED_COLUMNS)
        neg_ids = {r["image_id"] for r in self.negatives}
        orphans = [r["image_id"] for r in self.annex if r["image_id"] not in neg_ids]
        if orphans:
            fail(f"{ANNEX_CSV.name} has rows absent from negatives.csv ({', '.join(orphans[:5])}...): "
                 "fix the CSV files first (python scripts/check_repo.py)")
        self.done = {(r["dataset"], r["source_record_id"]) for r in self.annex}
        self.blocked = {(r["dataset"], r["source_record_id"]) for r in self.rejected}
        self.pending: list[tuple[Record, Processed, Path]] = []
        self.skip_log = cache_dir / "skipped.csv"
        self.skipped = set()
        if self.skip_log.exists():
            with open(self.skip_log, newline="", encoding="utf-8") as f:
                self.skipped = {(r["dataset"], r["source_record_id"]) for r in csv.DictReader(f)}
        PARTIAL_DIR.mkdir(parents=True, exist_ok=True)
        for leftover in PARTIAL_DIR.iterdir():  # interrupted before a commit
            leftover.unlink()

    def count(self, dataset: str) -> int:
        return sum(1 for r in self.annex if r["dataset"] == dataset)

    def group_counts(self) -> Counter:
        ids = {r["image_id"] for r in self.annex}
        return Counter(r["source_group"] for r in self.negatives if r["image_id"] in ids)

    def log_skip(self, record: Record, reason: str) -> None:
        new = not self.skip_log.exists()
        self.skip_log.parent.mkdir(parents=True, exist_ok=True)
        with open(self.skip_log, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["dataset", "source_record_id", "reason", "date"])
            w.writerow([record.dataset, record.record_id, reason, TODAY])
        self.skipped.add(record.key)

    def stage(self, record: Record, result: Processed) -> None:
        tmp = PARTIAL_DIR / f"{len(self.pending):06d}{result.ext}"
        tmp.write_bytes(result.data)
        self.pending.append((record, result, tmp))

    def next_number(self) -> int:
        numbers = [int(m.group(1)) for r in self.negatives + self.rejected
                   if (m := re.fullmatch(r"NEG-(\d+)", r["image_id"].strip()))]
        for folder in FOLDERS.values():
            if folder.is_dir():
                numbers += [int(m.group(1)) for p in folder.iterdir()
                            if (m := re.match(r"NEG-(\d+)", p.name))]
        return max(numbers, default=0) + 1

    def commit(self) -> int:
        if not self.pending:
            return 0
        number = self.next_number()
        new_neg, new_annex, moved = [], [], []
        try:
            for record, res, tmp in self.pending:
                image_id = f"NEG-{number:06d}"
                number += 1
                target = EXTERNAL_DIR / f"{image_id}{res.ext}"
                if target.exists():
                    fail(f"{target} already exists")
                lic = normalize_license(record.license_raw, record.license_url_raw)
                shutil.move(str(tmp), str(target))
                moved.append((target, tmp))
                new_neg.append({
                    "image_id": image_id, "filename": target.name, "source_url": record.source_url,
                    "creator": record.creator, "license": lic[0], "license_url": lic[1],
                    "rights_verified": "true", "sha256": sha256_bytes(res.data),
                    "source_group": record.source_group, "parent_id": "", "verified_on": TODAY,
                    "modifications": res.modifications, "availability": "external",
                    "description": f"Negative from {DATASET_LABELS[record.dataset]}: "
                                   f"{CATEGORY_LABELS.get(record.category, record.category)}",
                })
                new_annex.append({
                    "image_id": image_id, "dataset": record.dataset,
                    "source_record_id": record.record_id, "source_path": record.source_path,
                    "source_sha256": record.source_sha256, "original_width": res.original_width,
                    "original_height": res.original_height, "width": res.width, "height": res.height,
                    "processing": res.processing, "category": record.category,
                    "content": record.content, "site": record.site, "depth_m": record.depth_m,
                    "imaging_type": record.imaging_type, "attribution": record.attribution,
                    "dhash": res.dhash, "text_overlay": res.text_overlay, "collected_on": TODAY,
                    "collector_version": COLLECTOR_VERSION,
                })
            write_csvs([(NEGATIVES_CSV, NEG_COLUMNS, self.negatives + new_neg),
                        (ANNEX_CSV, ANNEX_COLUMNS, self.annex + new_annex)])
        except BaseException:
            for target, tmp in moved:  # roll back
                shutil.move(str(target), str(tmp))
            raise
        self.negatives += new_neg
        self.annex += new_annex
        self.done |= {(a["dataset"], a["source_record_id"]) for a in new_annex}
        n = len(self.pending)
        self.pending = []
        return n


# ========================================================== duplicate index --
def build_hash_index(store: Store, cache_dir: Path) -> ic.HashIndex:
    """dhash of every image of the collection (positives, negatives, external)."""
    index = ic.HashIndex()
    for a in store.annex:
        if a["dhash"]:
            index.add(a["dhash"], a["image_id"])
    external = {a["image_id"] for a in store.annex}
    cache_file = cache_dir / "dhash_cache.json"
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    for r in store.images + store.negatives:
        if r["image_id"] in external:
            continue
        path = FOLDERS.get(r["availability"], ROOT / "missing") / r["filename"]
        if path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff") or not path.exists():
            continue
        key = r["sha256"] or str(path)
        if key not in cache:
            image = ic.load_bgr(path)
            if image is None:
                continue
            cache[key] = ic.dhash(image)
        index.add(cache[key], r["image_id"])
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(cache))
    return index


# ================================================================= commands --
def parse_targets(args) -> dict:
    targets = dict(DEFAULT_TARGETS)
    if args.target:
        given = {}
        for t in args.target:
            name, _, value = t.partition("=")
            if name not in DATASETS or not value.isdigit():
                fail(f"--target {t}: expected one of {', '.join(DATASETS)}=N")
            given[name] = int(value)
        targets = {k: given.get(k, 0) for k in DATASETS} if not args.only else {**targets, **given}
    if args.only:
        targets = {k: (v if k in args.only else 0) for k, v in targets.items()}
    return targets


def ordered_candidates(source: Source, store: Store, args, remaining: int):
    records = source.candidates(remaining)
    blocked = store.done | store.blocked | (set() if args.retry_skipped else store.skipped)
    records = [r for r in records if r.key not in blocked]
    if source.keep_order:
        return records
    return interleave(records, cap_of(args), store.group_counts())


def cmd_plan(args) -> None:
    store = Store(Path(args.cache_dir))
    targets = parse_targets(args)
    for name, target in targets.items():
        if not target:
            continue
        have = store.count(name)
        log(f"\n== {DATASET_LABELS[name]}: target {target}, already collected {have}")
        if have >= target:
            continue
        source = SOURCES[name](args)
        records = ordered_candidates(source, store, args, target - have)
        log(source.describe())
        groups = Counter(r.source_group for r in records[: (target - have)])
        cats = Counter(r.category for r in records[: (target - have)])
        lics = Counter(normalize_license(r.license_raw)[0] for r in records[: (target - have)])
        log(f"{len(records)} candidates in order. The first {target - have} would come from "
            f"{len(groups)} source groups (largest: {groups.most_common(1)[0][1] if groups else 0} images).")
        log("Categories: " + ", ".join(f"{k} {v}" for k, v in cats.most_common()))
        log("Licences:   " + ", ".join(f"{k} {v}" for k, v in lics.most_common()))
        if len(records) < target - have:
            log(f"warning: only {len(records)} candidates for {target - have} images: "
                "relax the filters or lower the target")
    log("\nPlan only: nothing downloaded except metadata.")


def cmd_collect(args) -> None:
    cache_dir = Path(args.cache_dir)
    store = Store(cache_dir)
    targets = parse_targets(args)
    log("Indexing the existing collection for duplicate detection...")
    hashes = build_hash_index(store, cache_dir)
    log(f"{len(hashes)} images indexed.")
    summary = {}
    try:
        for name, target in targets.items():
            remaining = target - store.count(name)
            if target <= 0 or remaining <= 0:
                continue
            log(f"\n== {DATASET_LABELS[name]}: {remaining} images to collect (target {target})")
            source = SOURCES[name](args)
            records = ordered_candidates(source, store, args, remaining)
            log(source.describe())
            log(f"{len(records)} candidates.")
            accepted, reasons, clip_count = 0, Counter(), Counter()
            fetched = source.fetch(records)
            try:
                for record, data, error in fetched:
                    if accepted >= remaining:
                        break
                    if error:
                        reasons["download error"] += 1
                        if args.verbose:
                            log(f"  {record.record_id}: {error}")
                        continue
                    lic = normalize_license(record.license_raw, record.license_url_raw)
                    try:
                        if lic is None:
                            raise Skip("unknown licence")
                        result = process(data, record, args, nd="ND" in lic[2])
                        near = hashes.nearest(result.dhash, args.dup_distance)
                        if near:
                            raise Skip(f"near-duplicate of {near[0]} (distance {near[1]})")
                    except Skip as skip:
                        reason = str(skip)
                        reasons[reason.split(" (")[0] if not reason.startswith("near") else "near-duplicate"] += 1
                        store.log_skip(record, reason)
                        continue
                    store.stage(record, result)
                    hashes.add(result.dhash, f"{record.dataset}:{record.record_id}")
                    accepted += 1
                    clip_count[record.sub_group] += 1
                    if len(store.pending) >= args.batch:
                        store.commit()
                        log(f"  {accepted}/{remaining} committed")
            finally:
                close = getattr(fetched, "close", None)
                if close:
                    close()
            n = store.commit()
            if n:
                log(f"  {accepted}/{remaining} committed")
            summary[name] = (accepted, reasons)
            if reasons.get("SHA-256 differs from the source's", 0) > max(10, accepted):
                log("warning: most downloads do not match the SHA-256 given by the source. The source's "
                    "metadata may have changed meaning; check the adapter before going further.")
            if accepted < remaining:
                log(f"warning: candidates exhausted, {accepted}/{remaining} collected")
    except KeyboardInterrupt:
        n = store.commit()
        log(f"\nInterrupted: {n} pending image(s) committed. Run the same command again to continue.")
        finalize()
        sys.exit(130)
    log("\nSummary")
    for name, (accepted, reasons) in summary.items():
        log(f"  {DATASET_LABELS[name]}: {accepted} added, total now {store.count(name)}")
        for reason, n in reasons.most_common():
            log(f"      skipped, {reason}: {n}")
    finalize()


def finalize() -> None:
    try:
        import build_licenses
        current, new = build_licenses.expected_text()
        if current != new:
            build_licenses.LICENSES_MD.write_text(new, encoding="utf-8")
            log("metadata/LICENSES.md updated")
    except Exception as error:
        log(f"warning: could not update metadata/LICENSES.md ({error}); run python scripts/build_licenses.py")
    log("Next: python scripts/check_repo.py, then review a sample: "
        "python scripts/collect_negatives.py review --sample 200")


def cmd_restore(args) -> None:
    store = Store(Path(args.cache_dir))
    neg_by_id = {r["image_id"]: r for r in store.negatives}
    todo = defaultdict(list)
    for a in store.annex:
        neg = neg_by_id[a["image_id"]]
        path = EXTERNAL_DIR / neg["filename"]
        if path.exists() and sha256_file(path) == neg["sha256"]:
            continue
        if args.only and a["dataset"] not in args.only:
            continue
        todo[a["dataset"]].append((neg, a))
    if not todo:
        log("data/external/ is complete.")
        return
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    changed_hashes, failures = {}, 0
    for name, rows in todo.items():
        log(f"== {DATASET_LABELS[name]}: {len(rows)} file(s) to restore")
        source = SOURCES[name](args)
        by_record = {(a["source_record_id"]): (neg, a) for neg, a in rows}
        done = 0
        for record, data, error in source.fetch(source.records_from_annex(rows)):
            neg, a = by_record[record.record_id]
            if error or data is None:
                failures += 1
                log(f"  {neg['image_id']}: {error}")
                continue
            if a["source_sha256"] and sha256_bytes(data) != a["source_sha256"]:
                failures += 1
                log(f"  {neg['image_id']}: the source file changed (SHA-256 differs); not restored")
                continue
            try:
                res = process(data, record, args, nd=False, forced=parse_processing(a["processing"]))
            except Skip as skip:
                failures += 1
                log(f"  {neg['image_id']}: {skip}")
                continue
            (EXTERNAL_DIR / neg["filename"]).write_bytes(res.data)
            digest = sha256_bytes(res.data)
            if digest != neg["sha256"]:
                distance = ic.hamming(res.dhash, a["dhash"]) if a["dhash"] else 99
                changed_hashes[neg["image_id"]] = (digest, distance)
            done += 1
        log(f"  {done} restored")
    if changed_hashes:
        far = {k: v for k, v in changed_hashes.items() if v[1] > 2}
        log(f"\n{len(changed_hashes)} restored file(s) differ byte-wise from the recorded SHA-256 "
            "(another Pillow/libjpeg version re-encodes slightly differently).")
        if far:
            log(f"warning: {len(far)} of them also look different (perceptual hash): {', '.join(list(far)[:10])}")
        if args.update_hashes:
            for r in store.negatives:
                if r["image_id"] in changed_hashes:
                    r["sha256"] = changed_hashes[r["image_id"]][0]
            write_csvs([(NEGATIVES_CSV, NEG_COLUMNS, store.negatives)])
            log("sha256 values updated in negatives.csv (--update-hashes).")
        else:
            log("Images are usable as they are. To record the new hashes on this machine: "
                "restore --update-hashes (do not commit that change unless you mean it).")
    if failures:
        log(f"\n{failures} file(s) could not be restored.")
        sys.exit(1)


def cmd_review(args) -> None:
    store = Store(Path(args.cache_dir))
    neg_by_id = {r["image_id"]: r for r in store.negatives}
    rows = [a for a in store.annex if (not args.only or a["dataset"] in args.only)]
    if args.flagged:
        rows = [a for a in rows if a["text_overlay"]]
    rng = random.Random(args.seed)
    if args.sample and len(rows) > args.sample:
        rows = rng.sample(rows, args.sample)
    rows.sort(key=lambda a: a["image_id"])
    out_dir = STAGING_DIR / "review"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("sheet-*.jpg"):
        old.unlink()
    tile_w, tile_h, cols, per_sheet = 320, 200, 5, 30
    sheets = 0
    for start in range(0, len(rows), per_sheet):
        chunk = rows[start:start + per_sheet]
        n_rows = (len(chunk) + cols - 1) // cols
        sheet = np.full((n_rows * (tile_h + 22), cols * tile_w, 3), 255, np.uint8)
        for i, a in enumerate(chunk):
            image = ic.load_bgr(EXTERNAL_DIR / neg_by_id[a["image_id"]]["filename"])
            y, x = (i // cols) * (tile_h + 22), (i % cols) * tile_w
            if image is not None:
                s = min(tile_w / image.shape[1], tile_h / image.shape[0])
                small = cv2.resize(image, (int(image.shape[1] * s), int(image.shape[0] * s)),
                                   interpolation=cv2.INTER_AREA)
                sheet[y:y + small.shape[0], x:x + small.shape[1]] = small
            label = f"{a['image_id']} {a['dataset'][:4]} {a['category'][:6]}"
            cv2.putText(sheet, label, (x + 3, y + tile_h + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 180), 1)
        sheets += 1
        cv2.imwrite(str(out_dir / f"sheet-{sheets:03d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    log(f"{len(rows)} images on {sheets} sheet(s) in {out_dir.relative_to(ROOT)}/")
    log("Reject a bad one with: python scripts/collect_negatives.py reject NEG-xxxxxx --reason \"...\"")


def cmd_reject(args) -> None:
    store = Store(Path(args.cache_dir))
    ids = set(args.ids)
    annex = {a["image_id"]: a for a in store.annex}
    unknown = [i for i in ids if i not in annex]
    if unknown:
        fail(f"not collected by this script (not in {ANNEX_CSV.name}): {', '.join(sorted(unknown))}")
    neg_by_id = {r["image_id"]: r for r in store.negatives}
    rejected = store.rejected + [{
        "dataset": annex[i]["dataset"], "source_record_id": annex[i]["source_record_id"],
        "image_id": i, "reason": args.reason, "rejected_on": TODAY} for i in sorted(ids)]
    write_csvs([
        (NEGATIVES_CSV, NEG_COLUMNS, [r for r in store.negatives if r["image_id"] not in ids]),
        (ANNEX_CSV, ANNEX_COLUMNS, [a for a in store.annex if a["image_id"] not in ids]),
        (REJECTED_CSV, REJECTED_COLUMNS, rejected),
    ])
    for i in ids:
        (EXTERNAL_DIR / neg_by_id[i]["filename"]).unlink(missing_ok=True)
    log(f"{len(ids)} image(s) rejected; they will not be collected again. "
        "Run collect again to replace them.")
    finalize()


# ===================================================================== main --
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def common(s, selection: bool = True):
        s.add_argument("--cache-dir", default=str(DEFAULT_CACHE),
                       help="downloads and indexes (DeepFish archive: 7.1 GB). Default: .cache/negatives/")
        s.add_argument("--only", nargs="+", choices=DATASETS, help="restrict to these sources")
        s.add_argument("--seed", type=int, default=2026, help="selection seed (default 2026)")
        s.add_argument("--workers", type=int, default=8, help="parallel downloads (FathomNet)")
        s.add_argument("--verbose", action="store_true")
        s.add_argument("--max-side", type=int, default=1280, help="longest side after resizing (default 1280)")
        s.add_argument("--jpeg-quality", type=int, default=92)
        s.add_argument("--fathomnet-thumbnails", action="store_true",
                       help="use FathomNet's 512 px thumbnails instead of the originals (much lighter)")
        s.add_argument("--deepfish-tar", help="path to an already downloaded DeepFish.tar")
        s.add_argument("--reefsupport-revision", help="Hugging Face revision (default: latest)")
        s.add_argument("--reefsupport-images-glob", default="data/images/*.parquet")
        s.add_argument("--reefsupport-metadata-glob", default="data/metadata/*.parquet")
        if not selection:
            return
        s.add_argument("--target", action="append", metavar="SOURCE=N",
                       help="number of images wanted for a source, e.g. --target deepfish=100 "
                            "(repeatable; defaults: fathomnet=4000 deepfish=4000 reefsupport=2000)")
        s.add_argument("--min-side", type=int, default=320, help="minimum shorter side, in pixels (default 320)")
        s.add_argument("--max-per-group", type=int, default=0,
                       help="hard limit of images per source_group (dive, habitat, survey site). "
                            "Default: none; groups are interleaved round-robin anyway")
        s.add_argument("--dup-distance", type=int, default=4,
                       help="perceptual-hash distance under which an image is a near-duplicate (default 4)")
        s.add_argument("--keep-text", action="store_true",
                       help="keep images with burned-in text (location recorded) instead of skipping them")
        s.add_argument("--exclude-nc", action="store_true", help="skip NonCommercial licences")
        s.add_argument("--exclude-nd", action="store_true", help="skip NoDerivatives licences")
        s.add_argument("--retry-skipped", action="store_true", help="retry records skipped in earlier runs")
        s.add_argument("--batch", type=int, default=50, help="commit to the CSV files every N images")
        s.add_argument("--fathomnet-release", help="FathomNet release date (default: latest)")
        s.add_argument("--fathomnet-min-depth", type=float, help="keep FathomNet images at least this deep (m)")
        s.add_argument("--fathomnet-max-depth", type=float, help="keep FathomNet images at most this deep (m)")
        s.add_argument("--deepfish-per-clip", type=int, default=5,
                       help="frames kept per DeepFish video clip, evenly spaced (default 5)")
        s.add_argument("--deepfish-empty-share", type=float, default=0.5,
                       help="share of DeepFish frames without fish (default 0.5)")
        s.add_argument("--reefsupport-include-sources", default="",
                       help="comma-separated source_id patterns to keep (default: all)")
        s.add_argument("--reefsupport-exclude-sources", default="roboflow-*,noaa-pifsc-bleaching",
                       help="comma-separated source_id patterns to skip. Default: the Roboflow forks "
                            "(licence asserted by the uploader, sporadic watermarks) and the NOAA "
                            "224 px patches")
        s.add_argument("--reefsupport-per-rowgroup", type=int, default=40,
                       help="images taken from each Parquet row group downloaded (default 40)")
        s.add_argument("--reefsupport-overshoot", type=float, default=1.5,
                       help="candidates selected per image wanted, to absorb rejections (default 1.5)")

    s = sub.add_parser("plan", help="show what would be collected (downloads metadata only)")
    common(s)
    s.set_defaults(func=cmd_plan)
    s = sub.add_parser("collect", help="download, check and add images (resumable)")
    common(s)
    s.set_defaults(func=cmd_collect)
    s = sub.add_parser("restore", help="rebuild data/external/ from the CSV files")
    common(s, selection=False)
    s.add_argument("--update-hashes", action="store_true",
                   help="record the SHA-256 of re-encoded files that differ byte-wise")
    s.set_defaults(func=cmd_restore)
    s = sub.add_parser("review", help="contact sheets of collected images, in staging/review/")
    s.add_argument("--sample", type=int, default=200, help="random sample size (0: all)")
    s.add_argument("--flagged", action="store_true", help="only images kept with --keep-text")
    s.add_argument("--only", nargs="+", choices=DATASETS)
    s.add_argument("--seed", type=int, default=2026)
    s.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    s.set_defaults(func=cmd_review)
    s = sub.add_parser("reject", help="remove collected images and never collect them again")
    s.add_argument("ids", nargs="+", help="NEG-xxxxxx identifiers")
    s.add_argument("--reason", required=True)
    s.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    s.set_defaults(func=cmd_reject)
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
