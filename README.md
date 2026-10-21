# CoelacanthDataset

An openly documented image collection for coelacanth research and conservation.

Part of the [Chalumna](https://github.com/Chalumna) organisation.

## Goal

This repository collects, documents and verifies images of coelacanths (*Latimeria*) for scientific research and open-source computer vision projects.

The goal is to make legally reusable images easier to find, while preserving their provenance, licensing information and scientific context.

The collection is intended to support several tasks, including:

- detecting living coelacanths in images and videos;
- individual re-identification from body spot patterns;
- developing and evaluating computer vision methods for coelacanth research.

## Collection policy

Images are reviewed individually before inclusion. A publicly accessible image is not necessarily free to redistribute.

The repository separates images into three levels, recorded in the `availability` column of `metadata/images.csv`.

### Unrestricted collection

`data/unrestricted/` (`availability`: `public`)

Images for which redistribution without mandatory attribution has been verified, including:

- works released under CC0;
- works confirmed to be in the public domain;
- images covered by explicit permissions allowing public redistribution without mandatory attribution.

The applicable legal basis is documented for each image. This directory does not imply that every image is under CC0 or that all images have identical legal status.

### Images with additional conditions

`data/with-conditions/` (`availability`: `public_with_conditions`)

Images that may be redistributed under their respective licences, but require attribution or impose other conditions.

Each image must have a documented source, rights holder or credited creator where known, licence, and applicable conditions.

Images with unclear rights, or licences that do not permit the intended redistribution, are never published here (see below).

### Local-only images

`data/local-only/` (`availability`: `local_only`)

Images whose redistribution rights are unclear, unverified or not yet confirmed are kept on the maintainer's machine and are never pushed to this repository. They are still listed in `metadata/images.csv`, so that the record of what the full collection contains is complete, but the files themselves are not part of this repository.

### Frames and negative examples

Some images are frames extracted from videos of the collection. A frame keeps the metadata of its video, and its `parent_id` points to it.

Frames that show no coelacanth are stored as negative examples, with `NEG-` identifiers, in `metadata/negatives.csv`. When the source is a video filmed in a single place, they let a model learn what the same scene looks like without a coelacanth.

## Data and metadata

Image files are accompanied by structured metadata, including where available:

- source URL and original file identifier;
- creator and rights holder;
- licence and licence URL;
- verification status and date;
- species identification and whether the specimen is living;
- image hash and modification history;
- the origin group of the image (`source_group`) and the file it was derived from (`parent_id`);
- individual identity, only when reliably established.

See [metadata/LICENSES.md](metadata/LICENSES.md) for the human-readable rights register and `metadata/` for machine-readable records.

### Main columns of `metadata/images.csv`

| Column | Meaning |
|---|---|
| `image_id`, `filename` | `COEL-xxxxxx` identifier and file name. `NEG-xxxxxx` identifiers are used in `negatives.csv`. |
| `species`, `live_status` | Species and whether the animal is alive. |
| `source_url`, `creator` | Original page and credited creator. |
| `license`, `license_url`, `rights_verified`, `verified_on` | Licence, link to its text, whether it was checked against the source, and when. |
| `sha256` | Hash of the file. |
| `source_group` | Origin of the image: same photographer, observation, video or filming session. |
| `parent_id` | Identifier of the file this one was derived from (frame of a video, crop of an image). |
| `modifications` | What was changed from the original. |
| `availability` | `public`, `public_with_conditions` or `local_only`. |

## Scientific integrity

Species identifications and other scientific annotations are reviewed where possible. Uncertain identifications are marked as such rather than presented as confirmed facts.

Multiple photographs must not be assumed to depict different individuals. Individual identities are recorded only when supported by reliable evidence.

Exact observation sites and sensitive location metadata are not published, to help protect coelacanths.

## Intended use

The collection is designed to support reproducible research and open-source computer vision.

When training or evaluating a model, split the data by `source_group`, not image by image: frames of the same video, or photographs from the same observation, look alike and must stay on the same side of a split.

The presence of an image in this repository does not establish that it is suitable for every use. Users must follow the licence and any applicable conditions recorded for the specific image.

The right to redistribute an image does not automatically settle every question concerning model training, privacy, other third-party rights or applicable law.

## Repository layout

```text
data/
  unrestricted/        public images: CC0 or public domain
  with-conditions/     public images under conditions (attribution, non-commercial...)
  local-only/          never pushed: images are only listed in the CSV
metadata/
  images.csv           one row per image or video (COEL-)
  negatives.csv        one row per negative example (NEG-)
  LICENSES.md          rights register, partly generated from the CSV files
scripts/
  extract_frames.py    extract frames from a video of the collection
  check_repo.py        integrity checks
  build_licenses.py    regenerate the tables of LICENSES.md
  requirements.txt
```

`raw/` (source videos) and `staging/` (frames waiting to be reviewed) are local working folders and are not tracked by Git.

## Scripts

```bash
pip install -r scripts/requirements.txt

# extract frames from a video of the collection (see the script's help)
python scripts/extract_frames.py stage --parent COEL-000014 --count 30
python scripts/extract_frames.py commit --parent COEL-000014

python scripts/build_licenses.py   # regenerate metadata/LICENSES.md
python scripts/check_repo.py       # integrity checks, run before every commit
```

## Contributing

Contributions are welcome, especially:

- finding potential sources of openly reusable images;
- verifying image provenance and licences;
- correcting species identifications and metadata;
- obtaining explicit permissions from photographers and research institutions.

Please provide the original source and licence evidence when proposing an image. Do not upload images with unclear redistribution rights. Run `python scripts/check_repo.py` before opening a pull request.

## Licence

The code and scripts are licensed under the MIT License, as specified in [LICENSE](LICENSE).

**The MIT License does not apply to the image files.** Each image retains its own copyright status, licence and conditions, documented in [metadata/LICENSES.md](metadata/LICENSES.md).

## Status

Early stage. The collection is being built progressively. The number of images, their scientific quality and their licensing status will evolve as sources are reviewed.

## Notes

This is an independent project within the Chalumna organisation. It is not affiliated with or endorsed by any research institution unless explicitly stated.

## References

- Thornycroft, R. E. & Booth, A. J. (2012). *Computer-aided identification of coelacanths, Latimeria chalumnae, using scale patterns*. Marine Biology Research, 8, 300–306. https://doi.org/10.1080/17451000.2011.628679
- Décamps et al. (2017). *Data from: The third dimension: a novel set-up for filming coelacanths in their natural environment*. Dryad. https://doi.org/10.5061/dryad.rh07b
- Sakaue et al. (2021). Frontiers in Marine Science. https://doi.org/10.3389/fmars.2021.755275
