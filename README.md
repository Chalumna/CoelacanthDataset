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

The repository separates images into two collections.

### Unrestricted collection

`data/unrestricted/`

Images for which redistribution without mandatory attribution has been verified, including:

- works released under CC0;
- works confirmed to be in the public domain;
- images covered by explicit permissions allowing public redistribution without mandatory attribution.

The applicable legal basis is documented for each image. This directory does not imply that every image is under CC0 or that all images have identical legal status.

### Images with additional conditions

`data/with-conditions/`

Images that may be redistributed under their respective licences, but require attribution or impose other conditions.

Each image must have a documented source, rights holder or credited creator where known, licence, and applicable conditions.

Images with unclear rights, or licences that do not permit the intended redistribution, are excluded from both collections.

## Data and metadata

Image files are accompanied by structured metadata, including where available:

- source URL and original file identifier;
- creator and rights holder;
- licence and licence URL;
- verification status and date;
- species identification and whether the specimen is living;
- image hash and modification history;
- individual identity, only when reliably established.

See [LICENSES.md](LICENSES.md) for the human-readable rights register and `metadata/` for machine-readable records.

## Scientific integrity

Species identifications and other scientific annotations are reviewed where possible. Uncertain identifications are marked as such rather than presented as confirmed facts.

Multiple photographs must not be assumed to depict different individuals. Individual identities are recorded only when supported by reliable evidence.

Exact observation sites and sensitive location metadata are not published, to help protect coelacanths.

## Intended use

The collection is designed to support reproducible research and open-source computer vision.

The presence of an image in this repository does not establish that it is suitable for every use. Users must follow the licence and any applicable conditions recorded for the specific image.

The right to redistribute an image does not automatically settle every question concerning model training, privacy, other third-party rights or applicable law.

## Repository layout

```text
data/
  unrestricted/
  with-conditions/
metadata/
  images.csv
  licenses.csv
scripts/
docs/
```

## Contributing

Contributions are welcome, especially:

- finding potential sources of openly reusable images;
- verifying image provenance and licences;
- correcting species identifications and metadata;
- obtaining explicit permissions from photographers and research institutions.

Please provide the original source and licence evidence when proposing an image. Do not upload images with unclear redistribution rights.

## Licence

The code and scripts are licensed under the MIT License, as specified in [LICENSE](LICENSE).

**The MIT License does not apply to the image files.** Each image retains its own copyright status, licence and conditions, documented in [LICENSES.md](LICENSES.md).

## Status

Early stage. The collection is being built progressively. The number of images, their scientific quality and their licensing status will evolve as sources are reviewed.

## Notes

This is an independent project within the Chalumna organisation. It is not affiliated with or endorsed by any research institution unless explicitly stated.

## References

- Thornycroft, R. E. & Booth, A. J. (2012). *Computer-aided identification of coelacanths, Latimeria chalumnae, using scale patterns*. Marine Biology Research, 8, 300–306. https://doi.org/10.1080/17451000.2011.628679
