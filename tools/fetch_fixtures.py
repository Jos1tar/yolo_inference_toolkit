"""Pick parity fixture images from COCO val2017 by a fixed, documented rule.

    python tools/fetch_fixtures.py            # -> tests/fixtures/*.jpg + ATTRIBUTION.md

Why COCO val2017: the M1 models are trained on COCO train2017, so val images are
unseen but in-distribution, and every image carries its Flickr license.

Selection rule (deterministic, no hand-picking):
  * license is redistributable without NC/ND terms: CC BY, CC BY-SA,
    "No known copyright restrictions", "United States Government Work"
  * no crowd annotations, >= 4 objects, >= 3 distinct categories,
    at least one of person / car / chair
  * one image per aspect-ratio bucket: landscape 4:3, portrait, square, wide
  * within a bucket: most distinct categories first, then most objects, then lowest image id

Downloads the 241 MB annotation zip once into models/_coco/ (gitignored).
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
ANN_MEMBER = "annotations/instances_val2017.json"
IMG_URL = "http://images.cocodataset.org/val2017/{file_name}"

ALLOWED_LICENSES = {
    "Attribution License",
    "Attribution-ShareAlike License",
    "No known copyright restrictions",
    "United States Government Work",
}
ANCHOR_CATEGORIES = {"person", "car", "chair"}

# name -> predicate on aspect = width / height
BUCKETS = {
    "landscape": lambda a: 1.25 <= a < 1.45,
    "portrait": lambda a: a <= 0.8,
    "square": lambda a: 0.9 <= a <= 1.1,
    "wide": lambda a: a >= 1.6,
}


def load_annotations(cache: Path) -> dict:
    js = cache / "instances_val2017.json"
    if not js.exists():
        cache.mkdir(parents=True, exist_ok=True)
        print(f"downloading {ANN_URL} (241 MB, once) ...")
        with urllib.request.urlopen(ANN_URL) as r:
            data = r.read()
        with zipfile.ZipFile(io.BytesIO(data)) as z, z.open(ANN_MEMBER) as src, open(js, "wb") as dst:
            shutil.copyfileobj(src, dst)
    return json.loads(js.read_text(encoding="utf-8"))


def select(coco: dict) -> dict[str, dict]:
    """Return {bucket: image_record} following the rule in the module docstring."""
    licenses = {lic["id"]: lic for lic in coco["licenses"]}
    cat_name = {c["id"]: c["name"] for c in coco["categories"]}

    anns = defaultdict(list)
    for a in coco["annotations"]:
        anns[a["image_id"]].append(a)

    scored = []
    for img in coco["images"]:
        if licenses[img["license"]]["name"] not in ALLOWED_LICENSES:
            continue
        objs = anns.get(img["id"], [])
        if not objs or any(a["iscrowd"] for a in objs):
            continue
        cats = {cat_name[a["category_id"]] for a in objs}
        if len(objs) < 4 or len(cats) < 3 or not cats & ANCHOR_CATEGORIES:
            continue
        aspect = img["width"] / img["height"]
        scored.append(((-len(cats), -len(objs), img["id"]), aspect, img, sorted(cats), len(objs)))
    scored.sort(key=lambda t: t[0])

    picked: dict[str, dict] = {}
    for bucket, ok in BUCKETS.items():
        for _, aspect, img, cats, n in scored:
            if ok(aspect) and img["id"] not in {p["id"] for p in picked.values()}:
                picked[bucket] = {**img, "categories": cats, "n_objects": n,
                                  "license_name": licenses[img["license"]]["name"],
                                  "license_url": licenses[img["license"]]["url"]}  # fmt: skip
                break
    return picked


def write_attribution(picked: dict[str, dict], out: Path) -> None:
    lines = [
        "# Fixture images",
        "",
        "From the COCO 2017 validation set (https://cocodataset.org), selected by `tools/fetch_fixtures.py`",
        "with a fixed rule (see the script docstring). Each image keeps its original Flickr license.",
        "",
        "| file | bucket | size | objects | categories | license | source |",
        "|---|---|---|---|---|---|---|",
    ]
    for bucket, im in picked.items():
        lines.append(
            f"| coco_{im['id']}.jpg | {bucket} | {im['width']}x{im['height']} | {im['n_objects']} | "
            f"{', '.join(im['categories'])} | [{im['license_name']}]({im['license_url']}) | {im['flickr_url']} |"
        )
    lines += ["", "COCO annotations are licensed under CC BY 4.0 by the COCO Consortium."]
    (out / "ATTRIBUTION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("tests/fixtures"))
    ap.add_argument("--cache", type=Path, default=Path("models/_coco"))
    args = ap.parse_args()

    picked = select(load_annotations(args.cache))
    missing = set(BUCKETS) - set(picked)
    if missing:
        print(f"warning: no image matched buckets {sorted(missing)}")

    args.out.mkdir(parents=True, exist_ok=True)
    for bucket, im in picked.items():
        dst = args.out / f"coco_{im['id']}.jpg"
        if not dst.exists():
            with urllib.request.urlopen(IMG_URL.format(file_name=im["file_name"])) as r:
                dst.write_bytes(r.read())
        size = f"{im['width']}x{im['height']}"
        print(f"{bucket:<10} {dst.name}  {size}  {im['license_name']}  {', '.join(im['categories'])}")
    write_attribution(picked, args.out)
    print(f"wrote {args.out / 'ATTRIBUTION.md'}")


if __name__ == "__main__":
    main()
