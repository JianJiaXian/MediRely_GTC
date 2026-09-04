"""Prepare LOCAL IU Chest X-Ray demo/bank image assets for the MediRely live demo.

This helper does NOT download, scrape, or redistribute any dataset. The IU Chest
X-Ray (OpenI / Indiana University) images are a gated/licensed third-party source
you must obtain yourself from the official provider. This script only:

  1. checks that you have pointed it at a local directory of IU X-Ray PNGs,
  2. copies (or symlinks) the three demo images referenced by your
     gallery_sources.json into gtc_demo/artifacts/demo_images/ (gitignored),
  3. reports which files are present/missing so the live demo can find them.

Usage:
  python scripts/prepare_local_iu_assets.py --iu-image-dir /path/to/iu_xray/images \
         --gallery configs/gallery_sources.example.json [--link]

Nothing here reaches the network. If --iu-image-dir is missing, it prints where to
obtain the data officially and exits without doing anything.
"""
import argparse
import json
import os
import shutil

OFFICIAL = ("Obtain IU Chest X-Ray images from the official Open-i / Indiana "
            "University source: https://openi.nlm.nih.gov/  (respect its license). "
            "This project does not redistribute the images.")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--iu-image-dir", required=True,
                   help="local directory containing IU X-Ray PNGs you already obtained")
    p.add_argument("--gallery", default="gtc_demo/artifacts/gallery_sources.json",
                   help="gallery_sources.json (or the .example.json template) listing demo image paths")
    p.add_argument("--dest", default="gtc_demo/artifacts/demo_images",
                   help="local destination for the 3 demo images (gitignored)")
    p.add_argument("--link", action="store_true",
                   help="symlink instead of copy (saves disk)")
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.isdir(args.iu_image_dir):
        print(f"[prepare] --iu-image-dir not found: {args.iu_image_dir}")
        print("[prepare] " + OFFICIAL)
        raise SystemExit(2)
    if not os.path.isfile(args.gallery):
        print(f"[prepare] gallery file not found: {args.gallery}")
        raise SystemExit(2)

    os.makedirs(args.dest, exist_ok=True)
    gallery = json.load(open(args.gallery, encoding="utf-8"))
    present, missing = [], []
    for key, entry in gallery.items():
        if not isinstance(entry, dict) or "image_path" not in entry:
            continue
        fname = os.path.basename(str(entry["image_path"]))
        src = os.path.join(args.iu_image_dir, fname)
        dst = os.path.join(args.dest, fname)
        if not os.path.isfile(src):
            missing.append(fname)
            continue
        if os.path.exists(dst) or os.path.islink(dst):
            os.remove(dst)
        if args.link:
            os.symlink(os.path.abspath(src), dst)
        else:
            shutil.copy2(src, dst)
        present.append(fname)

    print(f"[prepare] prepared {len(present)} demo image(s) into {args.dest}")
    for f in present:
        print(f"          ok  {f}")
    if missing:
        print(f"[prepare] {len(missing)} image(s) NOT found in {args.iu_image_dir}:")
        for f in missing:
            print(f"          --  {f}")
        print("[prepare] " + OFFICIAL)
        raise SystemExit(1)
    print("[prepare] done. Point gallery_sources.json image_path values at these files.")


if __name__ == "__main__":
    main()
