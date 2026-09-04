"""Fetch the researcher-generated public MediRely artifacts and verify SHA256.

This helper NEVER downloads gated MedSigLIP weights (google/medsiglip-448) or any
IU Chest X-Ray source images/reports — those must be obtained from their official
sources (see README). It only fetches the four audited-safe, researcher-generated
artifacts listed in public_artifacts/ARTIFACTS.md, from a base URL you provide
(e.g. a GitHub Release, or your own Google Drive direct-download links), and checks
each file's SHA256 against the manifest before use.

Usage:
  # from a GitHub Release (files named exactly as below):
  python scripts/download_public_artifacts.py --base-url https://github.com/<you>/MediRely_GTC/releases/download/v1.0

  # or from explicit per-file URLs (e.g. Google Drive direct links):
  python scripts/download_public_artifacts.py --url best_iu_full_image_text_xrv.pth=https://drive.google.com/uc?id=...

By default writes into public_artifacts/. Skips files already present with a matching hash.
"""
import argparse
import hashlib
import json
import os
import urllib.request

# name -> expected sha256 (must match public_artifacts/ARTIFACTS.md)
EXPECTED = {
    "best_iu_full_image_text_xrv.pth": "93c4761b830d5e5323ef812fab0fadaf0cc9a3689d2a46ad59299619a097f3ce",
    "medsiglip_train_embeddings.npz":  "5f2c57b622b7f842b29a760fad51b6c071835da6367581c36c335045cdbc924d",
    "iu_train_memory_bank_public.pt":  "e12d02ab82903f5dc4d7423c1e99cd302f5029984e02cc49c4c4c9c99ef279b6",
    "step4_medsiglip_comparison.json": "fea744270095fed8a8f304e06d5a1a7c30858c9c7ccb4f0b550695456abac295",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", help="base URL; files are fetched as <base>/<filename>")
    p.add_argument("--url", action="append", default=[],
                   help="explicit name=URL override (repeatable), e.g. Google Drive direct link")
    p.add_argument("--dest", default="public_artifacts")
    return p.parse_args()


def main():
    args = parse_args()
    overrides = dict(kv.split("=", 1) for kv in args.url)
    os.makedirs(args.dest, exist_ok=True)
    failed = []
    for name, want in EXPECTED.items():
        dst = os.path.join(args.dest, name)
        if os.path.isfile(dst) and sha256(dst) == want:
            print(f"[ok]   {name} already present, sha matches")
            continue
        url = overrides.get(name) or (f"{args.base_url.rstrip('/')}/{name}" if args.base_url else None)
        if not url:
            print(f"[skip] {name}: no --base-url or --url given")
            failed.append(name)
            continue
        print(f"[get]  {name} <- {url}")
        urllib.request.urlretrieve(url, dst)  # only the URL you explicitly provide
        got = sha256(dst)
        if got != want:
            print(f"[FAIL] {name}: sha256 mismatch\n       expected {want}\n       got      {got}")
            failed.append(name)
        else:
            print(f"[ok]   {name}: sha256 verified")
    if failed:
        raise SystemExit(f"{len(failed)} artifact(s) missing/failed: {failed}")
    print("[done] all public artifacts present and verified.")
    print("Note: MedSigLIP weights and IU X-Ray images are NOT fetched here — see README.")


if __name__ == "__main__":
    main()
