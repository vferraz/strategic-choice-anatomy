#!/usr/bin/env python3
"""Fetch, verify and unpack the released data deposit.

The repository ships every table a figure reads (``data/``, ~40 MB, git-tracked). The raw
substrate, activations and steering outputs are far too large for git and live in a Zenodo
deposit; this script fetches them into ``data_heavy/`` in the layout
``strategic_anatomy.config`` expects.

    python scripts/download_data.py                          # everything (~21 GB)
    python scripts/download_data.py --component layerc       # one component (34 MB)
    python scripts/download_data.py --list                   # sizes, download nothing
    python scripts/download_data.py --dest /mnt/big/sca      # somewhere with room
    python scripts/download_data.py --verify-only            # re-check what is on disk
    python scripts/download_data.py --verify-committed       # check data/ against MANIFEST

Already-complete components are skipped unless ``--force`` is given, so an interrupted run
resumes by re-invoking the same command.

If you already have a copy of the deposit, you do not need this script at all -- point
``SCA_DATA_ROOT`` at it::

    export SCA_DATA_ROOT=/path/to/deposit

Standard library only, so it runs before the project's dependencies are installed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------------------
# Deposit coordinates. Filled in when the Zenodo record is published (Phase 7); until then
# the script explains what to do rather than failing with a connection error.
# ---------------------------------------------------------------------------------------
ZENODO_RECORD_ID: str | None = None          # e.g. "10123456"
ZENODO_API = "https://zenodo.org/api/records/{record_id}"
ZENODO_DOI: str | None = None                # e.g. "10.5281/zenodo.10123456"

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "data" / "MANIFEST.json"

CHUNK = 1 << 20  # 1 MiB


def _fmt(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:,.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} TB"


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        sys.exit(f"manifest not found at {MANIFEST_PATH} — is this a complete clone?")
    return json.loads(MANIFEST_PATH.read_text())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(CHUNK)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------
# Verify the git-tracked tables
# ---------------------------------------------------------------------------------------

def verify_committed(manifest: dict) -> int:
    files = manifest["files"]
    missing, bad, ok = [], [], 0
    for rel, rec in sorted(files.items()):
        p = REPO_ROOT / rel
        if not p.exists():
            missing.append(rel)
            continue
        if sha256_file(p) != rec["sha256"]:
            bad.append(rel)
            continue
        ok += 1
    print(f"committed tables: {ok}/{len(files)} verified")
    for rel in missing:
        print(f"  MISSING   {rel}")
    for rel in bad:
        print(f"  MISMATCH  {rel}")
    if missing or bad:
        print("\nA mismatch means your copy differs from the released one. Re-clone, or check "
              "whether a build script overwrote a committed table.")
        return 1
    return 0


# ---------------------------------------------------------------------------------------
# Deposit components
# ---------------------------------------------------------------------------------------

def components(manifest: dict) -> dict[str, dict]:
    return {c["name"]: c for c in manifest["deposit"]["components"]}


def component_complete(dest: Path, name: str) -> bool:
    """A component counts as present if its root exists and holds at least one file."""
    root = dest / name
    if not root.exists():
        return False
    for _ in root.rglob("*"):
        return True
    return False


def resolve_urls(record_id: str) -> dict[str, tuple[str, str | None, int]]:
    """Map archive filename -> (download url, checksum, size) from the Zenodo record."""
    url = ZENODO_API.format(record_id=record_id)
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            record = json.loads(resp.read().decode())
    except urllib.error.URLError as exc:
        sys.exit(f"could not reach Zenodo record {record_id}: {exc}")
    out = {}
    for entry in record.get("files", []):
        key = entry.get("key") or entry.get("filename")
        link = (entry.get("links") or {}).get("self")
        checksum = entry.get("checksum", "")
        if checksum.startswith("md5:"):
            checksum = None  # we verify with sha256 from our own manifest instead
        out[key] = (link, checksum, int(entry.get("size", 0)))
    return out


def download(url: str, target: Path, expected_bytes: int = 0) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    done = 0
    with urllib.request.urlopen(url, timeout=120) as resp, tmp.open("wb") as fh:
        total = expected_bytes or int(resp.headers.get("Content-Length") or 0)
        while True:
            block = resp.read(CHUNK)
            if not block:
                break
            fh.write(block)
            done += len(block)
            if total:
                pct = 100 * done / total
                print(f"\r    {_fmt(done)} / {_fmt(total)}  ({pct:5.1f}%)", end="", flush=True)
            else:
                print(f"\r    {_fmt(done)}", end="", flush=True)
    print()
    tmp.replace(target)


def unpack(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:*") as tf:
        # Refuse absolute paths and parent-directory escapes.
        for member in tf.getmembers():
            name = Path(member.name)
            if name.is_absolute() or ".." in name.parts:
                sys.exit(f"refusing to unpack unsafe path from {archive.name}: {member.name}")
        # `filter="data"` is the safe extraction mode; it is the default from Python 3.14.
        try:
            tf.extractall(dest, filter="data")
        except TypeError:
            tf.extractall(dest)


def verify_component(dest: Path, name: str, expected_sha: str | None,
                     archive: Path | None) -> bool:
    if expected_sha and archive and archive.exists():
        actual = sha256_file(archive)
        if actual != expected_sha:
            print(f"  FAIL   {name}: archive sha256 {actual[:16]}… != {expected_sha[:16]}…")
            return False
        print(f"  OK     {name}: archive sha256 matches the manifest")
        return True
    if component_complete(dest, name):
        n = sum(1 for _ in (dest / name).rglob("*"))
        print(f"  PRESENT {name}: {n} entries under {dest / name}"
              + ("" if expected_sha else "  (no published checksum yet)"))
        return True
    print(f"  ABSENT {name}: nothing at {dest / name}")
    return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--component", action="append", metavar="NAME",
                    help="fetch only this component (repeatable); default: all")
    ap.add_argument("--dest", type=Path, default=None,
                    help="where to unpack (default: $SCA_DATA_ROOT, else <repo>/data_heavy)")
    ap.add_argument("--list", action="store_true", help="show components and sizes, fetch nothing")
    ap.add_argument("--dry-run", action="store_true", help="show what would be fetched")
    ap.add_argument("--force", action="store_true", help="re-fetch components already present")
    ap.add_argument("--keep-archives", action="store_true",
                    help="keep the downloaded .tar.gz files after unpacking")
    ap.add_argument("--verify-only", action="store_true",
                    help="verify what is already on disk; download nothing")
    ap.add_argument("--verify-committed", action="store_true",
                    help="verify the git-tracked data/ tree against data/MANIFEST.json")
    args = ap.parse_args(argv)

    manifest = load_manifest()

    if args.verify_committed:
        return verify_committed(manifest)

    comps = components(manifest)

    if args.dest is not None:
        dest = args.dest.expanduser().resolve()
    else:
        import os
        env = os.environ.get("SCA_DATA_ROOT")
        dest = Path(env).expanduser().resolve() if env else REPO_ROOT / "data_heavy"

    wanted = args.component or list(comps)
    unknown = [c for c in wanted if c not in comps]
    if unknown:
        sys.exit(f"unknown component(s): {', '.join(unknown)}\n"
                 f"available: {', '.join(comps)}")

    if args.list or args.dry_run:
        print(f"destination: {dest}\n")
        print(f"{'component':26s} {'size':>12s}  {'status':9s} contents")
        print("-" * 100)
        total = 0
        for name in wanted:
            c = comps[name]
            size = c["bytes_uncompressed"]
            total += size
            status = "present" if component_complete(dest, name) else "missing"
            print(f"{name:26s} {_fmt(size):>12s}  {status:9s} {c['contents']}")
        print("-" * 100)
        print(f"{'TOTAL':26s} {_fmt(total):>12s}")
        if ZENODO_RECORD_ID is None:
            print("\nNote: the Zenodo record id is not yet set in this script "
                  "(ZENODO_RECORD_ID); sizes come from data/MANIFEST.json.")
        return 0

    if args.verify_only:
        print(f"verifying {dest}\n")
        allok = True
        for name in wanted:
            archive = dest / comps[name]["archive"] if args.keep_archives else None
            allok &= verify_component(dest, name, comps[name].get("sha256"), archive)
        return 0 if allok else 1

    if ZENODO_RECORD_ID is None:
        print("The data deposit has not been published yet, so this script has no record to "
              "fetch from.\n")
        print("Once it is, this script will download and verify automatically. In the "
              "meantime, if you already have a copy of the deposit tree, point the code at "
              "it:\n")
        print("    export SCA_DATA_ROOT=/path/to/deposit\n")
        print("Expected layout (see docs/DATA.md):\n")
        for name in wanted:
            print(f"    $SCA_DATA_ROOT/{comps[name]['layout']}")
        print("\nEverything that does not need the deposit — the Tier-1 figures and every "
              "committed table — already works from this clone:\n")
        print("    make figures-tier1\n")
        return 2

    urls = resolve_urls(ZENODO_RECORD_ID)
    dest.mkdir(parents=True, exist_ok=True)
    failed = []

    for name in wanted:
        c = comps[name]
        if component_complete(dest, name) and not args.force:
            print(f"[{name}] already present at {dest / name} — skipping (--force to refetch)")
            continue
        archive_name = c["archive"]
        if archive_name not in urls:
            print(f"[{name}] FAIL: {archive_name} is not in Zenodo record {ZENODO_RECORD_ID}")
            failed.append(name)
            continue
        url, _zchecksum, size = urls[archive_name]
        archive = dest / archive_name
        print(f"[{name}] downloading {archive_name} ({_fmt(size or c['bytes_uncompressed'])})")
        download(url, archive, size)

        expected = c.get("sha256")
        if expected:
            actual = sha256_file(archive)
            if actual != expected:
                print(f"[{name}] FAIL: sha256 {actual} != manifest {expected}")
                failed.append(name)
                continue
            print(f"[{name}] sha256 OK")
        else:
            print(f"[{name}] no published checksum in the manifest — skipping hash check")

        print(f"[{name}] unpacking into {dest}")
        unpack(archive, dest)
        if not args.keep_archives:
            archive.unlink(missing_ok=True)
        print(f"[{name}] done")

    if failed:
        print(f"\nFAILED: {', '.join(failed)}")
        return 1

    free = shutil.disk_usage(dest).free
    print(f"\nall requested components present under {dest} ({_fmt(free)} free)")
    print("Verify any time with:  python scripts/download_data.py --verify-only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
