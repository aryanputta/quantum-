"""
Download script for network topology datasets used in quantum routing experiments.

Datasets:
  - SNAP email-EuAll  : European email communication graph
  - Rocketfuel AS1221 : Telstra autonomous system topology
  - CAIDA AS topology : Autonomous System relationship graph

Each dataset is saved under data/snap/, data/rocketfuel/, data/caida/
respectively.  When a network download fails (timeout, 404, no connectivity)
a realistic synthetic fallback is generated and saved in the same location so
downstream code can always find a file.

Usage
-----
    python data/download_datasets.py
    python data/download_datasets.py --data-root /custom/path --timeout 30
"""

from __future__ import annotations

import argparse
import gzip
import logging
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset catalogue
# ---------------------------------------------------------------------------

DATASETS = {
    "snap": {
        "name": "SNAP email-EuAll",
        "url": "https://snap.stanford.edu/data/email-EuAll.txt.gz",
        "filename": "email-EuAll.txt.gz",
        "format": "snap_edgelist",
        "description": "European email communication network (~265k nodes, 420k edges)",
    },
    "rocketfuel": {
        "name": "Rocketfuel AS1221 (Telstra)",
        "url": "https://research.cs.washington.edu/networking/rocketfuel/maps/weights/1221.r0.cch.gz",
        "filename": "1221.r0.cch.gz",
        "format": "rocketfuel_cch",
        "description": "Rocketfuel AS1221 router-level topology with weights",
    },
    "caida": {
        "name": "CAIDA AS topology",
        "url": "https://publicdata.caida.org/datasets/topology/as-relationships/serial-1/20231001.as-rel.txt.bz2",
        "filename": "caida-as-rel.txt.bz2",
        "format": "caida_as_rel",
        "description": "CAIDA AS relationship dataset (provider-customer / peer-peer)",
    },
}

# Default chunk size for streaming downloads (bytes)
_CHUNK_SIZE = 65_536  # 64 KiB


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _progress_bar(downloaded: int, total: Optional[int], width: int = 40) -> str:
    """Return a text progress bar string."""
    if total and total > 0:
        frac = min(downloaded / total, 1.0)
        filled = int(width * frac)
        bar = "#" * filled + "-" * (width - filled)
        pct = f"{frac * 100:5.1f}%"
        mb_done = downloaded / 1_048_576
        mb_total = total / 1_048_576
        return f"[{bar}] {pct}  {mb_done:.1f}/{mb_total:.1f} MB"
    else:
        mb_done = downloaded / 1_048_576
        return f"[{'?' * width}]  {mb_done:.1f} MB downloaded"


def download_file(
    url: str,
    dest_path: Path,
    timeout: int = 60,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> bool:
    """Stream-download *url* to *dest_path* with progress display.

    Parameters
    ----------
    url:
        Remote URL to fetch.
    dest_path:
        Local filesystem path where the file will be saved.
    timeout:
        Per-request timeout in seconds.
    max_retries:
        Number of additional attempts after the first failure.
    retry_delay:
        Seconds to wait between retries (doubled each attempt).

    Returns
    -------
    bool
        ``True`` on success, ``False`` on all failures.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")

    attempt = 0
    delay = retry_delay

    while attempt <= max_retries:
        if attempt > 0:
            log.info("  Retry %d/%d in %.0f s …", attempt, max_retries, delay)
            time.sleep(delay)
            delay *= 2

        attempt += 1
        try:
            log.info("  GET %s", url)
            response = requests.get(url, stream=True, timeout=timeout)
            response.raise_for_status()

            total_bytes = response.headers.get("Content-Length")
            total_bytes = int(total_bytes) if total_bytes else None
            downloaded = 0

            with open(tmp_path, "wb") as fh:
                for chunk in response.iter_content(chunk_size=_CHUNK_SIZE):
                    if not chunk:
                        continue
                    fh.write(chunk)
                    downloaded += len(chunk)
                    bar = _progress_bar(downloaded, total_bytes)
                    print(f"\r  {bar}", end="", flush=True)

            print()  # newline after progress bar
            shutil.move(str(tmp_path), str(dest_path))
            log.info("  Saved → %s  (%.2f MB)", dest_path, dest_path.stat().st_size / 1_048_576)
            return True

        except requests.exceptions.Timeout:
            log.warning("  Timeout on attempt %d", attempt)
        except requests.exceptions.HTTPError as exc:
            log.warning("  HTTP error on attempt %d: %s", attempt, exc)
        except requests.exceptions.ConnectionError as exc:
            log.warning("  Connection error on attempt %d: %s", attempt, exc)
        except OSError as exc:
            log.error("  File-system error: %s", exc)
            break
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    return False


# ---------------------------------------------------------------------------
# Synthetic fallback generators
# ---------------------------------------------------------------------------

def _write_snap_fallback(dest_path: Path, n_nodes: int = 500, seed: int = 42) -> None:
    """Write a synthetic SNAP-format edge list as gzip fallback."""
    rng = random.Random(seed)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    edges: list[tuple[int, int]] = []
    # Erdős–Rényi-style sparse graph mimicking email network structure
    for u in range(n_nodes):
        # Each node sends mail to ~5-15 random others
        k = rng.randint(5, 15)
        neighbours = rng.sample(range(n_nodes), min(k, n_nodes - 1))
        for v in neighbours:
            if v != u:
                edges.append((u, v))

    header = (
        "# Synthetic fallback: SNAP email-EuAll format\n"
        f"# Nodes: {n_nodes}\tEdges: {len(edges)}\n"
        "# FromNodeId\tToNodeId\n"
    )

    with gzip.open(dest_path, "wt", encoding="utf-8") as fh:
        fh.write(header)
        for u, v in edges:
            fh.write(f"{u}\t{v}\n")

    log.info("  Synthetic SNAP fallback written: %s  (%d edges)", dest_path, len(edges))


def _write_rocketfuel_fallback(dest_path: Path, n_nodes: int = 80, seed: int = 42) -> None:
    """Write a synthetic Rocketfuel .cch-format file as gzip fallback."""
    rng = random.Random(seed)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate symbolic router names
    pops = ["SYD", "MEL", "BNE", "PER", "ADL", "AKL", "SIN", "HKG"]
    routers = [f"{rng.choice(pops)}-{rng.randint(1, 999):03d}" for _ in range(n_nodes)]

    edges = []
    # Connected backbone: ring + random extra links
    for i in range(n_nodes):
        j = (i + 1) % n_nodes
        weight = rng.randint(1, 100)
        edges.append((routers[i], routers[j], weight))

    extra = int(n_nodes * 1.5)
    for _ in range(extra):
        u, v = rng.sample(range(n_nodes), 2)
        weight = rng.randint(1, 100)
        edges.append((routers[u], routers[v], weight))

    with gzip.open(dest_path, "wt", encoding="utf-8") as fh:
        fh.write("# Synthetic fallback: Rocketfuel AS1221 CCH format\n")
        for src, dst, w in edges:
            # CCH format: src dst weight
            fh.write(f"{src} {dst} {w}\n")

    log.info("  Synthetic Rocketfuel fallback written: %s  (%d edges)", dest_path, len(edges))


def _write_caida_fallback(dest_path: Path, n_as: int = 200, seed: int = 42) -> None:
    """Write a synthetic CAIDA AS-relationship file as bz2 fallback."""
    import bz2

    rng = random.Random(seed)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # AS numbers in plausible ranges
    as_numbers = sorted(rng.sample(range(1, 65536), n_as))

    lines = []
    # Provider-customer (type -1) and peer-peer (type 0) relationships
    for i in range(len(as_numbers) - 1):
        rel_type = rng.choice([-1, 0])
        lines.append(f"{as_numbers[i]}|{as_numbers[i + 1]}|{rel_type}")

    extra = int(n_as * 2)
    for _ in range(extra):
        a, b = rng.sample(as_numbers, 2)
        rel_type = rng.choice([-1, 0])
        lines.append(f"{a}|{b}|{rel_type}")

    with bz2.open(dest_path, "wt", encoding="utf-8") as fh:
        fh.write("# Synthetic fallback: CAIDA AS relationships\n")
        fh.write("# format: <provider-as>|<customer-as>|<rel>\n")
        for line in lines:
            fh.write(line + "\n")

    log.info("  Synthetic CAIDA fallback written: %s  (%d relationships)", dest_path, len(lines))


_FALLBACK_WRITERS = {
    "snap": _write_snap_fallback,
    "rocketfuel": _write_rocketfuel_fallback,
    "caida": _write_caida_fallback,
}


# ---------------------------------------------------------------------------
# Main download orchestration
# ---------------------------------------------------------------------------

def download_all(data_root: str | Path = "data", timeout: int = 60) -> dict[str, bool]:
    """Download all datasets, falling back to synthetic data on failure.

    Parameters
    ----------
    data_root:
        Root directory under which per-dataset subdirectories are created.
    timeout:
        HTTP request timeout in seconds.

    Returns
    -------
    dict[str, bool]
        Mapping of dataset key → ``True`` if real data was downloaded,
        ``False`` if synthetic fallback was used.
    """
    data_root = Path(data_root)
    results: dict[str, bool] = {}

    for key, meta in DATASETS.items():
        log.info("=" * 60)
        log.info("Dataset : %s", meta["name"])
        log.info("Info    : %s", meta["description"])

        dest_dir = data_root / key
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / meta["filename"]

        if dest_path.exists():
            log.info("  Already present: %s — skipping download.", dest_path)
            results[key] = True
            continue

        success = download_file(meta["url"], dest_path, timeout=timeout)

        if success:
            results[key] = True
        else:
            log.warning("  Download failed for '%s'.  Generating synthetic fallback …", key)
            _FALLBACK_WRITERS[key](dest_path)
            results[key] = False

    log.info("=" * 60)
    log.info("Summary:")
    for key, real in results.items():
        status = "real data" if real else "synthetic fallback"
        log.info("  %-12s %s", key, status)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download network topology datasets for quantum routing experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-root",
        default="data",
        help="Root directory for dataset storage.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="HTTP request timeout in seconds.",
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()) + ["all"],
        default="all",
        help="Which dataset to download.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    data_root = Path(args.data_root)

    if args.dataset == "all":
        keys_to_run = list(DATASETS.keys())
    else:
        keys_to_run = [args.dataset]

    overall_ok = True
    for key in keys_to_run:
        meta = DATASETS[key]
        log.info("=" * 60)
        log.info("Dataset : %s", meta["name"])

        dest_dir = data_root / key
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / meta["filename"]

        if dest_path.exists():
            log.info("  Already present — skipping.")
            continue

        success = download_file(meta["url"], dest_path, timeout=args.timeout)
        if not success:
            log.warning("  Generating synthetic fallback for '%s' …", key)
            _FALLBACK_WRITERS[key](dest_path)
            overall_ok = False

    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
