"""CLI helper to download public failure-attribution benchmarks into an external directory.

Never downloads datasets into `/workspace` by default; defaults to `/tmp/fa-datasets`.
"""

from __future__ import annotations

import argparse
import io
import os
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetInfo:
    key: str
    title: str
    source_url: str
    license_name: str
    gated: bool
    citation: str
    kind: str  # "github_tar" | "direct_file"
    target_subdir: str
    filename: str = ""


DATASETS: dict[str, DatasetInfo] = {
    "whowhen": DatasetInfo(
        key="whowhen",
        title="Who&When (Agents_Failure_Attribution)",
        source_url="https://codeload.github.com/ag2ai/Agents_Failure_Attribution/tar.gz/refs/heads/main",
        license_name="MIT",
        gated=False,
        citation="Zhang et al., 'Which Agent Causes Task Failures and When? On Automated Failure Attribution of LLM Multi-Agent Systems', ICML 2025.",
        kind="github_tar",
        target_subdir="Agents_Failure_Attribution",
    ),
    "trail": DatasetInfo(
        key="trail",
        title="TRAIL (PatronusAI/trail-benchmark)",
        source_url="https://codeload.github.com/PatronusAI/trail-benchmark/tar.gz/refs/heads/main",
        license_name="MIT (GitHub) / CC-BY-4.0 (HF patronus-ai/trail-benchmark)",
        gated=False,
        citation="Deshpande et al., 'TRAIL: Trace Reasoning and Agentic Issue Localization', Patronus AI 2025.",
        kind="github_tar",
        target_subdir="trail-benchmark",
    ),
    "longrca-mini": DatasetInfo(
        key="longrca-mini",
        title="LongRCA-Mini (CLoud5-real/longrca-bench)",
        source_url="https://huggingface.co/datasets/CLoud5-real/longrca-bench/resolve/main/longrca-mini.parquet",
        license_name="Public HF dataset (see upstream benchmark licenses; no redistribution in repo)",
        gated=False,
        citation="LongRCA Bench, 'Diagnosing Responsible Roles and Root Causes in Long-Horizon Agent Failures', arXiv:2608.15242 (2026).",
        kind="direct_file",
        target_subdir="longrca",
        filename="longrca-mini.parquet",
    ),
    "longrca-full": DatasetInfo(
        key="longrca-full",
        title="LongRCA-Full (CLoud5-real/longrca-bench)",
        source_url="https://huggingface.co/datasets/CLoud5-real/longrca-bench/resolve/main/longrca-full.parquet",
        license_name="Public HF dataset (see upstream benchmark licenses; no redistribution in repo)",
        gated=False,
        citation="LongRCA Bench, 'Diagnosing Responsible Roles and Root Causes in Long-Horizon Agent Failures', arXiv:2608.15242 (2026).",
        kind="direct_file",
        target_subdir="longrca",
        filename="longrca-full.parquet",
    ),
    "agentrx": DatasetInfo(
        key="agentrx",
        title="AgentRx (microsoft/AgentRx)",
        source_url="https://codeload.github.com/microsoft/AgentRx/tar.gz/refs/heads/main",
        license_name="MIT",
        gated=False,
        citation="Barke et al., 'AgentRx: Diagnosing AI Agent Failures from Execution Trajectories', arXiv:2602.02475 (2026).",
        kind="github_tar",
        target_subdir="AgentRx",
    ),
}

DEFAULT_SUITE = ("whowhen", "trail", "longrca-mini", "agentrx")


def _safe_extract_tarball(raw_bytes: bytes, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(raw_bytes), mode="r:gz") as tf:
        for member in tf.getmembers():
            parts = Path(member.name).parts
            if len(parts) <= 1:
                continue
            rel = Path(*parts[1:])
            if ".." in rel.parts or rel.is_absolute():
                continue
            out_path = dest_dir / rel
            if member.isdir():
                out_path.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                out_path.parent.mkdir(parents=True, exist_ok=True)
                extracted = tf.extractfile(member)
                if extracted is not None:
                    out_path.write_bytes(extracted.read())


def download_dataset(name: str, dest_root: str | Path = "/tmp/fa-datasets", *, force: bool = False) -> Path:
    """Download one dataset by key into `dest_root/<target_subdir>` and return its local path."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}; choose from {sorted(DATASETS)}")
    info = DATASETS[name]
    root = Path(dest_root)
    target_dir = root / info.target_subdir

    headers = {"User-Agent": "tracepress-failure-attribution/0.1"}
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token and "huggingface.co" in info.source_url:
        headers["Authorization"] = f"Bearer {hf_token}"

    print(f"[{info.key}] {info.title}")
    print(f"  License : {info.license_name} (gated={info.gated})")
    print(f"  Citation: {info.citation}")

    if info.kind == "direct_file":
        out_file = target_dir / info.filename
        if out_file.exists() and not force:
            print(f"  Already present: {out_file}")
            return out_file
        target_dir.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(info.source_url, headers=headers)
        with urllib.request.urlopen(req, timeout=180) as resp:
            out_file.write_bytes(resp.read())
        print(f"  Saved to: {out_file}")
        return out_file

    if target_dir.exists() and any(target_dir.iterdir()) and not force:
        print(f"  Already present: {target_dir}")
        return target_dir

    req = urllib.request.Request(info.source_url, headers=headers)
    with urllib.request.urlopen(req, timeout=180) as resp:
        payload = resp.read()
    _safe_extract_tarball(payload, target_dir)
    print(f"  Extracted to: {target_dir}")
    return target_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download public failure-attribution datasets.")
    parser.add_argument(
        "--dataset",
        default="all",
        choices=["all", *sorted(DATASETS)],
        help="Dataset to download (default: all four core benchmarks).",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("/tmp/fa-datasets"),
        help="Destination directory outside the git repo (default: /tmp/fa-datasets).",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if destination exists.")
    args = parser.parse_args(argv)

    keys = list(DEFAULT_SUITE) if args.dataset == "all" else [args.dataset]
    for key in keys:
        download_dataset(key, args.dest, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
