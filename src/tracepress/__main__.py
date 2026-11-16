"""CLI helper: `python -m tracepress doctor` checks available AI coding subscriptions."""

from __future__ import annotations

import argparse
import json
import sys

from . import subscriptions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tracepress", description="TracePress subscription & environment helper.")
    sub = parser.add_subparsers(dest="command")
    doc = sub.add_parser("doctor", help="Check Claude Code, Codex and Cursor subscription backends")
    doc.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args(argv)

    if args.command in (None, "doctor"):
        info = subscriptions.doctor()
        if getattr(args, "json", False):
            print(json.dumps(info, indent=2))
            return 0
        rec = info["recommended"]
        print(f"Recommended subscription LM: {rec or 'none detected'}")
        for name, p in info["providers"].items():
            mark = "READY" if p["available"] else "MISSING"
            mode = f" ({p['mode']})" if p["mode"] else ""
            print(f"  [{mark:7s}] {name:12s}{mode:9s} -> {p['detail']}")
            if not p["available"]:
                print(f"            Hint: {p['login_hint']}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
