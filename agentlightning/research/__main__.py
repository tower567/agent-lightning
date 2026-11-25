# Copyright (c) Microsoft. All rights reserved.

"""CPU-only entry point: initialize metadata, generate a tiny fixture, analyze events."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import analyze
from .events import initialize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="Create a new immutable metadata.json; never start an agent")
    init.add_argument("directory", type=Path)
    init.add_argument("--metadata", type=Path, required=True)
    demo = sub.add_parser("demo", help="Generate 14 small synthetic lifecycle examples, no LLM/GPU")
    demo.add_argument("directory", type=Path)
    analysis = sub.add_parser("analyze", help="Analyze one or more runs, including paired conditions")
    analysis.add_argument("directories", nargs="+", type=Path)
    analysis.add_argument("--output", type=Path, required=True)
    analysis.add_argument("--seed", type=int, default=42)
    analysis.add_argument("--bootstrap", type=int, default=1000)
    analysis.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if args.command == "init":
        initialize(args.directory, json.loads(args.metadata.read_text(encoding="utf-8")))
        print(args.directory / "metadata.json")
    elif args.command == "demo":
        from .demo import create

        create(args.directory)
        report = analyze([args.directory], args.directory / "analysis")
        print(json.dumps(report["counts"], indent=2))
    else:
        if args.bootstrap < 1:
            parser.error("--bootstrap must be positive")
        report = analyze(
            args.directories, args.output, seed=args.seed, n_bootstrap=args.bootstrap, plots=not args.no_plots
        )
        print(json.dumps(report["counts"], indent=2))


if __name__ == "__main__":
    main()
